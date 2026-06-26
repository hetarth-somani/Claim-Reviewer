import time
import gc
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from PIL import Image

from src.config import (
    HF_VISION_MODEL_NAME, MAX_GPU_MEMORY_GB, CPU_OFFLOAD_GB,
    MAX_NEW_TOKENS_VISION, ALLOWED_PARTS, MAX_IMAGE_WORKERS
)
from src.utils import (
    logger, _quant_config, _dtype, _log_gpu_usage, parse_json, 
    resolve_image_path, extract_image_id, MAX_RETRIES, RETRY_BASE_DELAY
)

_STRICT_SYSTEM_PROMPT = """\
You are an expert insurance damage assessor. Your task is to examine the image and report only unmistakable physical damage.

Rules
- If you are not 100% sure that a mark, line, or irregularity is real damage, do not report it.
- Ignore reflections, shadows, dust, smudges, normal wear, or manufacturing tolerances.
- When in doubt, output an empty list.
- Only list damages that would be immediately obvious to a trained inspector.

Context
The customer claims a {claimed_issue} on the {claimed_part}.

Instructions
Look closely at the {claimed_part} area.
Decide if the image shows unambiguous damage on that part.

If yes, output a JSON list of damage entries.
If no, output an empty list [] for visible_damages.

Allowed damage types:
dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown

Response format
Respond with a JSON object ONLY (no preamble, no explanation):
{{
  "valid_image": true,
  "image_quality_issues": [],
  "object_visible": true,
  "visible_parts": [list of parts visible],
  "visible_damages": [
    {{
      "issue_type": "<damage_type from allowed list>",
      "object_part": "<part from allowed list: {allowed_parts}>",
      "severity": "unknown",
      "description": "<short description>"
    }}
  ],
  "authenticity_concerns": [],
  "anything_else_relevant": "none"
}}
Only the JSON, no extra text."""

_DEFAULT_EVIDENCE = {
    "valid_image":              False,
    "image_quality_issues":    ["blurry_image"],
    "object_visible":          False,
    "visible_parts":           [],
    "visible_damages":         [],
    "authenticity_concerns":   [],
    "anything_else_relevant":  "analysis failed",
}

class HfVisionModel:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        import time
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model_name = model_name or HF_VISION_MODEL_NAME

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(2) 

        _log_gpu_usage("before vision model load")
        logger.info("Loading vision model: %s …", self.model_name)

        quant_cfg = _quant_config()
        n_gpus  = torch.cuda.device_count() if torch.cuda.is_available() else 0
        max_mem = {i: f"{MAX_GPU_MEMORY_GB}GiB" for i in range(n_gpus)} if n_gpus > 0 else {}
        max_mem["cpu"] = f"{CPU_OFFLOAD_GB}GiB"
        
        load_kw = dict(
            torch_dtype=_dtype(),
            device_map="sequential" if torch.cuda.is_available() else "cpu",
            max_memory=max_mem if max_mem else None,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa" if torch.cuda.is_available() else None,
        )
        if quant_cfg:
            load_kw["quantization_config"] = quant_cfg

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, **load_kw
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=128 * 28 * 28,
            max_pixels=128 * 28 * 28,
        )

        _log_gpu_usage("after vision model load")
        logger.info("Vision model ready: %s", self.model_name)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image: Image.Image,
        max_new_tokens: int = MAX_NEW_TOKENS_VISION,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except ImportError:
            image_inputs = [image]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        _first_device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, return_tensors="pt"
        ).to(_first_device)

        def _run():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            decoded = [
                self.processor.decode(
                    out[len(inp):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for inp, out in zip(inputs["input_ids"], out_ids)
            ]
            return decoded[0].strip()

        try:
            return _run()
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM on vision model generate — clearing cache and retrying once …")
            gc.collect()
            torch.cuda.empty_cache()
            return _run()


def analyze_image(vision_model, image_path: str, claim_object: str, claimed_issue: str = "unknown", claimed_part: str = "unknown") -> dict:
    allowed_parts = ", ".join(ALLOWED_PARTS.get(claim_object, ["unknown"]))
    
    prompt = _STRICT_SYSTEM_PROMPT.format(
        claimed_issue=claimed_issue,
        claimed_part=claimed_part,
        allowed_parts=allowed_parts
    )

    full_path = resolve_image_path(image_path)
    parsed = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            img    = Image.open(full_path).convert("RGB")
            raw    = vision_model.generate(prompt, img, max_new_tokens=MAX_NEW_TOKENS_VISION)
            parsed = parse_json(raw)
            for k, v in _DEFAULT_EVIDENCE.items():
                parsed.setdefault(k, v)
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                logger.error("[Stage 2] ✗ All retries exhausted for %s: %s", image_path, exc)
                parsed = _DEFAULT_EVIDENCE.copy()
            else:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)

    parsed["_image_id"] = extract_image_id(image_path)
    return parsed


def analyze_all_images(
    vision_model, image_paths: List[str], claim_object: str, claimed_issue: str = "unknown", claimed_part: str = "unknown"
) -> List[dict]:
    if not image_paths:
        return []

    results = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
        future_to_idx = {
            executor.submit(analyze_image, vision_model, p, claim_object, claimed_issue, claimed_part): i
            for i, p in enumerate(image_paths)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("[Stage 2] Thread failed for image[%d]: %s", idx, exc)
                fallback = _DEFAULT_EVIDENCE.copy()
                fallback["_image_id"] = extract_image_id(image_paths[idx])
                results[idx] = fallback

    return results
