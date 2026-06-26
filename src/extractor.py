import gc
import json
import torch
from typing import Optional

from src.config import (
    TEXT_MODEL_NAME, MAX_GPU_MEMORY_GB, CPU_OFFLOAD_GB,
    MAX_NEW_TOKENS_TEXT, TEMPERATURE, ALLOWED_PARTS
)
from src.utils import logger, _quant_config, _dtype, _log_gpu_usage, parse_json, retry, MAX_RETRIES, RETRY_BASE_DELAY

_STAGE1_SYSTEM_PROMPT_PASS1 = """\
You are a claim extraction assistant. Read the customer support chat and extract the exact damage claim.

Output a strict JSON object with ONLY these fields (no extra text or explanation):
- "claim_summary": one-sentence description.
- "claimed_issue_type": pick ONLY from: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown.
- "claimed_object_part": the part mentioned, from the allowed list for {claim_object}: {allowed_parts}. Use "unknown" if unclear.
- "claimed_severity": none, low, medium, high, unknown.
- "extraction_confidence_score": integer from 1 to 5 indicating your certainty (5 is highest).
- "is_claim_ambiguous": true if vague/contradictory, else false.
- "any_extra_instructions": extra requests (or "none").

Example output for a car claim:
{{
  "claim_summary": "User reports a large dent on the front bumper",
  "claimed_issue_type": "dent",
  "claimed_object_part": "front_bumper",
  "claimed_severity": "medium",
  "extraction_confidence_score": 5,
  "is_claim_ambiguous": false,
  "any_extra_instructions": "none"
}}

claim_object: {claim_object}
user_claim: "{user_claim}"

Output ONLY the JSON object, no preamble."""

_STAGE1_SYSTEM_PROMPT_PASS2 = """\
You are a claim verification assistant. Review the following customer support chat and the claim extracted from it by an automated system.

Chat Transcript:
"{user_claim}"

Extracted Claim JSON:
{pass1_json}

If anything is factually wrong, misinterpreted, or missing in the extracted claim based on the transcript, correct it. Ensure you keep the exact same JSON format with the same keys (including extraction_confidence_score). If it is already correct, output the exact same JSON.
Output ONLY the JSON object, no preamble."""

_DEFAULT_EXTRACTION = {
    "claim_summary":          "extraction failed",
    "claimed_issue_type":     "unknown",
    "claimed_object_part":    "unknown",
    "claimed_severity":       "unknown",
    "extraction_confidence_score": 1,
    "is_claim_ambiguous":     True,
    "any_extra_instructions": "none",
}

class HfTextModel:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or TEXT_MODEL_NAME
        _log_gpu_usage("before text model load")
        logger.info("Loading text model: %s …", self.model_name)

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kw)
        self.model.eval()

        _log_gpu_usage("after text model load")
        logger.info("Text model ready: %s", self.model_name)

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = MAX_NEW_TOKENS_TEXT,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        _first_device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).to(_first_device)

        def _run():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = out_ids[0, inputs["input_ids"].shape[-1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        try:
            return _run()
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM on text model generate — clearing cache and retrying once …")
            gc.collect()
            torch.cuda.empty_cache()
            return _run()


@retry(max_attempts=MAX_RETRIES, base_delay=RETRY_BASE_DELAY)
def _call_extraction(model, messages) -> str:
    return model.generate(messages, max_new_tokens=MAX_NEW_TOKENS_TEXT)

def extract_claim(text_model, user_claim: str, claim_object: str) -> dict:
    allowed = ", ".join(ALLOWED_PARTS.get(claim_object, ["unknown"]))
    
    # PASS 1
    system_prompt_p1 = _STAGE1_SYSTEM_PROMPT_PASS1.format(
        claim_object=claim_object,
        allowed_parts=allowed,
        user_claim=user_claim,
    )
    messages_p1 = [
        {"role": "system", "content": system_prompt_p1},
        {"role": "user",   "content": "Please extract the claim now."},
    ]

    try:
        raw_p1 = _call_extraction(text_model, messages_p1)
        parsed_p1 = parse_json(raw_p1)
        for k, v in _DEFAULT_EXTRACTION.items():
            parsed_p1.setdefault(k, v)
    except Exception as exc:
        logger.error("Stage 1 Pass 1 failed for claim: '%s'. Error: %s", user_claim[:80], exc)
        return _DEFAULT_EXTRACTION.copy()

    # PASS 2
    system_prompt_p2 = _STAGE1_SYSTEM_PROMPT_PASS2.format(
        user_claim=user_claim,
        pass1_json=json.dumps(parsed_p1, indent=2)
    )
    messages_p2 = [
        {"role": "system", "content": system_prompt_p2},
        {"role": "user",   "content": "Please verify and output the corrected JSON."},
    ]
    
    try:
        raw_p2 = _call_extraction(text_model, messages_p2)
        parsed_p2 = parse_json(raw_p2)
        for k, v in _DEFAULT_EXTRACTION.items():
            parsed_p2.setdefault(k, v)
            
        p1_issue = parsed_p1.get("claimed_issue_type")
        p1_part = parsed_p1.get("claimed_object_part")
        p2_issue = parsed_p2.get("claimed_issue_type")
        p2_part = parsed_p2.get("claimed_object_part")
        
        conf = parsed_p1.get("extraction_confidence_score", 5)
        if p1_issue == p2_issue and p1_part == p2_part and conf <= 2:
            parsed_p2["is_claim_ambiguous"] = True
            
        uncertain_phrases = ["not sure", "may be", "maybe", "think", "possibly", "not fully sure"]
        lower_claim = user_claim.lower()
        if any(phrase in lower_claim for phrase in uncertain_phrases):
            if parsed_p2.get("claimed_issue_type") != "unknown":
                parsed_p2["claimed_issue_type"] = "unknown"
                parsed_p2["extraction_confidence_score"] = 1
                parsed_p2["is_claim_ambiguous"] = True

        return parsed_p2
        
    except Exception as exc:
        logger.warning("Stage 1 Pass 2 failed, falling back to Pass 1. Error: %s", exc)
        return parsed_p1
