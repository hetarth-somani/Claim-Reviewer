import gc
import json
import logging
import os
import re
import time
from functools import wraps
from typing import Callable, List
from pathlib import Path

import torch
from json_repair import repair_json

from src.config import DATASET_DIR, USE_4BIT_QUANT

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger("pipeline")

# Reduce memory fragmentation
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ══════════════════════════════════════════════════════════════════════════════
#  GPU MEMORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def free_gpu_memory(*objects):
    for obj in objects:
        if obj is None:
            continue
        for attr in ("model", "processor", "tokenizer"):
            sub = getattr(obj, attr, None)
            if sub is not None:
                del sub
        del obj

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb  = torch.cuda.memory_reserved()  / 1024**3
        logger.info(
            "GPU memory after free — allocated: %.2f GB | reserved: %.2f GB",
            allocated_gb, reserved_gb,
        )

def _log_gpu_usage(label: str = ""):
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        total_gb     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("VRAM %s: %.2f / %.2f GB used", label, allocated_gb, total_gb)

def _quant_config():
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU — 4-bit quant skipped (CPU mode).")
        return None
    if not USE_4BIT_QUANT:
        return None
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info("4-bit NF4 quantisation enabled (compute dtype: float16).")
        return cfg
    except Exception as exc:
        raise RuntimeError(f"bitsandbytes 4-bit quantisation FAILED: {exc}") from exc

def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
class JSONParseError(Exception):
    pass

def parse_json(text: str) -> dict:
    cleaned = re.sub(r'```(?:json)?\s*', '', text.strip()).strip()
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'```\s*$', '', cleaned).strip()

    for strategy in (
        lambda s: json.loads(s),
        lambda s: json.loads(repair_json(s)),
    ):
        try:
            result = strategy(cleaned)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(repair_json(match.group(0)))
        except Exception:
            pass

    raise JSONParseError(f"Could not parse JSON from: {text[:300]}")

def retry(max_attempts: int = 3, base_delay: float = 1.0):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "[retry] %s — attempt %d/%d failed: %s. Retrying in %.1fs …",
                            func.__name__, attempt, max_attempts, exc, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "[retry] %s — all %d attempts exhausted. Last error: %s",
                            func.__name__, max_attempts, exc,
                        )
            raise last_exc
        return wrapper
    return decorator

def split_image_paths(paths_str: str) -> List[str]:
    if not paths_str:
        return []
    return [p.strip() for p in paths_str.split(";") if p.strip()]

def resolve_image_path(relative_path: str) -> Path:
    return (DATASET_DIR / relative_path).resolve()

def extract_image_id(image_path: str) -> str:
    return Path(image_path).stem
