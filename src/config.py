import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL NAMES
# ══════════════════════════════════════════════════════════════════════════════
TEXT_MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"   # Fallback: Qwen/Qwen2.5-3B-Instruct
HF_VISION_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # Fallback: Qwen/Qwen2-VL-2B-Instruct

# ══════════════════════════════════════════════════════════════════════════════
#  DEVICE / QUANTISATION
# ══════════════════════════════════════════════════════════════════════════════
DEVICE         = "cuda"   # explicit — avoids device_map="auto" splitting issues
USE_4BIT_QUANT = True     # MANDATORY for 7B models on 16GB GPUs

# ══════════════════════════════════════════════════════════════════════════════
#  FILESYSTEM PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).parent.parent.resolve()
DATASET_DIR  = BASE_DIR / "data"
CLAIMS_CSV   = DATASET_DIR / "sample_claims.csv"
OUTPUT_CSV   = BASE_DIR / "output.csv"
EVIDENCE_CSV = DATASET_DIR / "evidence_requirements.csv"
HISTORY_CSV  = DATASET_DIR / "user_history.csv"

# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION / RETRY SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
MAX_RETRIES           = 3
RETRY_BASE_DELAY      = 1.0
MAX_NEW_TOKENS_TEXT   = 384
MAX_NEW_TOKENS_VISION = 512
MAX_NEW_TOKENS_FUSION = 512
TEMPERATURE           = 0.1
MAX_IMAGE_WORKERS     = 1

MAX_GPU_MEMORY_GB  = 13        
CPU_OFFLOAD_GB     = 20        

USE_HF_TEXT   = True
USE_HF_VISION = True

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part",
    "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

ALLOWED_PARTS = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    ],
}
ALLOWED_ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
]
ALLOWED_SEVERITIES  = ["none", "low", "medium", "high", "unknown"]
ALLOWED_RISK_FLAGS  = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]

# ══════════════════════════════════════════════════════════════════════════════
#  ONTOLOGY MAP
# ══════════════════════════════════════════════════════════════════════════════
ONTOLOGY_MAP = {
    "broken_part": {"crack", "glass_shatter", "dent", "broken_part", "crushed_packaging", "torn_packaging"},
    "missing_part": {"missing_part"},
    "crack": {"glass_shatter", "crack", "broken_part"},
    "glass_shatter": {"crack", "glass_shatter", "broken_part"},
    "dent": {"dent", "crushed_packaging", "scratch"},
    "scratch": {"scratch", "dent"},
    "torn_packaging": {"torn_packaging", "crushed_packaging", "broken_part", "crack"},
    "crushed_packaging": {"crushed_packaging", "torn_packaging", "dent", "broken_part"},
    "water_damage": {"water_damage", "stain"},
    "stain": {"stain", "water_damage"},
    "none": {"none"},
    "unknown": set(ALLOWED_ISSUE_TYPES)
}
