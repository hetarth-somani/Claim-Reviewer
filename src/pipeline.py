import csv
import time
from pathlib import Path

from src.config import (
    CLAIMS_CSV, OUTPUT_CSV, HISTORY_CSV, EVIDENCE_CSV,
    TEXT_MODEL_NAME, HF_VISION_MODEL_NAME, OUTPUT_COLUMNS
)
from src.utils import logger, free_gpu_memory, split_image_paths
from src.data_loader import load_claims, load_user_history, load_evidence_requirements, filter_evidence_reqs
from src.extractor import HfTextModel, extract_claim
from src.image_analyzer import HfVisionModel, analyze_all_images
from src.adjudicator import fuse_decision_rules

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"

def _sep(char: str = "─", width: int = 70) -> None:
    logger.info(char * width)

def run_pipeline(
    claims_csv:        Path | str | None = None,
    output_csv:        Path | str | None = None,
    text_model_name:   str | None = None,
    vision_model_name: str | None = None,
) -> Path:
    claims_csv = Path(claims_csv) if claims_csv else CLAIMS_CSV
    output_csv = Path(output_csv) if output_csv else OUTPUT_CSV
    t_name = text_model_name   or TEXT_MODEL_NAME
    v_name = vision_model_name or HF_VISION_MODEL_NAME

    logger.info("Loading claims from: %s", claims_csv)
    claims = load_claims(claims_csv)
    total  = len(claims)
    if total == 0:
        raise ValueError(f"No claims found in {claims_csv}")
    logger.info("Loaded %d claims.", total)

    user_history  = load_user_history(HISTORY_CSV)
    evidence_reqs = load_evidence_requirements(EVIDENCE_CSV)
    logger.info(
        "Loaded user_history (%d users) | evidence_requirements (%d rows).",
        len(user_history), len(evidence_reqs),
    )

    pipeline_start = time.perf_counter()

    # PHASE 1
    _sep("═")
    logger.info("PHASE 1 — Stage 1: Claim extraction  [text model: %s]", t_name)
    _sep("═")

    text_model = HfTextModel(model_name=t_name)
    extracted_claims: list[dict] = []
    phase1_start = time.perf_counter()

    for i, row in enumerate(claims):
        uid       = row.get("user_id", f"row_{i}")
        claim_obj = row.get("claim_object", "").strip()
        logger.info("[%d/%d] Stage 1 — user_id=%s", i + 1, total, uid)

        result = extract_claim(text_model, row.get("user_claim", ""), claim_obj)
        extracted_claims.append(result)

        logger.info(
            "  issue_type=%s | part=%s | ambiguous=%s",
            result.get("claimed_issue_type"),
            result.get("claimed_object_part"),
            result.get("is_claim_ambiguous"),
        )

    logger.info(
        "Phase 1 complete in %s.", _fmt_time(time.perf_counter() - phase1_start)
    )

    logger.info("Freeing text model from GPU …")
    free_gpu_memory(text_model)
    text_model = None

    # PHASE 2
    _sep("═")
    logger.info("PHASE 2 — Stage 2: Image analysis  [vision model: %s]", v_name)
    _sep("═")

    vision_model = HfVisionModel(model_name=v_name)
    all_image_evidences: list[list[dict]] = []
    phase2_start = time.perf_counter()

    for i, row in enumerate(claims):
        uid        = row.get("user_id", f"row_{i}")
        claim_obj  = row.get("claim_object", "").strip()
        paths_str  = row.get("image_paths", "")
        image_paths = split_image_paths(paths_str)

        logger.info(
            "[%d/%d] Stage 2 — user_id=%s | images=%d",
            i + 1, total, uid, len(image_paths),
        )
        claimed_issue = extracted_claims[i].get("claimed_issue_type", "unknown")
        claimed_part = extracted_claims[i].get("claimed_object_part", "unknown")
        evidences   = analyze_all_images(vision_model, image_paths, claim_obj, claimed_issue, claimed_part)
        all_image_evidences.append(evidences)
        valid_count = sum(1 for ev in evidences if ev.get("valid_image"))
        logger.info("  %d/%d images valid", valid_count, len(evidences))

    logger.info(
        "Phase 2 complete in %s.", _fmt_time(time.perf_counter() - phase2_start)
    )

    logger.info("Freeing vision model from GPU …")
    free_gpu_memory(vision_model)
    vision_model = None

    # PHASE 3
    _sep("═")
    logger.info("PHASE 3 — Stage 3: Decision fusion  [Rules Engine]")
    _sep("═")

    phase3_start = time.perf_counter()
    row_times: list[float] = []

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(output_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    try:
        for i, row in enumerate(claims):
            row_start = time.perf_counter()
            uid       = row.get("user_id", f"row_{i}")
            claim_obj = row.get("claim_object", "").strip()

            _sep()
            logger.info("[%d/%d] Stage 3 — user_id=%s", i + 1, total, uid)

            hist_row = user_history.get(uid)
            req_rows = filter_evidence_reqs(evidence_reqs, claim_obj)

            output_row = fuse_decision_rules(
                row,
                extracted_claims[i],
                all_image_evidences[i],
                hist_row,
                req_rows,
            )
            logger.info(
                "  claim_status=%-25s  severity=%-8s  flags=%s",
                output_row.get("claim_status"),
                output_row.get("severity"),
                output_row.get("risk_flags"),
            )

            writer.writerow(output_row)
            out_fh.flush()

            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            row_elapsed = time.perf_counter() - row_start
            row_times.append(row_elapsed)
            eta = (sum(row_times) / len(row_times)) * (total - i - 1)
            logger.info(
                "  ✓ Row written in %s | ETA: %s",
                _fmt_time(row_elapsed), _fmt_time(eta),
            )

    finally:
        out_fh.close()

    logger.info(
        "Phase 3 complete in %s.", _fmt_time(time.perf_counter() - phase3_start)
    )

    total_elapsed = time.perf_counter() - pipeline_start
    _sep("═")
    logger.info("Pipeline complete.")
    logger.info("  Rows processed : %d / %d", len(row_times), total)
    logger.info("  Output CSV     : %s", output_csv)
    logger.info("  Total time     : %s", _fmt_time(total_elapsed))
    if row_times:
        logger.info(
            "  Avg per row    : %s", _fmt_time(sum(row_times) / len(row_times))
        )
    _sep("═")

    return output_csv
