from sentence_transformers import SentenceTransformer, util

from src.utils import logger
from src.config import (
    ALLOWED_ISSUE_TYPES, ALLOWED_SEVERITIES, ALLOWED_PARTS, 
    ALLOWED_RISK_FLAGS, ONTOLOGY_MAP
)

_EMBED_MODEL = None
def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info("Loading sentence-transformers model...")
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL

def fuse_decision_rules(
    row: dict,
    extracted_claim: dict,
    image_evidences: list,
    user_history_row: dict | None,
    evidence_reqs_rows: list,
) -> dict:
    uid = row.get("user_id", "")
    claim_obj = row.get("claim_object", "").strip().lower()
    
    claimed_issue = str(extracted_claim.get("claimed_issue_type", "unknown")).lower()
    if claimed_issue not in ALLOWED_ISSUE_TYPES:
        claimed_issue = "unknown"
        
    claimed_part = str(extracted_claim.get("claimed_object_part", "unknown")).lower()
    allowed_parts_set = set(ALLOWED_PARTS.get(claim_obj, []))
    if allowed_parts_set and claimed_part not in allowed_parts_set:
        claimed_part = "unknown"

    risk_flags_set = set()
    
    for img in image_evidences:
        for q in img.get("image_quality_issues", []):
            if q and str(q).lower() != "none":
                risk_flags_set.add(str(q).lower())
        for a in img.get("authenticity_concerns", []):
            if a and str(a).lower() != "none":
                risk_flags_set.add(str(a).lower())

    if user_history_row:
        fraud_str = str(user_history_row.get("prior_fraud_flag", "")).lower()
        if fraud_str in ("true", "1", "yes"):
            risk_flags_set.add("user_history_risk")
            risk_flags_set.add("manual_review_required")

    if claimed_issue in ("missing_part", "missing_item", "unknown") and claimed_part in ("contents", "item", "product"):
        all_d = []
        for img in image_evidences:
            for d in img.get("visible_damages", []):
                all_d.append(d)
        if not all_d:
            justification = f"Cannot Assess. Claimed issue '{claimed_issue}' on '{claimed_part}', but no physical damage or evidence of tampering could be visually confirmed."
            output_row = {
                "user_id": uid,
                "image_paths": row.get("image_paths", ""),
                "user_claim": row.get("user_claim", ""),
                "claim_object": claim_obj,
                "evidence_standard_met": False,
                "evidence_standard_met_reason": justification,
                "risk_flags": ";".join(sorted([f for f in risk_flags_set if f in ALLOWED_RISK_FLAGS])) if risk_flags_set else "none",
                "issue_type": claimed_issue,
                "object_part": claimed_part,
                "claim_status": "not_enough_information",
                "claim_status_justification": justification,
                "supporting_image_ids": "none",
                "valid_image": sum(1 for img in image_evidences if img.get("valid_image", False)) > 0,
                "severity": "unknown",
            }
            return output_row

    all_damages = []
    valid_images_count = sum(1 for img in image_evidences if img.get("valid_image", False))
    valid_image_overall = valid_images_count > 0

    for img in image_evidences:
        iid = img.get("_image_id", "unknown_id")
        for dmg in img.get("visible_damages", []):
            dmg_copy = dmg.copy()
            dmg_copy["_image_id"] = iid
            all_damages.append(dmg_copy)

    part_visible = False
    supporting_img_ids = set()
    
    for img in image_evidences:
        visible_parts = [str(p).lower() for p in img.get("visible_parts", [])]
        if (claimed_part in visible_parts or 
            "body" in visible_parts or 
            "box" in visible_parts or 
            img.get("object_visible", False)):
            part_visible = True
            iid = img.get("_image_id")
            if iid:
                supporting_img_ids.add(iid)

    if not part_visible:
        claim_status = "not_enough_information"
        severity = "unknown"
        evidence_standard_met = False
        justification = f"Cannot Assess. Claimed part '{claimed_part}' is not visible in any image."
        supporting_img_ids.clear()
    else:
        part_damages = []
        for d in all_damages:
            dmg_part = str(d.get("object_part", "")).lower()
            if dmg_part == claimed_part or claimed_part == "unknown" or dmg_part in ("body", "box", "unknown", "whole_object"):
                part_damages.append(d)

        evidence_standard_met = True
        
        if not part_damages:
            if claimed_issue in ('missing_part', 'missing_item', 'unknown') and not all_damages:
                claim_status = "not_enough_information"
                severity = "none"
                justification = f"Cannot Assess. Claimed issue is '{claimed_issue}' but no physical damage or missing evidence could be visually confirmed."
            else:
                claim_status = "contradicted"
                severity = "none"
                justification = f"Contradicted. Claimed part '{claimed_part}' is visible but no damages were reported on it."
                risk_flags_set.add("claim_mismatch")
        else:
            matched = False
            best_match_dmg = None
            for d in part_damages:
                d_type = str(d.get("issue_type", "")).lower()
                if d_type in ONTOLOGY_MAP.get(claimed_issue, []):
                    matched = True
                    best_match_dmg = d
                    break
            
            if matched:
                claim_status = "supported"
                severity = str(best_match_dmg.get("severity", "unknown")).lower()
                if severity == "none":
                    severity = "unknown"
                justification = f"Supported (Ontology Match). Found '{best_match_dmg.get('issue_type')}' on '{claimed_part}' which supports claimed '{claimed_issue}'."
                if best_match_dmg.get("_image_id"):
                    supporting_img_ids.add(best_match_dmg["_image_id"])
            else:
                model = get_embed_model()
                claim_vec = model.encode(claimed_issue)
                
                best_sim = -1.0
                closest_dmg = None
                
                for d in part_damages:
                    d_type = str(d.get("issue_type", "")).lower()
                    d_vec = model.encode(d_type)
                    sim = float(util.cos_sim(claim_vec, d_vec)[0][0])
                    if sim > best_sim:
                        best_sim = sim
                        closest_dmg = d

                SIM_THRESHOLD = 0.60
                if closest_dmg and best_sim >= SIM_THRESHOLD:
                    claim_status = "supported"
                    severity = str(closest_dmg.get("severity", "unknown")).lower()
                    if severity == "none":
                        severity = "unknown"
                    justification = f"Supported (Semantic Match: {best_sim:.2f}). Found '{closest_dmg.get('issue_type')}' which conceptually matches '{claimed_issue}'."
                    if closest_dmg.get("_image_id"):
                        supporting_img_ids.add(closest_dmg["_image_id"])
                else:
                    claim_status = "supported"
                    severity = "unknown" 
                    fallback_dmg = part_damages[0]
                    justification = f"Supported (Lenient). Found '{fallback_dmg.get('issue_type')}' on the claimed part. Accepting as evidence for '{claimed_issue}'."
                    if fallback_dmg.get("_image_id"):
                        supporting_img_ids.add(fallback_dmg["_image_id"])

    if claim_status in ("contradicted", "not_enough_information"):
        severity = "none"
        
    if evidence_standard_met and supporting_img_ids:
        sup_ids_str = ";".join(sorted(list(set(supporting_img_ids))))
    else:
        sup_ids_str = "none"

    valid_flags = [f for f in risk_flags_set if f in ALLOWED_RISK_FLAGS]
    if not valid_flags:
        risk_flags_str = "none"
    else:
        risk_flags_str = ";".join(sorted(valid_flags))

    output_row = {
        "user_id": uid,
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": claim_obj,
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": justification,
        "risk_flags": risk_flags_str,
        "issue_type": claimed_issue,
        "object_part": claimed_part,
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": sup_ids_str,
        "valid_image": valid_image_overall,
        "severity": severity if severity in ALLOWED_SEVERITIES else "unknown",
    }
    
    return output_row
