import csv
import re
from pathlib import Path

from src.config import OUTPUT_CSV, CLAIMS_CSV, ALLOWED_ISSUE_TYPES

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def extract_sim_score(justification):
    match = re.search(r"Semantic Match: ([0-9.]+)", justification)
    if match:
        return match.group(1)
    return "N/A"

def extract_reported_damage(justification):
    match = re.search(r"Found '([^']+)'", justification)
    if match:
        return match.group(1)
    return "Unknown/None"

def get_adjudicator_branch(justification):
    if "Ontology Match" in justification:
        return "ontology"
    if "Semantic Match" in justification:
        return "semantic"
    if "Lenient" in justification:
        return "lenient"
    if "Cannot Assess" in justification:
        return "not_visible"
    if "Contradicted" in justification:
        return "contradicted"
    return "unknown"

def evaluate(predictions_csv: str = None, ground_truth_csv: str = None) -> dict:
    pred_path = Path(predictions_csv) if predictions_csv else OUTPUT_CSV
    truth_path = Path(ground_truth_csv) if ground_truth_csv else CLAIMS_CSV
    
    if not pred_path.exists() or not truth_path.exists():
        print(f"Error: Could not find {pred_path.resolve()} or {truth_path.resolve()}")
        return {}

    preds = {row["user_id"]: row for row in load_csv(pred_path)}
    truths = {row["user_id"]: row for row in load_csv(truth_path)}
    
    buckets = {
        "A": 0, # Part name mismatch
        "B": 0, # Damage vocab mismatch
        "C": 0, # Vision model missed damage
        "D": 0, # Stage 1 hallucination
        "E": 0, # Ground truth vocabulary mismatch
        "F": 0, # Evidence requirements not met
        "Other": 0
    }

    print("=" * 100)
    print("  CLASSIFICATION ERROR ANALYSIS")
    print("=" * 100)
    
    mismatch_count = 0
    mismatches = {}
    
    for uid, truth in truths.items():
        if uid not in preds:
            continue
            
        pred = preds[uid]
        pred_status = pred.get("claim_status", "").lower()
        truth_status = truth.get("claim_status", "").lower()
        
        if pred_status != truth_status:
            mismatch_count += 1
            justification = pred.get("claim_status_justification", "")
            
            branch = get_adjudicator_branch(justification)
            sim_score = extract_sim_score(justification)
            reported_damage = extract_reported_damage(justification)
            ontology_match = "True" if branch == "ontology" else "False"
            
            pred_issue = pred.get("issue_type", "")
            pred_part = pred.get("object_part", "")
            truth_issue = truth.get("issue_type", "")
            truth_part = truth.get("object_part", "")
            
            bucket = "Other"
            if truth_issue not in ALLOWED_ISSUE_TYPES:
                bucket = "E"
            elif pred_issue != truth_issue or pred_part != truth_part:
                bucket = "D"
            elif branch == "not_visible" and truth_status == "supported":
                bucket = "A"
            elif branch == "contradicted" and truth_status == "supported":
                bucket = "C"
            elif branch in ("semantic", "lenient") and truth_status == "contradicted":
                bucket = "B"
            elif branch == "not_visible" and truth_status == "supported":
                bucket = "F"
                
            if bucket == "Other":
                if truth_status == "supported" and pred_status == "not_enough_information":
                    bucket = "F"
                    
            buckets[bucket] += 1
            mismatches[uid] = bucket
            
            print(f"\n[MISMATCH] user_id: {uid} | Bucket: {bucket}")
            print(f"  Stage 1 Raw Claim : {truth.get('user_claim', '')}")
            print(f"  Stage 1 Extracted : issue='{pred_issue}', part='{pred_part}'")
            print(f"  Ground Truth Exp. : issue='{truth_issue}', part='{truth_part}'")
            print(f"  Vision Damages    : {reported_damage} (extracted from justification)")
            print(f"  Ontology Match    : {ontology_match}")
            print(f"  Highest Sim Score : {sim_score}")
            print(f"  Adjudicator Branch: {branch}")
            print(f"  Final Output      : {pred_status}")
            print(f"  Ground Truth      : {truth_status}")
            
    print("\n" + "=" * 100)
    print("  BUCKET SUMMARY")
    print("=" * 100)
    print(f"Total Mismatches Analyzed: {mismatch_count}")
    print(f"  [A] Part name mismatch                         : {buckets['A']}")
    print(f"  [B] Damage vocab mismatch not in ontology      : {buckets['B']}")
    print(f"  [C] Vision model missed damage entirely        : {buckets['C']}")
    print(f"  [D] Stage 1 extracted wrong issue/part         : {buckets['D']}")
    print(f"  [E] Ground truth has different allowed vocab   : {buckets['E']}")
    print(f"  [F] Evidence requirements not met              : {buckets['F']}")
    print(f"  [Other] Unclassified mismatches                : {buckets['Other']}")
    print("=" * 100)
    
    return mismatches
