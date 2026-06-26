from src.pipeline import run_pipeline
from src.evaluator import evaluate
from src.config import TEXT_MODEL_NAME, HF_VISION_MODEL_NAME, CLAIMS_CSV, OUTPUT_CSV

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║    Multi-Modal Evidence Reviewer  —  Pipeline                ║
║  Sequential batch loading — one model in VRAM at a time      ║
║  Phase 1: Stage 1 all rows  (text model)                     ║
║  Phase 2: Stage 2 all rows  (vision model)                   ║
║  Phase 3: Stage 3 all rows  (text model/rules)               ║
╚══════════════════════════════════════════════════════════════╝
""")
    print(f"  Text model   : {TEXT_MODEL_NAME}")
    print(f"  Vision model : {HF_VISION_MODEL_NAME}")
    print(f"  Claims CSV   : {CLAIMS_CSV}")
    print(f"  Output CSV   : {OUTPUT_CSV}")
    print()

    # Note: Running this pipeline will load large local models which requires 
    # significant VRAM (e.g. 15GB+ for 7B models in 4-bit quantisation). 
    # Please refer to the README.md for hardware requirements before executing.
    
    # output_path = run_pipeline()
    # print(f"\n✓ Done. Output written to: {output_path}")
    
    # print("\nStarting Error Analysis...")
    # evaluate(predictions_csv=output_path, ground_truth_csv=CLAIMS_CSV)

if __name__ == "__main__":
    main()
