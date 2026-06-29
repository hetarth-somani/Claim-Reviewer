# Multi-Modal Evidence Reviewer

A damage claim verification system that uses a combination of local LLM extraction, a vision-language model (VLM) for image analysis, and a deterministic rules engine for final adjudication.

## Overview

This pipeline verifies customer damage claims against submitted evidence (images). It operates in three sequential stages to optimize GPU memory (VRAM) usage:

1. **Stage 1: LLM Claim Extraction (`src/extractor.py`)**  
   Extracts a structured JSON claim from the customer support chat transcript.
2. **Stage 2: VLM Image Evidence Gathering (`src/image_analyzer.py`)**  
   Analyzes images with a Vision-Language Model to identify visible, unmistakable damages.
3. **Stage 3: Deterministic Rules Engine (`src/adjudicator.py`)**  
   Fuses the text claim and image evidence using ontology mapping, semantic embedding fallbacks, and leniency logic.

## Repository Structure

```
claim-adjudicator/
├── README.md
├── requirements.txt            # Python dependencies
├── main.py                     # Entrypoint script
├── data/
│   ├── sample_claims.csv       # Test dataset (claims and image paths)
│   └── evidence_requirements.csv # Object-specific filtering rules
|   └── user_history.csv         # for safety
└── src/
    ├── __init__.py
    ├── config.py               # Constants, model names, and ontology map
    ├── data_loader.py          # CSV loading utilities
    ├── extractor.py            # Stage 1: LLM extraction logic
    ├── image_analyzer.py       # Stage 2: VLM image analysis logic
    ├── adjudicator.py          # Stage 3: Fusion and decision rules
    ├── pipeline.py             # Sequential pipeline orchestration
    ├── evaluator.py            # Classification error analysis (predictions vs truth)
    └── utils.py                # Logging, GPU memory, and JSON helpers
```

## Setup Instructions

1. **Clone the repository** (or navigate to the directory).
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Hardware Requirements

> [!WARNING]
> This pipeline loads large localized LLM and VLM models (e.g. Qwen2.5-7B). 
> You **must** have a GPU with at least 15GB of VRAM (like an NVIDIA T4) to run the 7B models using 4-bit quantization. If you don't have enough VRAM, edit `src/config.py` to use smaller 3B or 2B models (e.g. `Qwen/Qwen2.5-3B-Instruct`).

## How to Run

### Full Pipeline
Run the entrypoint script to execute the complete end-to-end extraction, analysis, and adjudication on the dataset:
```bash
python main.py
```
*(Note: You will need to uncomment the function calls in `main.py` if you wish to actually load models and execute them.)*

### Individual Stages / Development
You can import modules directly for specific tasks:
- `python -c "import src.pipeline; src.pipeline.run_pipeline()"`
- Evaluation only: `python -c "import src.evaluator; src.evaluator.evaluate()"`

## Configuring the System
- **Vocabularies & Thresholds**: Update `src/config.py` to add new allowed damage types, object parts, or alter the `ONTOLOGY_MAP`.
- **Evidence Requirements**: The `data/evidence_requirements.csv` determines which rules apply to specific objects. Add rows to this CSV to support new claim objects or modify existing requirements.
