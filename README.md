# RAG-CFE: Retrieval-Augmented Counterfactual Explanations for Algorithmic Recourse

This repository implements **RAG-CFE**, a simple, reproducible pipeline that:
1) uses **DiCE** to generate \(k\) candidate counterfactuals (CFs) per factual instance,  
2) (optionally) uploads **domain documents** to the LLM as grounding context, and  
3) asks an **LLM (Gemini 2.5 Flash)** to select or generate a **single refined counterfactual**.  

We evaluate **RAG-CFE** against a DiCE baseline on the **UCI Adult Income** dataset with standard recourse metrics (validity, proximity \( \ell_1/\ell_2 \), Gower, sparsity, feasibility, plausibility).

---

## Repo layout

```
.
├── dice_cfe.py          # Generate DiCE CF candidates → /cf_pairs/*.json
├── rag_cfe.py           # Run RAG refinement with Gemini → /rag_cfe_outputs/*.json
├── evaluation.py        # Evaluate RAG vs baseline JSONs → prints aggregate summary
├── prompt.txt           # System instruction for the LLM (expected at /prompt.txt)
├── docs/                # Domain PDFs uploaded to Gemini


## Installation

Tested with Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          
pip install --upgrade pip

# Core libs
pip install dice-ml scikit-learn pandas numpy google-generativeai
```

You don’t need to manually download the Adult dataset or a model; `dice_ml.utils.helpers` handles it.

---

## Environment

Set your Gemini key:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

(Windows PowerShell)
```powershell
setx GEMINI_API_KEY "YOUR_API_KEY"
```

---

## 1) Generate DiCE candidates

```bash
python dice_cfe.py
```

This writes **per-factual case files** under:

```
/cf_pairs/test_point_0.json
/cf_pairs/test_point_1.json
...
```

Only **factuals with income=0** are saved, to match the positive-class recourse setup.

---

## 2) (Optional) Add domain documents for grounding

Place PDFs/CSVs/TXTs under:

```
/docs/
```

These will be uploaded to Gemini once at the start of `rag_cfe.py`. If you skip this folder, the LLM still runs but without file context.

---

## 3) RAG refinement (Gemini 2.5 Flash)

Ensure your **prompt** is available at the absolute path expected in the script:

- `rag_cfe.py` expects: **`/prompt.txt`**  
  (If your `prompt.txt` is in the repo root, either move/copy it to `/prompt.txt` or change `PROMPT_FILE` in the script.)

Run:

```bash
python rag_cfe.py
```

Outputs:

```
/rag_cfe_outputs/test_point_0_out.json
/rag_cfe_outputs/test_point_1_out.json
...
```

Each output is a **single JSON object** like:
```json
{
  "best_cf": [...],
  "explanation": "..."
}
```

—this is exactly what `evaluation.py` expects for the RAG side.

---

## 4) Baseline JSONs for evaluation (one per factual)

`evaluation.py` compares two folders:

- RAG results: **`/rag_cfe_outputs`** (already produced by step 3)  
- Baseline results: **`/dice_cfe_outputs`** (must be created as **one-best-CF per factual** JSON)

`dice_cfe.py` currently writes **candidate sets** into `/cf_pairs/`, not one-best-CF JSONs. For evaluation, create baseline JSONs by picking the first candidate (or the closest one) per case and writing the **RAG-style** schema to `/dice_cfe_outputs/`.


## 5) Evaluate

```bash
python evaluation.py
```

This will:
- load Adult + DiCE’s pretrained model,
- read `/rag_cfe_outputs/*.json` and `/dice_cfe_outputs/*.json`,
- compute metrics per factual and print an **aggregate summary**.

---

## Metrics

- **Validity**: fraction of CFs classified as positive (target class).  
- **Proximity**: normalized \( \ell_1 \), \( \ell_2 \) over continuous features.  
- **Gower**: distance for mixed types.  
- **Sparsity**: number of features changed.  
- **Feasibility**: immutables unchanged (`age`, `gender`, `race`) and monotonic constraint enforced (`hours_per_week` up only).  
- **Plausibility**: kNN density proxy on numeric features (higher is better).


---
