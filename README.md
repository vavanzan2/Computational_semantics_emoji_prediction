# Does Conversational Context Help Predict Emoji Use?
### A Computational Pipeline for Analysing Artificially Inserted and Spontaneous Emojis

---

## Overview

This repository contains the data processing pipeline developed to support the research paper submitted to the evaluation board. The study investigates whether **conversational context** improves the ability of a pre-trained language model to predict how likely the emoji used (either artifically or spontaneously) is a given a text/turn.

Two types of emoji use are examined:

- **Artificially inserted emojis** — emojis placed by the server into experimentally intercepted turns, which may be congruent or incongruent.
- **Spontaneous emojis** — emojis chosen freely by participants in naturally occurring turns.

For each case, the model's prediction is evaluated under two conditions:
1. Using only the target message (no context)
2. Using the target message preceded by the 10 previous turns (with context)

The difference in predicted probabilities between these two conditions (Δ = *p*_context − *p*_no-context) serves as the operationalisation of **contextual effect** on emoji prediction.

---

## Research Context

This pipeline was designed to complement qualitative and quantitative linguistic analysis of emoji use in mediated communication. The computational component provides:

- Automated identification of experimental turn sequences in annotated conversation data
- Probability scores from a pre-trained transformer model for each emoji candidate
- Statistical comparisons (paired t-tests) between predicted probabilities with and without conversational context
- Visualisations for reporting in the accompanying paper

---

## Model

The emoji prediction model used is [`cardiffnlp/twitter-roberta-base-emoji`](https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji), a RoBERTa-based model fine-tuned on the [TweetEval benchmark](https://github.com/cardiffnlp/tweeteval) for 20-class emoji classification. The 20 emoji classes are:

❤ 😍 😂 💕 🔥 😊 😎 ✨ 💙 😘 📷 🇺🇸 ☀ 💜 😉 💯 😁 🎄 📸 😜

> Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020).
> TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.
> *Findings of EMNLP 2020.*

---

## Repository Structure

```
.
├── config.py                     # Shared constants: file paths, model name, emoji list
├── step1_load_and_prepare.py     # Step 1: load Excel data, clean and sort turns
├── step2_build_pairs.py          # Step 2: extract experimental pairs and build context
├── step3_predict_artificial.py   # Step 3: model predictions on artificially inserted emojis
├── step4_predict_spontaneous.py  # Step 4: model predictions on spontaneous emojis
├── step5_analysis.py             # Step 5: summarize results, run paired t-tests, identify extreme cases, and generate plots
├── main.py                       # Entry point: run all steps or selected steps via CLI
├── requirements.txt              # Python dependencies
│
├── emoji_data.xlsx               # Input data file (not included — see Data section below)
│
├── data/                         # Auto-generated intermediate files
│   ├── df2.csv                   # Cleaned and sorted conversation data
│   ├── pairs_test.csv            # Identified experimental pairs with extracted emojis
│   ├── pairs_eval.csv            # Model predictions for artificially inserted emojis
│   └── normal_emoji_df.csv       # Model predictions for spontaneous emojis
│
└── plots/                        # Auto-generated figures
    ├── artificial_scatter.png    # Scatter: p(no context) vs p(context) — artificial emojis
    ├── artificial_delta_hist.png # Distribution of context effect — artificial emojis
    ├── spontaneous_delta_hist.png# Distribution of context effect — spontaneous emojis
    └── comparison_histogram.png  # Side-by-side comparison: artificial vs spontaneous
```

---

## Requirements

- Python 3.9 or higher
- Internet access (required only on first run, to download the pre-trained model)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Package        | Purpose                                      |
|----------------|----------------------------------------------|
| `pandas`       | Data manipulation and tabular processing     |
| `openpyxl`     | Reading `.xlsx` Excel files                  |
| `numpy`        | Numerical operations                         |
| `torch`        | Deep learning inference (PyTorch)            |
| `transformers` | Loading pre-trained RoBERTa model            |
| `datasets`     | Accessing TweetEval via HuggingFace Hub      |
| `matplotlib`   | Plot generation                              |
| `scipy`        | Paired t-tests (`scipy.stats.ttest_rel`)     |

---

## Data

The input file `emoji_data.xlsx` contains conversation data and is **not included** in this repository due to research data privacy. The file must be placed in the root directory before running the pipeline.

The spreadsheet is expected to contain at minimum the following columns:

| Column | Description |
|---|---|
| `Group ID` | Identifier of the conversation group |
| `Turn No.` | Turn identifier in the format `group-turn` (e.g. `12-3`) |
| `Turntype` | Type of turn: `normalturn`, `artificialturn`, or `interceptedturn` |
| `Text` | Message content |
| `Type of intervention` | Experimental condition label (e.g. `congruent`, `incongruent`) |

If the filename differs from `emoji_data.xlsx`, update the `DATA_PATH` variable in `config.py`.

---

## Running the Pipeline

```bash
# Run all steps in sequence (recommended for first execution)
python main.py --all

# Run a specific step only
python main.py --step 3

# Run a range of steps (e.g. steps 2 through 4)
python main.py --step 2 --ate 4

# List all available steps
python main.py --listar

# Show help
python main.py --help
```

### Pipeline steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_load_and_prepare.py` | Load Excel data, normalise column values, sort turns |
| 2 | `step2_build_pairs.py` | Detect experimental sequences, extract inserted emojis, build context strings |
| 3 | `step3_predict_artificial.py` | Run model on artificially inserted emoji pairs (with and without context) |
| 4 | `step4_predict_spontaneous.py` | Run model on spontaneous emoji turns (with and without context) |
| 5 | `step5_analysis.py` | Compute summary statistics, run t-tests, save plots |

---

## Output

After a full run, the following outputs are available:

**Intermediate data** (in `data/`):  
CSV files at each step allow any individual step to be re-run independently without reprocessing the entire pipeline.

**Statistical output** (printed to terminal during Step 5):  
- Mean and median Δ by congruence group (congruent vs incongruent)
- Paired t-test results for artificial and spontaneous emoji conditions
- Top cases where context most increased / decreased prediction probability

**Figures** (in `plots/`):

| Figure | Description |
|--------|-------------|
| `artificial_scatter.png` | Scatter plot of p(no context) vs p(context) for artificially inserted emojis |
| `artificial_delta_hist.png` | Histogram of Δ values for artificially inserted emojis |
| `spontaneous_delta_hist.png` | Histogram of Δ values for spontaneous emojis |
| `comparison_histogram.png` | Grouped histogram comparing prediction distributions across both conditions |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: emoji_data.xlsx` | Ensure the data file is in the project root and `DATA_PATH` in `config.py` is correct |
| SSL certificate error when downloading the model | Add `os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"` at the top of `config.py` (common in corporate or institutional networks) |
| `KeyError` for a column name | Verify the Excel file contains all required columns listed in the Data section above |
| Re-running only the analysis | If `data/` files already exist, run `python main.py --step 5` directly |
| Slow first run | Expected — the model (~500 MB) is downloaded once and cached locally for subsequent runs |

---

## Notes on Reproducibility

- All intermediate results are serialised as CSV files in `data/`, enabling full reproducibility of the analysis and statistical steps without re-running model inference.
- The model is loaded in inference mode (`torch.inference_mode()`) with no gradient computation, ensuring deterministic outputs on CPU.
- GPU acceleration is used automatically if a CUDA-compatible device is available; otherwise, the pipeline runs on CPU.
