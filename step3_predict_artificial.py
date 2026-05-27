# =============================================================================
# step3_predict_artificial.py
#
# GOAL: For each experimental pair (artificially inserted emoji),
#       run the emoji prediction model with and without conversational context
#       and compute the delta (difference in probability).
#       Saves the result to data/pairs_eval.csv.
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODEL_NAME, TWEETEVAL_20, EMOJI_TO_ID, ID_TO_EMOJI, PATHS


def load_model():
    """Loads the tokenizer and model from Hugging Face and sets up the device."""
    print(f"  Loading model '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device).eval()
    print(f"  Model loaded : {device}")
    return tokenizer, model, device


@torch.inference_mode()
def predict_probs(texts, tokenizer, model, device, batch_size=32, max_length=128):
    """
     Takes a list of texts and returns an (N x 20) matrix with the probability 
     of each of the 20 emojis for each text.
    """
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all)


def run():
    print("[Step 3] Running predictions on the artificial emoji pairs...")

    df2        = pd.read_csv(PATHS["df2"],        dtype=str)
    pairs_test = pd.read_csv(PATHS["pairs_test"], dtype=str)

    # Keep only pairs whose emoji is in the set of 20 emojis known by the model.
    pairs_eval = pairs_test[pairs_test["emoji_inserted"].isin(TWEETEVAL_20)].copy()
    print(f"   Pairs with an emoji supported by the model: {len(pairs_eval)}")

    if pairs_eval.empty:
        raise ValueError(
            "No pairs with an emoji from TWEETEVAL_20 were found. "
            "Check the file generated in Step 2."
        )

    tokenizer, model, device = load_model()

    # True class IDs (the emoji that was inserted)
    true_ids = pairs_eval["emoji_inserted"].map(EMOJI_TO_ID).astype(int).values

    # Prediction without context
    print("  Predicting without context...")
    probs_noctx = predict_probs(
        pairs_eval["intercepted_noemoji"].astype(str).tolist(),
        tokenizer, model, device
    )

    # Prediction with context (10 previous turns + current text)
    print("  Predicting with context...")
    probs_ctx = predict_probs(
        pairs_eval["intercepted_with_context"].astype(str).tolist(),
        tokenizer, model, device
    )

    # Probability of the emoji in each scenario
    pairs_eval["p_noctx"] = probs_noctx[np.arange(len(pairs_eval)), true_ids]
    pairs_eval["p_ctx"]   = probs_ctx  [np.arange(len(pairs_eval)), true_ids]

    # Delta: effect of context on emoji probability
    pairs_eval["delta_context"] = pairs_eval["p_ctx"] - pairs_eval["p_noctx"]

    # Mapping : turn → type of intervention (congruent / incongruent)
    if "Type of intervention" not in df2.columns:
        print("  # Warning: column 'Type of intervention' not found in df2. "
              "The congruent/incongruent classification will be marked as unknown.")
        pairs_eval["Type of intervention"] = "unknown"
    else:
        turn_to_intervention = (
            df2[["Turn No.", "Type of intervention"]]
            .drop_duplicates()
            .set_index("Turn No.")["Type of intervention"]
            .to_dict()
        )
        pairs_eval["Type of intervention"] = pairs_eval["artificial_2_turn"].map(
            turn_to_intervention
        )

    # Simplified classification as congruent / incongruent
    pairs_eval["congruence_simple"] = (
        pairs_eval["Type of intervention"]
        .astype(str)
        .str.lower()
        .str.contains("incongruent")
        .map({True: "incongruent", False: "congruent"})
    )

    os.makedirs(os.path.dirname(PATHS["pairs_eval"]), exist_ok=True)
    pairs_eval.to_csv(PATHS["pairs_eval"], index=False)
    print(f"  Result saved in : {PATHS['pairs_eval']}")

    return pairs_eval


if __name__ == "__main__":
    run()
