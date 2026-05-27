# =============================================================================
# step4_predict_spontaneous.py
#
# # GOAL: For normal turns (normalturn) where the user used
#       an emoji spontaneously, run the model with and without context
#       and compute the delta as a comparison group for the artificial cases.
#       Saves the result to data/normal_emoji_df.csv.
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODEL_NAME, EMOJI_PATTERN, TWEETEVAL_20, EMOJI_TO_ID, PATHS
from step2_build_pairs import build_context


def load_model():
    """ Loads the tokenizer and model from Hugging Face and sets up the device."""
    print(f"  Loading mdoel '{MODEL_NAME}'...")
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
    print("[Step 4] Running predictions on turns with spontaneous emojis....")

    df2 = pd.read_csv(PATHS["df2"], dtype=str)

    # Filter only normalturns
    normal_df = df2[df2["Turntype"] == "normalturn"].copy()
    print(f"  Normal turns found: {len(normal_df)}")

    # Extract the first emoji of each turn
    normal_df["user_emoji"] = normal_df["Text"].apply(
        lambda x: re.findall(EMOJI_PATTERN, str(x))[0]
        if re.findall(EMOJI_PATTERN, str(x)) else None
    )

    # Keep only the turns with emoji compatible with the model  (TWEETEVAL_20)
    normal_emoji_df = normal_df[
        normal_df["user_emoji"].notna() &
        normal_df["user_emoji"].isin(TWEETEVAL_20)
    ].copy()
    print(f"  Turns with emojis compatible with the model : {len(normal_emoji_df)}")

    if normal_emoji_df.empty:
        raise ValueError(
            "No normalturn with compatible emoji found. "
            "Verify the columns 'Turntype' and the EMOJI_PATTERN on config.py."
        )

    # Remove the emoji from the intercepted text (so the model does not "cheat")
    normal_emoji_df["text_noemoji"] = normal_emoji_df["Text"].apply(
        lambda x: re.sub(EMOJI_PATTERN, "", str(x)).strip()
    )

    # Build a version with conversational context (10 previous turns).
    normal_emoji_df["text_with_context"] = normal_emoji_df.apply(
        lambda r: build_context(
            df2,
            r["Group ID"],
            r["Turn No."],
            r["text_noemoji"],
            n_prev=10
        ),
        axis=1
    )

    tokenizer, model, device = load_model()

    # True class IDs (the emoji chosen by the user)
    true_ids = normal_emoji_df["user_emoji"].map(EMOJI_TO_ID).astype(int).values

    # Prediction without context
    print("  Predicting without context...")
    probs_noctx = predict_probs(
        normal_emoji_df["text_noemoji"].astype(str).tolist(),
        tokenizer, model, device
    )

    # Prediction with context
    print("  Predicting with context...")
    probs_ctx = predict_probs(
        normal_emoji_df["text_with_context"].astype(str).tolist(),
        tokenizer, model, device
    )

    # Probability of the correct emoji in each condition
    normal_emoji_df["p_noctx"] = probs_noctx[np.arange(len(normal_emoji_df)), true_ids]
    normal_emoji_df["p_ctx"]   = probs_ctx  [np.arange(len(normal_emoji_df)), true_ids]

    # Delta: effect of context on emoji probability
    normal_emoji_df["delta_context"] = normal_emoji_df["p_ctx"] - normal_emoji_df["p_noctx"]

    os.makedirs(os.path.dirname(PATHS["normal_emoji"]), exist_ok=True)
    normal_emoji_df.to_csv(PATHS["normal_emoji"], index=False)
    print(f"  Result saved in: {PATHS['normal_emoji']}")

    return normal_emoji_df


if __name__ == "__main__":
    run()
