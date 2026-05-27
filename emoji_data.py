import sys
import re
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from networkx import display
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import ttest_rel

PATH = "emoji_data.xlsx"
df = pd.read_excel(PATH)

print(df.shape)
print(df.columns.tolist())


def turn_order(x):
    s = str(x)
    m = re.match(r"(\d+)-(\d+)", s)
    if m:
        return int(m.group(1))*1000 + int(m.group(2))
    return 0

df2 = df.copy()

df2["Group ID"] = df2["Group ID"].astype(str)
df2["Turntype"] = df2["Turntype"].astype(str).str.lower()

df2["order"] = df2["Turn No."].map(turn_order)

df2 = df2.sort_values(["Group ID","order"]).reset_index(drop=True)

df2[["Group ID","Turn No.","Turntype","Text"]].head(5)


pairs_test = []

for i in range(2, len(df2)):
    if (
        df2.loc[i, "Turntype"] == "interceptedturn"
        and df2.loc[i-1, "Turntype"] == "artificialturn"
        and df2.loc[i-2, "Turntype"] == "artificialturn"
        and df2.loc[i, "Group ID"] == df2.loc[i-1, "Group ID"] == df2.loc[i-2, "Group ID"]
    ):
        pairs_test.append({
            "Group ID": df2.loc[i, "Group ID"],
            "intercepted_turn": df2.loc[i, "Turn No."],
            "artificial_1_turn": df2.loc[i-2, "Turn No."],
            "artificial_2_turn": df2.loc[i-1, "Turn No."],
            "artificial_text": df2.loc[i-1, "Text"],
            "intercepted_text": df2.loc[i, "Text"],
        })

pairs_test = pd.DataFrame(pairs_test)

print(pairs_test.shape)
pairs_test.head(5)


emoji_pattern = r"[😂😊😎😉😁😜❤😍💕🔥✨💙😘📷🇺🇸☀💜💯🎄📸😖😞😭😓😫]"

def find_inserted(intercepted, artificial):
    art = re.findall(emoji_pattern, artificial)
    inte = re.findall(emoji_pattern, intercepted)

    for e in art:
        if e not in inte:
            return e

    if len(art) > 0:
        return art[-1]

    return None


pairs_test["emoji_inserted"] = pairs_test.apply(
    lambda r: find_inserted(r["intercepted_text"], r["artificial_text"]),
    axis=1
)

pairs_test.head(20)



print(pairs_test.columns.tolist())

pairs_test = pairs_test[pairs_test["emoji_inserted"].notna()].copy()

pairs_test["intercepted_noemoji"] = pairs_test["intercepted_text"].apply(
    lambda x: re.sub(emoji_pattern, "", str(x)).strip()
)

pairs_test.head(20)



print(pairs_test.shape)


def build_context(group_id, turn_no, current_text, n_prev=10):
    g = df2[df2["Group ID"] == str(group_id)].copy().reset_index(drop=True)

    # find the position of the actual turn in the group
    idx_list = g.index[g["Turn No."] == turn_no].tolist()
    if len(idx_list) == 0:
        return current_text

    idx = idx_list[0]

    start = max(0, idx - n_prev)
    prev_turns = g.iloc[start:idx]["Text"].astype(str).tolist()

    return "\n".join(prev_turns + [str(current_text)])



pairs_test["intercepted_with_context"] = pairs_test.apply(
    lambda r: build_context(
        r["Group ID"],
        r["intercepted_turn"],
        r["intercepted_noemoji"],
        n_prev=10
    ),
    axis=1
)

pairs_test[[
    "intercepted_turn",
    "intercepted_noemoji",
    "intercepted_with_context"
]].head(15)



dataset = load_dataset("tweet_eval", "emoji")
labels = dataset["train"].features["label"].names

df = dataset["train"].to_pandas()
df["emoji"] = df["label"].apply(lambda x: labels[x])

df.head(10)


print(len(dataset["train"]))
print(len(dataset["validation"]))
print(len(dataset["test"]))

MODEL_NAME = "cardiffnlp/twitter-roberta-base-emoji"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

TWEETEVAL_20 = [
    "❤","😍","😂","💕","🔥","😊","😎","✨","💙","😘",
    "📷","🇺🇸","☀","💜","😉","💯","😁","🎄","📸","😜"
]

emoji_to_id = {e:i for i,e in enumerate(TWEETEVAL_20)}

@torch.inference_mode()
def predict_probs(texts, batch_size=32, max_length=128):
    probs_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs_all.append(probs)

    return np.vstack(probs_all)


id_to_emoji = {v: k for k, v in emoji_to_id.items()}

# Create a dictionary by reversing the emoji-to-ID mapping.
emoji_to_id

no_emoji_df = df2[
    ~df2["Text"].astype(str).apply(
        lambda x: any(e in x for e in TWEETEVAL_20)
    )
].copy()

# Filter rows without model emojis and create a new dataframe.
no_emoji_df = df2[
    df2["Text"].astype(str).apply(
        lambda x: not any(e in str(x) for e in TWEETEVAL_20)
    )
].copy()


# Filters turns without emojis supported by the model.
texts = no_emoji_df["Text"].astype(str).tolist()
probs = predict_probs(texts)

id_to_emoji = {v: k for k, v in emoji_to_id.items()}
emoji_order = [id_to_emoji[i] for i in range(len(TWEETEVAL_20))]

for i, emo in enumerate(emoji_order):
    no_emoji_df[f"prob_{emo}"] = probs[:, i]

pred_ids = probs.argmax(axis=1)
pred_probs = probs.max(axis=1)

no_emoji_df["predicted_emoji"] = [id_to_emoji[i] for i in pred_ids]
no_emoji_df["predicted_prob"] = pred_probs

# Computes emoji probabilities and stores the predictions in the dataframe.
no_emoji_df[[
    "Turn No.",
    "Text",
    "Type of intervention",
    "predicted_emoji",
    "predicted_prob"
]].head(60)

# Prints the rows with turn, text, intervention, and prediction.
no_emoji_df[
    no_emoji_df["predicted_prob"] >= 0.7
][[
    "Turn No.",
    "Text",
    "Type of intervention",
    "predicted_emoji",
    "predicted_prob"
]].sort_values("predicted_prob", ascending=False)

# Filters predictions >= 0.7 and sorts them by highest probability.
turn_id = "235-53"

row = no_emoji_df[no_emoji_df["Turn No."] == turn_id].iloc[0]
probs_turn = pd.DataFrame({
    "emoji": emoji_order,
    "probability": [row[f"prob_{emo}"] for emo in emoji_order]
}).sort_values("probability", ascending=False)

print("Turn No.:", row["Turn No."])
print("Text:", row["Text"])
print("Type of intervention:", row["Type of intervention"])

print(probs_turn)


pairs_eval = pairs_test[pairs_test["emoji_inserted"].isin(TWEETEVAL_20)].copy()

print("Available pairs:", pairs_eval.shape)

true_ids = pairs_eval["emoji_inserted"].map(emoji_to_id).astype(int).values

# intercepted only
probs_noctx = predict_probs(pairs_eval["intercepted_noemoji"].astype(str).tolist())
probs_ctx   = predict_probs(pairs_eval["intercepted_with_context"].astype(str).tolist())

pairs_eval["p_noctx"] = probs_noctx[np.arange(len(pairs_eval)), true_ids]
pairs_eval["p_ctx"]   = probs_ctx[np.arange(len(pairs_eval)), true_ids]

pairs_eval["delta_context"] = pairs_eval["p_ctx"] - pairs_eval["p_noctx"]

pairs_eval[[
    "intercepted_turn",
    "emoji_inserted",
    "p_noctx",
    "p_ctx",
    "delta_context"
]].head(30)


turn_to_intervention = (
    df2[["Turn No.", "Type of intervention"]]
    .drop_duplicates()
    .set_index("Turn No.")["Type of intervention"]
    .to_dict()
)

pairs_eval["Type of intervention"] = pairs_eval["artificial_2_turn"].map(turn_to_intervention)

pairs_eval[["intercepted_turn", "artificial_2_turn", "Type of intervention"]].head(10)

# Maps turns to intervention type and adds it to the dataframe.
pairs_eval["congruence_simple"] = (
    pairs_eval["Type of intervention"]
    .astype(str)
    .str.lower()
    .str.contains("incongruent")
    .map({True: "incongruent", False: "congruent"})
)

summary_context = (
    pairs_eval.groupby("congruence_simple")
    .agg(
        n=("emoji_inserted", "size"),
        mean_p_noctx=("p_noctx", "mean"),
        mean_p_ctx=("p_ctx", "mean"),
        mean_delta=("delta_context", "mean"),
        median_delta=("delta_context", "median"),
    )
    .reset_index()
)

summary_context


pairs_eval[["delta_context"]].describe()



plt.figure()
plt.scatter(pairs_eval["p_noctx"], pairs_eval["p_ctx"])
plt.plot([0,1],[0,1])
plt.xlabel("Without context")
plt.ylabel("With context")
plt.title("Effect of conversational context on emoji prediction")
plt.show()

plt.figure(figsize=(6,6))

plt.scatter(
    pairs_eval["p_noctx"],
    pairs_eval["p_ctx"],
    alpha=0.7
)

plt.plot([0,1],[0,1], linestyle="--")

plt.xlabel("Probability without context")
plt.ylabel("Probability with context")
plt.title("Before vs After Context")

plt.show()


ttest_rel(pairs_eval["p_noctx"], pairs_eval["p_ctx"])



plt.figure(figsize=(6,4))

plt.hist(pairs_eval["delta_context"], bins=10, alpha=0.8)

plt.axvline(0, linestyle="--")

plt.xlabel("Change in probability (context − no context)")
plt.ylabel("Number of examples")

plt.title("Distribution of context effect on emoji prediction")

plt.show()


normal_df = df2[df2["Turntype"] == "normalturn"].copy()

#Filters only rows with normalturn and creates a copy of the dataframe.
def extract_emoji(text):
    emojis = re.findall(emoji_pattern, str(text))
    return emojis[0] if emojis else None

normal_df["user_emoji"] = normal_df["Text"].apply(extract_emoji)

# Extracts the emoji from the text and stores it in the user_emoji column.
normal_emoji_df = normal_df[normal_df["user_emoji"].notna()].copy()

# Filters rows with an extracted emoji and creates a new dataframe.
normal_emoji_df = normal_emoji_df[
    normal_emoji_df["user_emoji"].isin(TWEETEVAL_20)
].copy()

# Filters emojis present in the TWEETEVAL_20 set and copies the dataframe.
normal_emoji_df["text_noemoji"] = normal_emoji_df["Text"].apply(
    lambda x: re.sub(emoji_pattern, "", str(x)).strip()
)

normal_emoji_df["text_with_context"] = normal_emoji_df.apply(
    lambda r: build_context(
        r["Group ID"],
        r["Turn No."],
        r["text_noemoji"]
    ),
    axis=1
)

true_ids = normal_emoji_df["user_emoji"].map(emoji_to_id).astype(int).values

probs_noctx = predict_probs(normal_emoji_df["text_noemoji"].tolist())
probs_ctx   = predict_probs(normal_emoji_df["text_with_context"].tolist())

normal_emoji_df["p_noctx"] = probs_noctx[np.arange(len(normal_emoji_df)), true_ids]
normal_emoji_df["p_ctx"]   = probs_ctx[np.arange(len(normal_emoji_df)), true_ids]

normal_emoji_df["delta_context"] = normal_emoji_df["p_ctx"] - normal_emoji_df["p_noctx"]

normal_emoji_df[[
    "Turn No.",
    "user_emoji",
    "p_noctx",
    "p_ctx",
    "delta_context"
]].head(100)

print("Spontaneous cases where context helps most:")
display(
    normal_emoji_df.sort_values("delta_context", ascending=False)[[
        "Turn No.", "user_emoji", "p_noctx", "p_ctx", "delta_context"
    ]].head(10)
)

print("Spontaneous cases where context hurts most:")
display(
    normal_emoji_df.sort_values("delta_context", ascending=True)[[
        "Turn No.", "user_emoji", "p_noctx", "p_ctx", "delta_context"
    ]].head(10)
)
normal_emoji_df["delta_context"].describe()

ttest_rel(normal_emoji_df["p_noctx"], normal_emoji_df["p_ctx"])

plt.figure(figsize=(6,4))

plt.hist(normal_emoji_df["delta_context"], bins=10, alpha=0.8)

plt.axvline(0, linestyle="--")

plt.xlabel("Change in probability (context − no context)")
plt.ylabel("Number of examples")

plt.title("Distribution of context effect on emoji prediction (spontaneous emojis)")

plt.show()

normal_emoji_df.groupby("Type of intervention").size().reset_index(name="count")

normal_emoji_df["user_emoji"].value_counts()


normal_emoji_df["user_emoji"].isin(TWEETEVAL_20).value_counts()


pd.crosstab(
    normal_emoji_df["Type of intervention"],
    normal_emoji_df["user_emoji"].isin(TWEETEVAL_20),
    rownames=["Intervention"],
    colnames=["Emoji in model set"]
)


artificial = pairs_eval["p_noctx"].dropna()
user = normal_emoji_df["p_noctx"].dropna()

bins = np.linspace(0, 1, 11)

counts_artificial, edges = np.histogram(artificial, bins=bins)
counts_user, _ = np.histogram(user, bins=bins)

centers = (edges[:-1] + edges[1:]) / 2
bar_width = (edges[1] - edges[0]) * 0.4

plt.figure(figsize=(6,4))

plt.bar(
    centers - bar_width/2,
    counts_artificial,
    width=bar_width,
    label="Artificial emoji",
)

plt.bar(
    centers + bar_width/2,
    counts_user,
    width=bar_width,
    label="User emoji",
)

plt.xlabel("Probability of emoji (no context)")
plt.ylabel("Number of examples")
plt.title("Emoji prediction without conversational context")
plt.legend()
plt.tight_layout()
plt.show()


art_best = pairs_eval.sort_values("delta_context", ascending=False)[[
    "Group ID",
    "intercepted_turn",
    "artificial_2_turn",
    "emoji_inserted",
    "intercepted_noemoji",
    "intercepted_with_context",
    "p_noctx",
    "p_ctx",
    "delta_context",
    "Type of intervention"
]]

print("Artificial cases where context helps most:")
display(art_best.head(5))


art_worst = pairs_eval.sort_values("delta_context", ascending=True)[[
    "Group ID",
    "intercepted_turn",
    "artificial_2_turn",
    "emoji_inserted",
    "intercepted_noemoji",
    "intercepted_with_context",
    "p_noctx",
    "p_ctx",
    "delta_context",
    "Type of intervention"
]]

print("Artificial cases where context hurts most:")
display(art_worst.head(100))

# Top spontaneous cases where context helps most
sp_best = normal_emoji_df.sort_values("delta_context", ascending=False)[[
    "Group ID",
    "Turn No.",
    "user_emoji",
    "text_noemoji",
    "text_with_context",
    "p_noctx",
    "p_ctx",
    "delta_context",
    "Type of intervention"
]]

print("Spontaneous cases where context helps most:")
display(sp_best.head(5))

# Top spontaneous cases where context hurts most
sp_worst = normal_emoji_df.sort_values("delta_context", ascending=True)[[
    "Group ID",
    "Turn No.",
    "user_emoji",
    "text_noemoji",
    "text_with_context",
    "p_noctx",
    "p_ctx",
    "delta_context",
    "Type of intervention"
]]

print("Spontaneous cases where context hurts most:")
display(sp_worst.head(100))