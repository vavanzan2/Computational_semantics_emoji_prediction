# =============================================================================
# config.py — Constants and paths shared across all steps
# =============================================================================

# Data path 
DATA_PATH = "emoji_data.xlsx"

# Emoji prediction model (RoBERTa trained on TweetEval)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-emoji"

# Regex to detect emojis in text
EMOJI_PATTERN = r"[😂😊😎😉😁😜❤😍💕🔥✨💙😘📷🇺🇸☀💜💯🎄📸😖😞😭😓😫]"

# The 20 emojis the model knows (order matters: index = class ID )
TWEETEVAL_20 = [
    "❤", "😍", "😂", "💕", "🔥", "😊", "😎", "✨", "💙", "😘",
    "📷", "🇺🇸", "☀", "💜", "😉", "💯", "😁", "🎄", "📸", "😜"
]

# Mapping between emoji and ID 
EMOJI_TO_ID = {e: i for i, e in enumerate(TWEETEVAL_20)}
ID_TO_EMOJI  = {i: e for i, e in enumerate(TWEETEVAL_20)}

# Path where each step saves/reads the files 
PATHS = {
    "df2":         "data/df2.csv",
    "pairs_test":  "data/pairs_test.csv",
    "pairs_eval":  "data/pairs_eval.csv",
    "normal_emoji": "data/normal_emoji_df.csv",
    "plots":       "plots",
}
