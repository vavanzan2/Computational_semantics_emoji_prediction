# =============================================================================
# step2_build_pairs.py
#
# GOAL: Identify experimental pairs (an interceptedturn preceded by
#       two artificialturns), extract the artificially inserted emoji,
#       remove the emoji from the text, and build the context version.
#       Saves the result to data/pairs_test.csv.
# =============================================================================

import os
import re
import pandas as pd

from config import EMOJI_PATTERN, TWEETEVAL_20, PATHS


def find_inserted_emoji(intercepted_text, artificial_text):
    """
    Returns the emoji that was inserted into the artificial text but was not
    present in the original intercepted text. If all emojis in the artificial
    text were already present in the intercepted text, returns the last emoji
    from the artificial text. Returns None if there is no emoji in the artificial text.
    """
    emojis_art  = re.findall(EMOJI_PATTERN, str(artificial_text))
    emojis_inte = re.findall(EMOJI_PATTERN, str(intercepted_text))

    for e in emojis_art:
        if e not in emojis_inte:
            return e

    if emojis_art:
        return emojis_art[-1]

    return None


def build_context(df2, group_id, turn_no, current_text, n_prev=10):
    """
      Returns the current text preceded by the last n_prev turns from the same group,
      separated by line breaks. Useful for providing conversational context
      to the language model.
    """
    g = df2[df2["Group ID"] == str(group_id)].reset_index(drop=True)

    idx_list = g.index[g["Turn No."] == turn_no].tolist()
    if not idx_list:
        return current_text  # turn not found, return the current text

    idx   = idx_list[0]
    start = max(0, idx - n_prev)
    # Ensure that each previous turn is a string and ignore empty/NaN values.
    prev_series = g.iloc[start:idx]["Text"].fillna("").tolist()
    prev_turns = [str(x) for x in prev_series if str(x) != ""]

  # Always return a string (only the current text if there are no previous turns).
    return "\n".join(prev_turns + [str(current_text)])


def run():
    print("[Step 2] Building experimental pairs...")

    df2 = pd.read_csv(PATHS["df2"], dtype=str)

    # Find the pattern: an interceptedturn immediately following two artificialturns in the same group.
    pairs = []
    for i in range(2, len(df2)):
        same_group = (
            df2.loc[i,     "Group ID"] == df2.loc[i - 1, "Group ID"] ==
            df2.loc[i - 2, "Group ID"]
        )
        turn_pattern = (
            df2.loc[i,     "Turntype"] == "interceptedturn" and
            df2.loc[i - 1, "Turntype"] == "artificialturn"  and
            df2.loc[i - 2, "Turntype"] == "artificialturn"
        )

        if same_group and turn_pattern:
            pairs.append({
                "Group ID":        df2.loc[i,     "Group ID"],
                "intercepted_turn": df2.loc[i,     "Turn No."],
                "artificial_1_turn": df2.loc[i - 2, "Turn No."],
                "artificial_2_turn": df2.loc[i - 1, "Turn No."],
                "artificial_text":  df2.loc[i - 1, "Text"],
                "intercepted_text": df2.loc[i,     "Text"],
            })

    pairs_test = pd.DataFrame(pairs)
    print(f"  Pairs interceptedturn found: {len(pairs_test)}")

    if pairs_test.empty:
        raise ValueError(
            "No pairs found. Check whether the 'Turntype' column contains "
            "the values 'interceptedturn' and 'artificialturn'."
        )

    # Identify the artificially inserted emoji in each pair.
    pairs_test["emoji_inserted"] = pairs_test.apply(
        lambda r: find_inserted_emoji(r["intercepted_text"], r["artificial_text"]),
        axis=1
    )

    # Keep only pairs where the emoji could be identified.
    pairs_test = pairs_test[pairs_test["emoji_inserted"].notna()].copy()
    print(f"  Pairs with an identifiable emoji: {len(pairs_test)}")

    # Remove the emoji from the intercepted text (so the model does not "cheat").
    pairs_test["intercepted_noemoji"] = pairs_test["intercepted_text"].apply(
        lambda x: re.sub(EMOJI_PATTERN, "", str(x)).strip()
    )

    # Build a version of the text with conversational context (10 previous turns).
    pairs_test["intercepted_with_context"] = pairs_test.apply(
        lambda r: build_context(
            df2,
            r["Group ID"],
            r["intercepted_turn"],
            r["intercepted_noemoji"],
            n_prev=10
        ),
        axis=1
    )

    os.makedirs(os.path.dirname(PATHS["pairs_test"]), exist_ok=True)
    pairs_test.to_csv(PATHS["pairs_test"], index=False)
    print(f"  Pairs saved in: {PATHS['pairs_test']}")

    return pairs_test


if __name__ == "__main__":
    run()
