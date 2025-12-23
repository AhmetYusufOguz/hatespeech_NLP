import pandas as pd
import re

# Load dataset
df = pd.read_excel(
    "hatespeech_dataset.xlsx",
    sheet_name="Dengeli Veriseti",
    header=None
)

df.columns = ["id", "text", "label"]

flags = []

# Keyword patterns
THREAT_WORDS = r"\b(döv|öldür|kes|vur|yak|boğ)\w*"
SELF_HARM_WORDS = r"\b(kendini|intihar|öl|öldür kendini|nefes almanın anlamı)\b"
INSULT_WORDS = r"\b(aptal|salak|mal|geri zekalı|şerefsiz)\b"
HATE_WORDS = r"\b(nefret ediyorum|iğreniyorum|tiksiniyorum)\b"
GROUP_WORDS = r"\b(bunlar|bu insanlar|hepsi|ırk|millet|din|topluluk)\b"

for idx, row in df.iterrows():
    text = str(row["text"]).lower()
    label = row["label"]

    reason = []

    # Threat but not labeled as threat
    if re.search(THREAT_WORDS, text) and label not in ["tehdit", "niyet"]:
        reason.append("VIOLENCE_WORD_BUT_NOT_THREAT")

    # Self-harm indicators
    if re.search(SELF_HARM_WORDS, text) and label != "niyet":
        reason.append("SELF_HARM_BUT_NOT_NIYET")

    # Insult indicators
    if re.search(INSULT_WORDS, text) and label not in ["saldırgan", "nefret"]:
        reason.append("INSULT_BUT_NOT_SALDIRGAN")

    # Hate expressions without group
    if re.search(HATE_WORDS, text) and not re.search(GROUP_WORDS, text):
        if label == "nefret":
            reason.append("PERSONAL_HATE_LABELED_AS_NEFRET")

    # Group hate but labeled neutral
    if re.search(GROUP_WORDS, text) and re.search(INSULT_WORDS, text):
        if label == "hiçbiri":
            reason.append("GROUP_ATTACK_LABELED_HICBIRI")

    flags.append(", ".join(reason))

df["FLAG_REASON"] = flags

# Keep only flagged rows
flagged_df = df[df["FLAG_REASON"] != ""]

# Save results
flagged_df.to_excel("flagged_for_review.xlsx", index=False)

print(f"Flagged {len(flagged_df)} rows for manual review.")
