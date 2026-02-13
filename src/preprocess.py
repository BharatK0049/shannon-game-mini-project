import nltk
from nltk.corpus import brown
import re
from pathlib import Path

def clean_text(text):
    """
    Convert text to lowercase, only a-z and spaces
    Multiple spaces collapse into one
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove punctuation, digits
    text = re.sub(r"\s+", " ", text)        # Collapse spaces
    
    return text.strip()

# Main

nltk.download("brown")

# Storing the entire brown corpus as a single string
words = brown.words()
raw_text = " ".join(words)

# Clean text
cleaned_text = clean_text(raw_text)

# Train/test split 80-20 split
split_index = int(len(cleaned_text) * 0.8)

train_text = cleaned_text[:split_index] # First 80% of the brown corpus words
test_text = cleaned_text[split_index:] # Words from the 80th percentile onwards

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Save files
(data_dir / "brown_train.txt").write_text(train_text)
(data_dir / "brown_test.txt").write_text(test_text)

print("Preprocessing complete.")
print(f"Training characters: {len(train_text)}")
print(f"Testing characters: {len(test_text)}")