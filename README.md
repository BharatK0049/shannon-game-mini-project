# Multi-Purpose NLP Suite: Shannon Game & Noisy Channel Spell Checker

## Overview
This project implements a flexible **Word-Level N-Gram Model** designed to explore the predictability of the English language. Inspired by Claude Shannon’s 1951 paper *Prediction and Entropy of Printed English*, it empirically measures language entropy and provides practical applications for sequence prediction and error correction.

Key features include:
* **Dynamic N-Gram Engine:** Supports context lengths from Bigrams up to a user-defined cap (default $N=5$).
* **Shannon Game Implementation:** Empirically calculates Entropy and Perplexity based on word-prediction difficulty.
* **Noisy Channel Spell Checker:** Uses the Bayesian argmax $P(w|c)P(c)$ to correct spelling errors using context-aware probabilities.

---

## Core Applications

### 1. The Shannon Game (Predictive Modeling)
The model predicts the most likely next word given a specific context.
* **Task:** Focuses on predicting the word that is not there in the sentence.
* **Metric:** Uses **Perplexity** to quantify model uncertainty. A lower perplexity indicates the model has narrowed down the possibilities to a smaller "effective" vocabulary.

### 2. Noisy Channel Spell Checker
Corrects misspelled words by balancing the likelihood of the error against the probability of the word occurring in a specific context.
* **Formula:** $\hat{c} = \text{argmax}_{c \in \text{candidates}} P(w|c)P(c)$.
* **Logic:** Generates candidates using edit distance (deletions, transposes, replaces, inserts) and ranks them using the N-Gram Language Model.

---

## Technical Details

### Word-Level Prediction Logic
Unlike character-level models, this implementation processes the **Brown Corpus** as a sequence of word tokens.
* **Context:** A tuple of $n-1$ preceding words.
* **Dynamic Sizing:** The system adapts to different input lengths by utilizing the available context from the user's input up to the defined model cap.

### Estimating Entropy and Perplexity
After "playing" the game on the test set (`brown_test.txt`), entropy is estimated using the rank $r$ of the correct word:
$$H = \sum P(r) \log_2(r)$$
Perplexity is then derived as $PP = 2^H$.

---

## Project Structure
```text
shannon_game/
├── data/
│   ├── brown_train.txt      # 80% Training corpus (Word tokens)
│   └── brown_test.txt       # 20% Test corpus (Unseen)
├── src/
│   ├── preprocess.py        # Tokenization and cleaning
│   ├── ngram_model.py       # Dynamic N-Gram frequency engine
│   ├── shannon_game.py      # Entropy and Perplexity calculation logic
│   ├── spell_checker.py     # Noisy Channel Model implementation
│   └── main.py              # Interactive CLI with automated evaluation
├── requirements.txt
└── README.md
```

## Module Descriptions

### `preprocess.py`
* **Purpose**: Handles data acquisition and cleaning.
* **Logic**: Downloads the Brown corpus via NLTK, converts text to lowercase, and removes all non-alphabetic characters.
* **Output**: Tokenizes the text into a clean list of words and saves them into an 80/20 train-test split for experimentation.

### `ngram_model.py`
* **Purpose**: Implements the core Language Model (LM).
* **Logic**: A flexible $N$-Gram implementation that stores frequency counts for word sequences. It maps a context (a tuple of $n-1$ words) to a counter of possible succeeding words.
* **Capability**: Supports dynamic context lengths, allowing the model to provide predictions based on the best available match in the training data.

### `shannon_game.py`
* **Purpose**: Quantifies language predictability.
* **Logic**: Iterates through the test set and calculates the "rank" of the correct word in the model's predictions.
* **Metrics**: Calculates global first-guess accuracy, entropy (average bits of information per word), and perplexity.

### `spell_checker.py`
* **Purpose**: Implements the Noisy Channel Model for error correction.
* **Logic**: Uses a two-step process to correct misspelled words:
    1. **Candidate Generation**: Generates potential correct words using a Damerau-Levenshtein distance of 1 or 2.
    2. **Bayesian Ranking**: Calculates $P(w|c)P(c)$, where $P(c)$ is derived from the N-Gram context provided by `ngram_model.py`.
* **Edit Distance Logic**: The model considers four types of single-character edits:
    * **Deletions**: Removing a character (e.g., "stats" → "stats").
    * **Insertions**: Adding a character (e.g., "stts" → "stats").
    * **Substitutions**: Swapping one character for another (e.g., "stets" → "stats").
    * **Transpositions**: Swapping two adjacent characters (e.g., "stats" → "stats").

### `main.py`
* **Purpose**: Orchestrates the entire pipeline and provides a user interface.
* **Logic**: Loads tokens, trains the $N$-Gram model, runs the automated evaluation to report perplexity, and enters an interactive loop for real-time word prediction and spell checking.

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Preprocess the corpus
```
python src/preprocess.py
```

### Run the Shannon game
```
python src/preprocess.py
```
