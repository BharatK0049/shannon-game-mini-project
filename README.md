# Shannon Game

## Overview

This project implements a **Shannon game** for English text using a **character-level trigram (3-gram) model**, inspired by Claude Shannon’s paper *Prediction and Entropy of Printed English (1951)*.

The goal is to demonstrate Shannon’s core idea computationally:

> *The predictability of language can be measured by how difficult it is to guess the next symbol given prior context.*

Rather than reproducing Shannon’s mathematical proofs, this project focuses on **empirically measuring predictability** using statistical language models.

---

## How Prediction Works

The Brown corpus is split into a **training set** and a **testing (evaluation) set**.

- The **training set** is used only to learn statistical patterns of English.
- The **testing set** is used to play the Shannon game and evaluate predictability.

This separation ensures the experiment measures **generalization**, not memorization.

---

### Training Set

Using the training set (`brown_train.txt`), the trigram model records:

- How frequently each character follows a given **two-character context**

For example, from the training data:

`"th" → 'e' occurs most often, followed by 'a', 'i', etc.`


No prediction or guessing is performed on the training data.  
It is used **only** to learn frequency statistics.

---

### Testing Set

Using the testing set (`brown_test.txt`):

- The **two-character context** is taken from the test text itself
- The **actual next character** is also taken from the test text

The model then ranks possible next characters **using only statistics learned from the training corpus**.

In effect, the model is asked:

> *“Given this unseen context, how difficult is it to predict what actually comes next?”*

---

### What Do the Number of Guesses Mean?

For each character in the test text, the model:

1. Ranks possible next characters by frequency
2. Treats the most frequent character as the **first guess**
3. If incorrect, checks the second most frequent, then the third, and so on
4. Records the **rank at which the correct character appears**

Interpretation:

- **Low guess rank (e.g., Guess 1 or Guess 2):** highly predictable
- **High guess rank:** many plausible continuations

These guess ranks quantify **prediction difficulty**.

---

## Estimating Entropy and Redundancy

Entropy and redundancy are computed **after the Shannon game**, using the distribution of guess ranks.

If the correct character is found at rank \( r \), it contributes approximately:


log_2(r) {bits of information}

Entropy is estimated as the **average** of this quantity over all predictions in the test corpus.

Redundancy is then computed relative to the maximum possible entropy:

Redundancy = 1 - (H / H_max)

where  H_max = log_2(27) for the 26 letters plus space.

---

## Perplexity

Perplexity is derived directly from entropy:

Perplexity = 2^H

Perplexity represents the **effective number of plausible next characters** after accounting for context.

For example:
- \( H = 1.35 \) bits → perplexity ≈ **2.5**
- This means the model behaves as if it is choosing among **2–3 plausible characters on average**

---

## Project Structure

```text
shannon_game/
├── data/
│   ├── brown_train.txt      # Training corpus
│   └── brown_test.txt       # Test corpus (unseen during training)
├── src/
│   ├── preprocess.py        # Loads and cleans the Brown corpus
│   ├── ngram_model.py       # Trigram language model
│   ├── shannon_game.py      # Shannon prediction game logic
│   └── main.py              # Runs the full experiment
├── requirements.txt
└── README.md


## Module Descriptions

`preprocess.py`

- Downloads the Brown corpus using NLTK
- Converts text to lowercase
- Removes punctuation and non-alphabetic characters
- Treats space as a valid character
- Splits the corpus into:
- 80% training data
- 20% test data

Saves cleaned text as .txt files for transparency

`ngram_model.py`

- Implements a trigram (N = 3) character model
- Uses:
    - 2-character context → next character prediction
- Stores frequency counts of next characters for each context
- Provides ranked predictions based on frequency

This models the statistical structure of English at the character level.

`shannon_game.py`

- Implements the Shannon game
- For each character in the test text:
    - Uses the previous two characters as context
    - Guesses the next character in order of likelihood
    - Records the number of guesses required
- Aggregates guess-rank statistics
- Estimates entropy and redundancy

This simulates Shannon’s original prediction experiments programmatically.

`main.py`

- Loads training and test text
- Trains the trigram model
- Runs the Shannon game
- Prints prediction statistics, entropy, redundancy, and perplexity

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
