# Shannon Game

## Overview
This project implements a Shannon game for English text using a character-level trigram (3-gram) model, inspired by Claude Shannon’s paper Prediction and Entropy of Printed English (1951).

The goal is to demonstrate the core idea computationally:
```
*The predictability of language can be measured by how difficult it is to guess the next symbol given prior context.*
```

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
```

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

This simulates Shannon’s original prediction experiments programmatically.

`main.py`

- Loads training and test text
- Trains the trigram model
- Runs the Shannon game
- Prints a summary of prediction results

## How to Run

```bash
pip install -r requirements.txt
```

```
python src/preprocess.py
```

```
python src/preprocess.py
```
