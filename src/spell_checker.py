import re
from typing import List, Set

class SpellChecker:
    """
    A context-aware spell checker using the Noisy Channel Model and N-Gram probabilities.
    
    This class identifies potential spelling errors by generating candidates within
    a specific edit distance and selects the most likely correction based on the
    surrounding word context provided by an N-Gram language model.
    """

    def __init__(self, n_gram_model):
        """
        Initializes the SpellChecker with a trained language model.

        Args:
            n_gram_model: A trained NGramModel instance containing word 
                          frequency counts and contexts.
        """
        self.lm = n_gram_model
        # Build the vocabulary: a set of all unique words seen during training
        # to distinguish between valid words and potential misspellings.
        self.vocab = set()
        for counter in self.lm.model.values():
            self.vocab.update(counter.keys())

    def edits1(self, word: str) -> Set[str]:
        """
        Generates all possible strings that are exactly one edit away from the word.

        The edits include:
        1. Deletions: Removing one character.
        2. Transpositions: Swapping two adjacent characters.
        3. Replaces: Changing one character to another.
        4. Insertions: Adding a new character.

        Args:
            word (str): The input word to transform.

        Returns:
            Set[str]: A set of unique strings representing all 1-edit variations.
        """
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        # Split the word into all possible left (L) and right (R) pairs
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes    = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts    = [L + c + R for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)

    def get_candidates(self, word: str) -> List[str]:
        """
        Identifies valid candidate words from the vocabulary based on edit distance.

        Prioritizes candidates in this order:
        1. The word itself (if it is already in the vocabulary).
        2. Words that are 1 edit away.
        3. Words that are 2 edits away.

        Args:
            word (str): The word to find candidates for.

        Returns:
            List[str]: A list of known vocabulary words that are close to the input.
        """
        # Step 1: If word is valid, return it immediately
        if word in self.vocab: 
            return [word]
        
        # Step 2: Look for words 1 edit away
        e1 = self.edits1(word)
        candidates = [w for w in e1 if w in self.vocab]
        if candidates: 
            return candidates
        
        # Step 3: Look for words 2 edits away (edits of the edits)
        e2 = {e2 for e1_word in e1 for e2 in self.edits1(e1_word)}
        candidates = [w for w in e2 if w in self.vocab]
        
        # Fallback: If no known candidates are found, return the original word
        return candidates if candidates else [word]

    def correct(self, word: str, context: tuple) -> str:
        """
        Selects the best correction for a word using the Noisy Channel Model.

        Formula: argmax P(word | candidate) * P(candidate | context)
        In this implementation, we assume P(word | candidate) is roughly equal 
        for all edit-based candidates and focus on the Language Model probability P(c).

        Args:
            word (str): The potentially misspelled word.
            context (tuple): A tuple of preceding words (n-1) used for prediction.

        Returns:
            str: The most probable candidate word according to the language model.
        """
        candidates = self.get_candidates(word)
        
        best_word = word
        max_score = -1
        
        for cand in candidates:
            # P(c) - probability of the candidate given the n-gram context.
            # We use raw counts from the model for simplicity.
            # 0.1 is used as a tiny smoothing value for unseen context-word pairs.
            score = self.lm.model[context].get(cand, 0.1)
            
            if score > max_score:
                max_score = score
                best_word = cand
                
        return best_word
    
    def correct_sentence(self, sentence: str) -> str:
        """
        Processes and corrects an entire sentence word-by-word.

        This method maintains a sliding context window to ensure that each
        correction is informed by the words that came before it.

        Args:
            sentence (str): The input sentence string.

        Returns:
            str: The corrected sentence string.
        """
        # Tokenize and normalize the sentence
        words = sentence.lower().split()
        corrected_words = []
        
        for i, word in enumerate(words):
            # Determine the context window based on position
            if i == 0:
                # Start of sentence: Use placeholders
                context = ("<s>", "<s>") 
            elif i == 1:
                # Second word: Use one placeholder and the first (corrected) word
                context = ("<s>", corrected_words[0])
            else:
                # Standard case: Use the two previously corrected words as context
                context = (corrected_words[i-2], corrected_words[i-1])
            
            # Identify the best candidate for the current word
            corrected_word = self.correct(word, context)
            corrected_words.append(corrected_word)
            
        return " ".join(corrected_words)