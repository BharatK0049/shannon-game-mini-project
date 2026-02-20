from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class NGramModel:
    """
    A flexible N-Gram language model supporting multiple context depths.

    This model stores frequency counts of words following specific sequences 
    (contexts). It is designed to train on all levels from Bigrams up to a 
    specified cap, allowing for dynamic "backoff-style" predictions in the 
    Shannon Game and Spell Checker applications.
    """

    def __init__(self, n: int = 5):
        """
        Initializes the NGramModel with a maximum context depth.

        Args:
            n (int): The maximum 'n' for the n-gram (e.g., 5 for a 5-gram model).
                     This acts as the upper limit for context length.
        """
        self.n = n
        # A dictionary mapping a context (tuple of words) to a Counter of next words.
        # Counter stores how many times each word followed that specific context.
        self.model: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocab = set()

    def train(self, tokens: List[str]):
        """
        Trains the model by recording frequencies for all N-Gram levels.

        This method slides windows of varying sizes across the token list. 
        By training on all levels from 2-gram to n-gram, the model can 
        provide predictions even if a long context is not found in the training data.

        Args:
            tokens (List[str]): A list of word tokens from the training corpus.
        """
        self.vocab.update(tokens)
        # Iterate through each level from 2 (Bigram) up to the specified cap (N-gram)
        for level in range(2, self.n + 1):
            # Context size is always one less than the N-gram level
            context_size = level - 1
            
            # Slide a window across the tokens to capture context and the following word
            for i in range(len(tokens) - context_size):
                # Create a tuple of (context_size) words as the key
                context = tuple(tokens[i : i + context_size])
                # The word immediately following the context
                next_word = tokens[i + context_size]
                
                # Increment the frequency count for this word in this context
                self.model[context][next_word] += 1

    def get_ranked_predictions(self, context: Tuple[str, ...]) -> List[str]:
        """
        Retrieves possible next words for a given context, ranked by likelihood.

        This method provides the basis for 'predicting the word that is not there' 
        in the Shannon Game by identifying the most statistically probable 
        continuations.

        Args:
            context (Tuple[str, ...]): A sequence of words representing the current state.

        Returns:
            List[str]: A list of candidate words sorted from most frequent to least frequent.
        """
        # Check if this specific context exists in our training history
        if context not in self.model:
            return []

        # Retrieve the counter for this context
        counter = self.model[context]
        
        # Return only the words, sorted by their frequency (P(c))
        return [word for word, _ in counter.most_common()]