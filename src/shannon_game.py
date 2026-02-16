from collections import Counter
from typing import List
from ngram_model import NGramModel
import math

class ShannonGame:
    """
    Implements a word-level Shannon Game to measure language predictability.
    
    This class simulates the process of guessing the next word in a sequence 
    given a specific context, allowing for the empirical calculation of 
    entropy, perplexity, and redundancy.
    """

    def __init__(self, model: NGramModel):
        """
        Initializes the game with a trained N-Gram language model.

        Args:
            model (NGramModel): The trained N-Gram model used to provide
                                ranked word predictions.
        """
        self.model = model
        # Records the rank at which the correct word was found for every prediction.
        # This distribution is the basis for estimating entropy.
        self.guess_counts = Counter()

    def play(self, tokens: List[str]):
        """
        Evaluates the model's predictive power by 'playing' through test data.

        For each word in the test set, the model uses the preceding (n-1) words 
        as context to guess the actual next word.

        Args:
            tokens (List[str]): A list of words (tokens) from the test corpus.
        """
        n = self.model.n
        # Slide through the tokens using a window determined by the N-Gram size
        for i in range(len(tokens) - (n - 1)):
            # The context is the tuple of words preceding the target word
            context = tuple(tokens[i : i + n - 1])
            true_word = tokens[i + n - 1]

            # Get the list of all possible next words, ranked by their frequency in training
            ranked_guesses = self.model.get_ranked_predictions(context)

            # Skip the context if the model has no data for it (out-of-vocabulary context)
            if not ranked_guesses:
                continue

            try:
                # Find the 'rank' (1st guess, 2nd guess, etc.) of the actual word
                guess_rank = ranked_guesses.index(true_word) + 1
            except ValueError:
                # If the word never appeared after this context in training, 
                # treat it as being just outside the current prediction list.
                guess_rank = len(ranked_guesses) + 1

            self.guess_counts[guess_rank] += 1

    def report(self):
        """
        Prints a summary of prediction statistics, including accuracy and distribution.
        
        This fulfills the requirement to provide justification for the model's 
        predictive quality.
        """
        total = sum(self.guess_counts.values())
        print(f"\nTotal word predictions evaluated: {total}\n")

        # Display the distribution for the top 5 most common guess ranks
        for rank in sorted(self.guess_counts)[:5]:
            count = self.guess_counts[rank]
            percentage = 100 * count / total
            print(f"Guess {rank}: {count} times ({percentage:.2f}%)")

        # First-guess accuracy represents how often the most probable word was correct
        first_guess_accuracy = (
            100 * self.guess_counts[1] / total
            if total > 0 else 0
        )
        print(f"\nFirst-guess accuracy: {first_guess_accuracy:.2f}%")

    def estimate_entropy(self):
        """
        Estimates the average entropy (bits per word) from the guess-rank distribution.

        Based on Shannon's theory, if the correct word is found at rank 'r', it 
        contributes log2(r) bits of information.

        Returns:
            float: The estimated entropy H = Σ P(r) * log2(r).
        """
        total = sum(self.guess_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for rank, count in self.guess_counts.items():
            # Calculate the probability of the correct word appearing at this rank
            probability = count / total
            # Accumulate entropy using the log2 of the rank
            entropy += probability * math.log2(rank)

        return entropy

    def estimate_redundancy(self, entropy: float):
        """
        Calculates language redundancy based on the estimated entropy.

        Redundancy measures how much of the language is determined by its 
        structure rather than choice.

        Note: The current max_entropy calculation (log2 of 27) is a carryover
        from character-level models (26 letters + space). For word-level 
        models, this typically uses the log2 of the vocabulary size.

        Args:
            entropy (float): The estimated entropy value from estimate_entropy().

        Returns:
            float: The redundancy value R = 1 - (H / H_max).
        """
        # H_max for character models: log2(27)
        max_entropy = math.log2(27) 
        redundancy = 1 - (entropy / max_entropy)

        return redundancy