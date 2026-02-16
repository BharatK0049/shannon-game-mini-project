from collections import Counter
from typing import List
from ngram_model import NGramModel
import math

class ShannonGame:
    def __init__(self, model: NGramModel):
        self.model = model
        self.guess_counts = Counter()

    def play(self, tokens: List[str]):
            n = self.model.n
            # Slide through tokens based on N
            for i in range(len(tokens) - (n - 1)):
                context = tuple(tokens[i : i + n - 1])
                true_word = tokens[i + n - 1]

                ranked_guesses = self.model.get_ranked_predictions(context)

                if not ranked_guesses:
                    continue

                try:
                    guess_rank = ranked_guesses.index(true_word) + 1
                except ValueError:
                    guess_rank = len(ranked_guesses) + 1

                self.guess_counts[guess_rank] += 1

    def report(self):
        """
        Print a summary of the Shannon game results.
        """
        total = sum(self.guess_counts.values())
        print(f"\nTotal word predictions evaluated: {total}\n")

        # Show top 5 ranks to keep output clean
        for rank in sorted(self.guess_counts)[:5]:
            count = self.guess_counts[rank]
            percentage = 100 * count / total
            print(f"Guess {rank}: {count} times ({percentage:.2f}%)")

        first_guess_accuracy = (
            100 * self.guess_counts[1] / total
            if total > 0 else 0
        )
        print(f"\nFirst-guess accuracy: {first_guess_accuracy:.2f}%")

    def estimate_entropy(self):
            total = sum(self.guess_counts.values())
            if total == 0:
                return 0.0

            entropy = 0.0
            for rank, count in self.guess_counts.items():
                probability = count / total
                # Shannon entropy formula based on rank
                entropy += probability * math.log2(rank)

            return entropy

    def estimate_redundancy(self, entropy):
        """
        Calculating redundancy based on entropy
        param: entropy
        """
        max_entropy = math.log2(27) # 26 letters + whitespace
        redundancy = 1 - (entropy / max_entropy)

        return redundancy