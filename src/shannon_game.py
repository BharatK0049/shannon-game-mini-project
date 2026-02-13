from collections import Counter
from typing import List
from ngram_model import TrigramModel
import math

class ShannonGame:
    def __init__(self, model: TrigramModel):
        self.model = model
        self.guess_counts = Counter()

    def play(self, text: str):
        """
        Play the Shannon game on test text.
        Records how many guesses were needed for each character.
        """
        for i in range(len(text) - 2):
            context = text[i:i+2]
            true_char = text[i+2]

            ranked_guesses: List[str] = self.model.get_ranked_predictions(context)

            # If we have never seen this context, skip it
            if not ranked_guesses:
                continue

            # Find guess rank
            try:
                guess_rank = ranked_guesses.index(true_char) + 1
            except ValueError:
                # Character never appeared after this context in training
                guess_rank = len(ranked_guesses) + 1

            self.guess_counts[guess_rank] += 1

    def report(self):
        """
        Print a summary of the Shannon game results.
        """
        total = sum(self.guess_counts.values())
        print(f"\nTotal predictions evaluated: {total}\n")

        for rank in sorted(self.guess_counts):
            count = self.guess_counts[rank]
            percentage = 100 * count / total
            print(f"Guess {rank}: {count} times ({percentage:.2f}%)")

        first_guess_accuracy = (
            100 * self.guess_counts[1] / total
            if total > 0 else 0
        )
        print(f"\nFirst-guess accuracy: {first_guess_accuracy:.2f}%")


    def demo(self, text: str, start_index: int, num_steps: int = 10):
        """
        Demonstrate live Shannon game predictions step-by-step.
        """
        print("\n--- SHANNON GAME DEMO ---\n")

        for i in range(start_index, start_index + num_steps):
            context = text[i:i+2]
            true_char = text[i+2]

            ranked = self.model.get_ranked_predictions(context)

            print(f"Context: '{context}'")
            print(f"Actual next character: '{true_char}'")

            if not ranked:
                print("No data for this context.\n")
                continue

            for idx, guess in enumerate(ranked[:5], start=1):
                mark = "✅" if guess == true_char else "❌"
                print(f"Guess {idx}: '{guess}' {mark}")

                if guess == true_char:
                    break

            print("-" * 30)

    def estimate_entropy(self):
        """
        Estimate entropy (bits per character) from guess-rank distribution.
        """
        total = sum(self.guess_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for rank, count in self.guess_counts.items():
            probability = count / total
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