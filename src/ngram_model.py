from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class NGramModel:
    def __init__(self, n: int = 5):
        """
        :param n: The maximum context depth (the cap).
        """
        self.n = n
        # A single dictionary can store tuples of varying lengths as keys
        self.model: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

    def train(self, tokens: List[str]):
        """
        Trains the model for all levels from 2-gram up to n-gram.
        """
        # Train every level from 2 up to the cap
        for level in range(2, self.n + 1):
            context_size = level - 1
            for i in range(len(tokens) - context_size):
                context = tuple(tokens[i : i + context_size])
                next_word = tokens[i + context_size]
                self.model[context][next_word] += 1

    def get_ranked_predictions(self, context: Tuple[str, ...]) -> List[str]:
        """
        Finds predictions for the specific context provided.
        """
        if context not in self.model:
            return []

        counter = self.model[context]
        return [word for word, _ in counter.most_common()]