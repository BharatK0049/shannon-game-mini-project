from collections import defaultdict, Counter
from typing import Dict


class TrigramModel:
    def __init__(self):
        """
        Maps a 2-character context to a Counter of next-character frequencies.
        Example:
            self.model["qu"] = Counter({"e": 1200, "i": 3})
        """
        self.model: Dict[str, Counter] = defaultdict(Counter)

    def train(self, text: str):
        """
        Build trigram counts from training text.
        """
        for i in range(len(text) - 2):
            context = text[i:i+2]
            next_char = text[i+2]
            self.model[context][next_char] += 1

    def get_ranked_predictions(self, context: str):
        """
        Given a 2-character context, return characters ranked
        from most likely to least likely.
        """
        if context not in self.model:
            return []

        counter = self.model[context]
        return [char for char, _ in counter.most_common()]
