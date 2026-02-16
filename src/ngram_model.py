from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class TrigramModel:
    def __init__(self):
        """
        Maps a 2-word context to a Counter of next-word frequencies.
        Example:
            self.model[("the", "quick")] = Counter({"brown": 10, "red": 2})
        """
        # Context is now a tuple of words, not a string
        self.model: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    def train(self, tokens: List[str]):
        """
        Build trigram counts from a list of words (tokens).
        """
        # Iterate through the list of words
        for i in range(len(tokens) - 2):
            w1 = tokens[i]
            w2 = tokens[i+1]
            next_word = tokens[i+2]
            
            context = (w1, w2)
            self.model[context][next_word] += 1

    def get_ranked_predictions(self, context: Tuple[str, str]) -> List[str]:
        """
        Given a 2-word context, return words ranked
        from most likely to least likely.
        """
        if context not in self.model:
            return []

        counter = self.model[context]
        # Return just the words, sorted by frequency
        return [word for word, _ in counter.most_common()]