import re
from typing import List, Set

class SpellChecker:
    def __init__(self, trigram_model):
        self.lm = trigram_model
        # Get a set of all unique words from our training data
        self.vocab = set()
        for counter in self.lm.model.values():
            self.vocab.update(counter.keys())

    def edits1(self, word: str) -> Set[str]:
        """Generate all strings that are 1 edit away from 'word'."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def get_candidates(self, word: str) -> List[str]:
        """Find known words that are 0, 1, or 2 edits away."""
        if word in self.vocab: return [word]
        
        # Priority: 1 edit away, then 2 edits away
        e1 = self.edits1(word)
        candidates = [w for w in e1 if w in self.vocab]
        if candidates: return candidates
        
        e2 = {e2 for e1_word in e1 for e2 in self.edits1(e1_word)}
        candidates = [w for w in e2 if w in self.vocab]
        
        return candidates if candidates else [word]

    def correct(self, word: str, context: tuple) -> str:
        """Apply Noisy Channel: P(w|c) * P(c)"""
        candidates = self.get_candidates(word)
        
        # P(c) is the probability from our Trigram Model
        # If the word isn't in our trigram context, we use its frequency
        best_word = word
        max_score = -1
        
        for cand in candidates:
            # P(c) - probability of candidate given the 2 previous words
            # For simplicity, we'll use raw counts from the model
            score = self.lm.model[context].get(cand, 0.1) # 0.1 as a tiny smoothing value
            
            if score > max_score:
                max_score = score
                best_word = cand
                
        return best_word
    
    def correct_sentence(self, sentence: str) -> str:
        words = sentence.lower().split()
        corrected_words = []
        
        for i, word in enumerate(words):
            # We need the previous two words for context
            if i == 0:
                context = ("<s>", "<s>") # Start of sentence markers
            elif i == 1:
                context = ("<s>", words[0])
            else:
                context = (corrected_words[i-2], corrected_words[i-1])
                
            corrected_word = self.correct(word, context)
            corrected_words.append(corrected_word)
            
        return " ".join(corrected_words)