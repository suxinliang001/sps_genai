import re
import random
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self, corpus):
        self.bigram_counts = defaultdict(Counter)
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for w1, w2 in zip(tokens, tokens[1:]):
                self.bigram_counts[w1][w2] += 1

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def generate_text(self, start_word, length):
        word = start_word.lower()
        result = [word]
        for _ in range(length - 1):
            next_words = self.bigram_counts.get(word)
            if not next_words:
                break
            words, counts = zip(*next_words.items())
            word = random.choices(words, weights=counts)[0]
            result.append(word)
        return " ".join(result)
