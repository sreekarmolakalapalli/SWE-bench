import json
import re
import pickle

from collections import Counter
from nltk.util import ngrams


def tokenize_code(code):
    """
    Tokenizes the given code string into meaningful tokens.
    """
    token_pattern = r"[A-Za-z_][A-Za-z0-9_]*|[\{\}\[\]\(\)\.;,]|[+\-*/=%<>!]+"
    tokens = re.findall(token_pattern, code)
    return tokens

def generate_trivial_ngrams(corpus, k=500):
    all_ngrams = []
    for n in range(1, 5):  # Extract n-grams for n = 1 to 4
        all_ngrams.extend(list(ngrams(corpus, n)))
    frequencies = Counter(all_ngrams)
    return dict(frequencies.most_common(k))

if __name__ == "__main__":
    
    corpus_file = "/Users/esha/Desktop/Capstone/SWE-bench/python_dataset/python_data.txt"  # File containing your Python corpus (one code snippet per line) (PLACEHOLDER)
    output_file = "trivial_ngrams.pkl"

    with open(corpus_file, "r") as file:
        corpus = file.read()
        tokenized_corpus = tokenize_code(corpus)

    trivial_ngrams = generate_trivial_ngrams(tokenized_corpus)

    with open(output_file, "wb") as file:
        pickle.dump(trivial_ngrams, file)

    print(f"Trivial n-grams saved to {output_file}")