import json
import re

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
    for code in corpus:
        tokens = tokenize_code(code)
        for n in range(1, 5):  # Extract n-grams for n = 1 to 4
            all_ngrams.extend(list(ngrams(tokens, n)))
    frequencies = Counter(all_ngrams)
    return dict(frequencies.most_common(k))

if __name__ == "__main__":
    
    corpus_file = "/Users/esha/Desktop/Capstone/SWE-bench/python_dataset/python_corpus.py"  # File containing your Python corpus (one code snippet per line) (PLACEHOLDER)
    output_file = "trivial_ngrams.json"

    with open(corpus_file, "r") as file:
        corpus = [line.strip() for line in file]

    trivial_ngrams = generate_trivial_ngrams(corpus)

    with open(output_file, "w") as file:
        json.dump(trivial_ngrams, file)

    print(f"Trivial n-grams saved to {output_file}")