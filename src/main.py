from pathlib import Path
from xml.parsers.expat import model
from ngram_model import TrigramModel
from shannon_game import ShannonGame


def load_tokens(path: str):
    """Reads file and returns a list of words."""
    text = Path(path).read_text()
    # simply split by whitespace since preprocess.py already cleaned it
    return text.split()


def main():
    train_tokens = load_tokens("data/brown_train.txt")
    test_tokens = load_tokens("data/brown_test.txt")

    print(f"Training on {len(train_tokens)} words...")
    model = TrigramModel()
    model.train(train_tokens)

    print(f"Playing Shannon Game on {len(test_tokens)} words...")
    game = ShannonGame(model)
    game.play(test_tokens)
    
    game.report()
    
    entropy = game.estimate_entropy()
    print(f"\nEstimated Entropy: {entropy:.3f} bits per word")
    
    # Perplexity = 2^Entropy
    perplexity = 2 ** entropy
    print(f"Estimated Perplexity: {perplexity:.2f}")

    test_context = ("of", "the")
    predictions = model.get_ranked_predictions(test_context)
    print(f"\nTop 5 words predicted after 'of the': {predictions[:5]}")

if __name__ == "__main__":
    main()