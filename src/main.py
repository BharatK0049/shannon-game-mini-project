from pathlib import Path
from ngram_model import TrigramModel
from shannon_game import ShannonGame


def load_text(path: str) -> str:
    return Path(path).read_text()


def main():
    train_text = load_text("data/brown_train.txt")
    test_text = load_text("data/brown_test.txt")

    # Train trigram model
    model = TrigramModel()
    model.train(train_text)

    # Play Shannon game
    game = ShannonGame(model)
    game.play(test_text)

    print("\nShowing live prediction demo:\n")
    game.demo(test_text, start_index=1000, num_steps=5)

    game.report()
    
    entropy = game.estimate_entropy()
    print(f"\nEstimated entropy: {entropy:.3f} bits per character")

    redundancy = game.estimate_redundancy(entropy)
    print(f"Estimated redundancy: {redundancy * 100:.2f}%")

if __name__ == "__main__":
    main()
