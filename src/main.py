from pathlib import Path
from ngram_model import TrigramModel
from shannon_game import ShannonGame
from spell_checker import SpellChecker

def load_tokens(path: str):
    """Reads file and returns a list of words."""
    text = Path(path).read_text()
    return text.split()

def run_interactive_menu(model, checker, game):
    """Handles real-world user inputs as requested."""
    while True:
        print("\n=== INTERACTIVE NLP SESSION ===")
        print("1. Predict word (Shannon Game)")
        print("2. Spell Check a sentence (Noisy Channel)")
        print("3. Exit")
        
        choice = input("\nSelect an option: ")

        if choice == '1':
            # Fulfills: "Predicting the word that is not there in the sentence"
            phrase = input("Enter two words to predict the next (e.g., 'of the'): ").lower().split()
            if len(phrase) >= 2:
                context = tuple(phrase[-2:])
                preds = model.get_ranked_predictions(context)
                print(f"\nTop 5 predictions for {context}: {preds[:5]}")
                print(f"Justification: These words have the highest P(c) in the training corpus.")
            else:
                print("Please enter at least two words.")

        elif choice == '2':
            # Fulfills: "Spell check noisy channel"
            sent = input("Enter a sentence to correct: ")
            # Note: Ensure you added 'correct_sentence' to your SpellChecker class
            corrected = checker.correct_sentence(sent)
            print(f"\nCorrected: {corrected}")
            print(f"Justification: Used Noisy Channel argmax P(w|c)P(c).")

        elif choice == '3':
            print("Exiting application.")
            break

def main():
    # 1. SETUP & TRAINING
    train_tokens = load_tokens("data/brown_train.txt")
    test_tokens = load_tokens("data/brown_test.txt")

    print(f"Training on {len(train_tokens)} words...")
    model = TrigramModel()
    model.train(train_tokens)
    checker = SpellChecker(model)
    game = ShannonGame(model)

    # 2. AUTOMATED EVALUATION (Fulfills assignment metrics)
    print(f"\nEvaluating on {len(test_tokens)} test words...")
    game.play(test_tokens)
    game.report()
    
    entropy = game.estimate_entropy()
    perplexity = 2 ** entropy
    print(f"Estimated Perplexity: {perplexity:.2f}") # Requirement: "Perplexity values should be given"

    # 3. INTERACTIVE MODE (Fulfills: "Actual real inputs")
    run_interactive_menu(model, checker, game)

if __name__ == "__main__":
    main()