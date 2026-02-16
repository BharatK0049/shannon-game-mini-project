from pathlib import Path
from ngram_model import NGramModel
from shannon_game import ShannonGame
from spell_checker import SpellChecker

def load_tokens(path: str):
    """Reads file and returns a list of words."""
    text = Path(path).read_text()
    return text.split()

def run_interactive_menu(model, checker, game):
    while True:
        print("\n=== INTERACTIVE NLP SESSION ===")
        print("1. Predict word (Shannon Game)")
        print("2. Spell Check a sentence (Noisy Channel)")
        print("3. Exit")
        
        choice = input("\nSelect an option: ")

        if choice == '1':
            # Fulfills: "Predicting the word that is not there in the sentence"
            phrase = input("Enter any phrase to predict the next word: ").lower().split()
            
            if phrase:
                # DYNAMIC LOGIC: Take as many words as available, up to the model's cap (n-1)
                max_context_len = model.n - 1
                context_len = min(len(phrase), max_context_len)
                context = tuple(phrase[-context_len:])
                
                preds = model.get_ranked_predictions(context)
                
                print(f"\nUsing a {len(context)}-word context: {context}")
                if preds:
                    print(f"Top 5 predictions: {preds[:5]}")
                    # JUSTIFICATION: Required by assignment notes
                    print(f"Justification: Prediction based on highest P(word|context) for a {len(context)+1}-gram.")
                else:
                    print("No predictions found for this specific context in the training data.")
            else:
                print("Please enter at least one word.")

        elif choice == '2':
            # Fulfills: "Spell check noisy channel"
            sent = input("Enter a sentence to correct: ")
            corrected = checker.correct_sentence(sent)
            print(f"\nCorrected: {corrected}")
            print(f"Justification: Used Noisy Channel argmax P(w|c)P(c) with dynamic context window.")

        elif choice == '3':
            break

def main():
    # 1. SETUP & TRAINING
    train_tokens = load_tokens("data/brown_train.txt")
    test_tokens = load_tokens("data/brown_test.txt")


    n_value = 5
    print(f"Training {n_value}-Gram Model...")
    model = NGramModel(n=n_value)
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