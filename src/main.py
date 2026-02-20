from pathlib import Path
from ngram_model import NGramModel
from shannon_game import ShannonGame
from spell_checker import SpellChecker

def load_tokens(path: str) -> list:
    """
    Reads a text file and returns a list of word tokens.

    Args:
        path (str): The file path to the cleaned corpus (e.g., 'data/brown_train.txt').

    Returns:
        list: A list of words split by whitespace.
    """
    text = Path(path).read_text()
    return text.split()

def run_interactive_menu(model: NGramModel, checker: SpellChecker, game: ShannonGame):
    """
    Provides a command-line interface for real-time NLP tasks.

    This menu allows users to test 'predicting the word that is not there' 
    and context-aware spell checking using dynamic input.

    Args:
        model (NGramModel): The trained multi-level N-Gram model.
        checker (SpellChecker): The Noisy Channel-based spell checker.
        game (ShannonGame): The session manager for entropy and rank stats.
    """
    while True:
        print("\n=== INTERACTIVE NLP SESSION ===")
        print("1. Predict word (Shannon Game)")
        print("2. Spell Check a sentence (Noisy Channel)")
        print("3. Exit")
        
        choice = input("\nSelect an option: ")

        if choice == '1':
            # Fulfills assignment task: "Predicting the word that is not there in the sentence"
            phrase = input("Enter any phrase to predict the next word: ").lower().split()
            
            if phrase:
                # DYNAMIC LOGIC: Determine context size based on available input, 
                # capped by the model's max N (n-1).
                max_context_len = model.n - 1
                context_len = min(len(phrase), max_context_len)
                context = tuple(phrase[-context_len:])
                
                preds = model.get_ranked_predictions(context)
                
                print(f"\nUsing a {len(context)}-word context: {context}")
                if preds:
                    print(f"Top 5 predictions: {preds[:5]}")
                    # JUSTIFICATION: Required by assignment notes to explain output logic
                    print(f"Justification: Prediction based on highest P(word|context) for a {len(context)+1}-gram.")
                else:
                    print("No predictions found for this specific context in the training data.")
            else:
                print("Please enter at least one word.")

        elif choice == '2':
            # Fulfills assignment task: "Spell check noisy channel"
            sent = input("Enter a sentence to correct: ")
            corrected = checker.correct_sentence(sent)
            print(f"\nCorrected: {corrected}")
            print(f"Justification: Used Noisy Channel argmax P(w|c)P(c) with dynamic context window.")

        elif choice == '3':
            print("Exiting interactive session.")
            break

def main():
    """
    Orchestrates the NLP pipeline: Training, Evaluation, and Interaction.
    """
    # 1. SETUP & TRAINING
    # Load tokenized Brown corpus data
    train_tokens = load_tokens("data/brown_train.txt")
    test_tokens = load_tokens("data/brown_test.txt")

    # Initialize model with a cap (e.g., 5-gram)
    n_value = 5
    print(f"Training {n_value}-Gram Model...")
    model = NGramModel(n=n_value)
    model.train(train_tokens)
    
    # Initialize application modules
    checker = SpellChecker(model)
    game = ShannonGame(model)

    # 2. AUTOMATED EVALUATION
    # Fulfills assignment requirement: "get it evaluated" and "Perplexity values should be given"
    print(f"\nEvaluating on {len(test_tokens)} test words...")
    game.play(test_tokens)
    game.report()
    
    # Perplexity (PP = 2^H) justifies model quality by showing average uncertainty
    entropy = game.estimate_entropy()
    redundancy = game.estimate_redundancy(entropy)
    print(f"Estimated Entropy: {entropy:.4f} bits/word")
    print(f"Estimated Redundancy: {redundancy:.2%}")
    perplexity = 2 ** entropy
    print(f"Estimated Perplexity: {perplexity:.2f}")

    # 3. INTERACTIVE MODE
    # Handles "Actual real inputs" from the user
    run_interactive_menu(model, checker, game)

if __name__ == "__main__":
    main()