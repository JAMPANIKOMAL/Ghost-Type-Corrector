import re
import os
import random
import sys
from tqdm import tqdm # Import tqdm for the progress bar

def clean_line(text):
    """
    Cleans a single line of text from the corpus.
    - Removes the starting line number (e.g., "1 \t")
    - Converts to lowercase
    - Removes all punctuation, symbols, and standalone numbers
    - Normalizes whitespace
    """
    
    # 1. Strip the line number (e.g., "1 \t" or "10 \t")
    match = re.search(r'^\d+\t(.*)', text)
    if match:
        text = match.group(1)
    
    # 2. Force to lowercase
    text = text.lower()
    
    # 3. Remove punctuation, symbols, and numbers
    # This regex [^a-z\s] means "find anything that is NOT (^) a letter (a-z) or whitespace (\s)"
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Normalize whitespace (replace multiple spaces/tabs with one)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Filter out very short or empty lines
    if len(text.split()) < 3:
        return None # Return None to indicate this line should be skipped
    
    return text

def add_noise_to_sentence(sentence, noise_level=0.15):
    """
    Takes a clean sentence and randomly introduces typos (noise).
    
    noise_level: The probability (e.g., 0.15 = 15%) that a word will be "noised".
    """
    
    words = sentence.split()
    new_sentence_words = []
    
    # All possible letters for insertion/substitution
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for word in words:
        # We only add noise if the word is long enough AND
        # a random chance (between 0.0 and 1.0) is below our noise level
        if random.random() < noise_level and len(word) > 3:
            
            # 1. Randomly pick a type of typo
            typo_type = random.choice(['delete', 'insert', 'substitute', 'swap'])
            
            if typo_type == 'delete':
                pos = random.randint(0, len(word) - 1)
                noised_word = word[:pos] + word[pos+1:]
            
            elif typo_type == 'insert':
                pos = random.randint(0, len(word))
                char = random.choice(alphabet)
                noised_word = word[:pos] + char + word[pos:]
            
            elif typo_type == 'substitute':
                pos = random.randint(0, len(word) - 1)
                char = random.choice(alphabet)
                noised_word = word[:pos] + char + word[pos+1:]
            
            elif typo_type == 'swap':
                if len(word) > 1:
                    pos = random.randint(0, len(word) - 2)
                    noised_word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                else:
                    noised_word = word
            
            new_sentence_words.append(noised_word)
        
        else:
            new_sentence_words.append(word)
            
    return ' '.join(new_sentence_words)

def main():
    """
    Main function to run the data preprocessing pipeline.
    """
    
    # Define our file paths
    # Note: This script is in 'src', so we go up one ('..') to the 'ai_model' folder
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', 'data')
    
    input_corpus = os.path.join(data_dir, 'corpus.txt')
    output_clean = os.path.join(data_dir, 'train_clean.txt')
    output_noisy = os.path.join(data_dir, 'train_noisy.txt')

    print(f"--- Starting Data Preprocessing ---")
    print(f"Input corpus: {input_corpus}")
    print(f"Output clean data: {output_clean}")
    print(f"Output noisy data: {output_noisy}")
    
    try:
        # Open the output files
        with open(input_corpus, 'r', encoding='utf-8') as f_in, \
             open(output_clean, 'w', encoding='utf-8') as f_clean, \
             open(output_noisy, 'w', encoding='utf-8') as f_noisy:
            
            # Use tqdm to show a progress bar
            # We count the lines first to give tqdm a total
            print("Counting lines in corpus (this may take a moment)...")
            line_count = sum(1 for line in open(input_corpus, 'r', encoding='utf-8'))
            print(f"Found {line_count} lines.")
            
            print("Processing corpus...")
            # Rewind the file (or just use the new f_in)
            f_in.seek(0)
            
            for line in tqdm(f_in, total=line_count, desc="Cleaning and Noising"):
                # 1. Clean the line
                clean_sentence = clean_line(line)
                
                # 2. Skip if the line was too short (clean_line returned None)
                if clean_sentence:
                    # 3. Create the noisy version
                    noisy_sentence = add_noise_to_sentence(clean_sentence)
                    
                    # 4. Write to our two new files
                    f_clean.write(clean_sentence + '\n')
                    f_noisy.write(noisy_sentence + '\n')

        print("\n--- Data Preprocessing Complete ---")
        print("Created 'train_clean.txt' and 'train_noisy.txt' in ai_model/data/")

    except FileNotFoundError:
        print(f"ERROR: Could not find '{input_corpus}'")
        print("Please make sure the 'corpus.txt' file is in the 'ai_model/data/' directory.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    # This block ensures the 'main' function is only run
    # when you execute this file directly as a script
    main()

