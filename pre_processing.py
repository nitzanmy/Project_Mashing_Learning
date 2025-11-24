import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "sensed_data.csv"
OUTPUT_FILE = "preprocessed_data.csv"

# --- NLTK Setup (Fail Fast & Robustness) ---
def download_nltk_resources():
    """Ensures necessary NLTK datasets are available."""
    resources = ['punkt', 'wordnet', 'stopwords', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            print(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

download_nltk_resources()
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Applies cleaning logic relevant to Text Classification.
    Ref: Lecture 2, Slide 11 (Cleaning noise)
    
    1. Unicode fix
    2. Lowercasing
    3. Regex (remove numbers/punctuation)
    4. Stopwords & Lemmatization
    """

    # 1. Remove non-ASCII (Web scraping noise)
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 2. Lowercase (Normalization)
    text = text.lower()
    
    # 3. Keep only alphabetic characters (Remove numbers and punctuation)
    # We do not need numbers for broad category classification (News vs Sport).
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenization
    words = text.split()
    
    # 5. Stop Word Removal & Lemmatization
    # Ref: Lecture 2, Slide 87 (Compact size, faster processing)
    processed_words = [
        LEMMATIZER.lemmatize(word, pos='v') 
        for word in words 
        if word not in STOP_WORDS and len(word) > 2 # Skip 1-2 letter garbage
    ]
    
    return " ".join(processed_words)

def main():
    # 1. Load Data

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}.")

    # 2. Handling Missing Values
    # Ref: Lecture 2, Slide 12 (Missing values -> delete) [cite: 127]
    initial_count = len(df)
    df = df.dropna(subset=['BodyText', 'Label'])
    
    # Fail Fast: Check if we lost too much data
    # if len(df) < initial_count:
    #     print(f"Dropped {initial_count - len(df)} rows due to missing values.")
    
    # assert len(df) > 0, "Error: Dataset is empty after dropping NaNs."

    # 3. Handling Redundancy (Duplicates)
    # Ref: Lecture 2, Slide 11 (Any redundancy? -> eliminate) 
    # Duplicates in text classification cause bias in the test set.
    duplicate_count = df.duplicated(subset=['BodyText']).sum()
    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate articles.")
        df = df.drop_duplicates(subset=['BodyText'])

    # 4. Text Cleaning & Normalization (The "Cleaning Noise" step)
    # Ref: Lecture 2, Slide 11 
    print("Processing text (Cleaning, Lemmatizing)...")
    
    # We process both Body and Title as both are useful features
    df['CleanedBody'] = df['BodyText'].apply(clean_text)
    df['CleanedTitle'] = df['Title'].apply(clean_text)

    # 5. Final Sanity Check
    # Remove rows that became empty strings after cleaning (e.g., a body with only numbers)
    # df = df[df['CleanedBody'].str.strip().astype(bool)]
    
    print(f"Final dataset size: {len(df)} rows.")

    # 6. Save
    # We keep the original Label but use the Cleaned text for the next stages
    output_columns = ['Label', 'CleanedTitle', 'CleanedBody'] 
    df[output_columns].to_csv(OUTPUT_FILE, index=False)
    
    print(f"Success! Pre-processed data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()