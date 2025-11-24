import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "segmented_data.csv"
OUTPUT_FILE = "features_kb.csv"

# --- Domain Knowledge Definitions ---
# Ref: Lecture 2, Slide 21 ("Humans usually define the set of features...")
# We use our domain expertise to pick words that distinguish categories 
# even after pre-processing.

# Specific domain keywords
SPORT_KEYWORDS = {'match', 'win', 'loss', 'score', 'team', 'league', 'player', 'coach', 'cup', 'medal', 'game', 'champion'}
POLITICS_KEYWORDS = {'government', 'minister', 'law', 'parliament', 'election', 'party', 'vote', 'campaign', 'policy', 'senate'}
CULTURE_KEYWORDS = {'film', 'movie', 'music', 'art', 'book', 'show', 'performance', 'star', 'festival', 'theatre', 'album'}
NEWS_KEYWORDS = {'breaking', 'report', 'update', 'police', 'investigation', 'accident', 'local', 'global', 'world', 'official', 'statement', 'daily'}

def count_keywords(text, keyword_set):
    """Counts how many words from the keyword_set appear in the text."""

    words = text.split()
    count = sum(1 for word in words if word in keyword_set)
    return count

def extract_features(df):
    """
    Extracts Knowledge-Based Features (Specific Features).
    Ref: Lecture 2, Slide 20 (Specific/Knowledge Based taxonomy)
    """
    # Fail Fast: Ensure input data is valid
    assert 'Segmented_Content' in df.columns, "Missing 'Segmented_Content' column."
    
    print("Extracting Knowledge-Based Features...")
    
    # 1. Structural Feature: Document Length (Number of Words)
    # Ref: Lecture 2, Slide 21 (Example: "# of pages")
    df['KB_WordCount'] = df['Segmented_Content'].apply(lambda x: len(str(x).split()))

    # 2. Structural Feature: Average Word Length
    # Ref: Lecture 2, Slide 22 (Example: "Size", "Shape")
    def get_avg_word_len(text):
        words = str(text).split()
        if not words: return 0
        return sum(len(w) for w in words) / len(words)
    
    df['KB_AvgWordLen'] = df['Segmented_Content'].apply(get_avg_word_len)

    # 3. Content Features: Domain Specific Keyword Counts
    # Ref: Lecture 2, Slide 21 (Example: "# 'love' word", "# 'past' word")
    # We count occurrences of words from our knowledge bases
    df['KB_Sport_Count'] = df['Segmented_Content'].apply(lambda x: count_keywords(x, SPORT_KEYWORDS))
    df['KB_Politics_Count'] = df['Segmented_Content'].apply(lambda x: count_keywords(x, POLITICS_KEYWORDS))
    df['KB_Culture_Count'] = df['Segmented_Content'].apply(lambda x: count_keywords(x, CULTURE_KEYWORDS))
    df['KB_News_Count'] = df['Segmented_Content'].apply(lambda x: count_keywords(x, NEWS_KEYWORDS))

    # Drop the raw text column (we now have numerical representations)
    # We keep 'Label' for the next steps.
    output_df = df[['Label', 'KB_WordCount', 'KB_AvgWordLen', 
                    'KB_Sport_Count', 'KB_Politics_Count', 'KB_Culture_Count', 'KB_News_Count']]
    
    return output_df

def main():
    input_path = Path(INPUT_FILE)

    df = pd.read_csv(input_path)
    
    # Extract features
    kb_features_df = extract_features(df)
    
    # Save
    kb_features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved Knowledge-Based features to {OUTPUT_FILE}")
    print(f"Features created: {list(kb_features_df.columns)}")

if __name__ == "__main__":
    main()