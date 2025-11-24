import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "segmented_data.csv"
OUTPUT_FILE = "features_tfidf.csv"

def extract_features(df):
    """
    Extracts Features using ONLY the Generic Method (TF-IDF).
    Ref: Lecture 2, Slide 35 (Term Frequency - Inverse Document Frequency)
    """
    print("Extracting Generic Features (TF-IDF)...")
    
    # 1. Initialize TF-IDF
    # max_features=1000: We take the top 1000 most distinguishing words.
    # stop_words='english': Removes "the", "is", "at", etc.
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # 2. Fit and Transform
    # We use the segmented content (Title + Body)
    tfidf_matrix = tfidf.fit_transform(df['Segmented_Content'].fillna(""))
    
    # 3. Convert to DataFrame
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    print(f"Created {len(feature_names)} features.")
    
    # 4. Attach Label
    final_df = pd.concat([df['Label'], tfidf_df], axis=1)
    
    return final_df

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(input_path)
    final_df = extract_features(df)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved TF-IDF features to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()