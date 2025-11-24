import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "features_kb.csv"
OUTPUT_FILE = "represented_data.csv"

def perform_feature_representation(df):
    """
    Performs Feature Representation via Normalization.
    Ref: Lecture 2, Slide 36 ("Values Normalization")
    """

    # Separate features from the label
    # We only want to normalize the numeric features, not the label
    feature_cols = [col for col in df.columns if col != 'Label']
    X = df[feature_cols]
    y = df['Label']

    print("Features before normalization (First 5 rows):")
    print(X.head())

    # --- Normalization Strategy: Min-Max Scaling ---
    # Motivation: As per Lecture 2, Slide 36: "Normalize all features' values... [0,1]"
    # This prevents features with large scales (like WordCount ~500) from dominating 
    # features with small scales (like Keyword Counts ~5) during model training.
    scaler = MinMaxScaler()
    
    # Fit and transform the features
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame to keep column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Reattach the label
    final_df = pd.concat([y, X_scaled_df], axis=1)
    
    return final_df

def main():
    input_path = Path(INPUT_FILE)

    df = pd.read_csv(input_path)
    
    # Apply Representation (Normalization)
    represented_df = perform_feature_representation(df)
    
    # Save
    represented_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved represented (normalized) data to {OUTPUT_FILE}")
    print("Data values are now scaled between 0 and 1.")

if __name__ == "__main__":
    main()