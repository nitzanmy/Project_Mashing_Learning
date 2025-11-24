import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "features_tfidf.csv" 
OUTPUT_FILE = "represented_data.csv"

def perform_feature_representation(df):
    """
    Performs Feature Representation via Normalization (Standard 0-1).
    """
    feature_cols = [col for col in df.columns if col != 'Label']
    X = df[feature_cols]
    y = df['Label']

    print("Features before normalization (First 5 rows):")
    print(X.head())

    # Standard Min-Max Scaling (0 to 1)
    scaler = MinMaxScaler() 
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    final_df = pd.concat([y, X_scaled_df], axis=1)
    
    return final_df

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(input_path)
    represented_df = perform_feature_representation(df)
    represented_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved represented data to {OUTPUT_FILE}")
    print("Data values are scaled between 0 and 1.")

if __name__ == "__main__":
    main()