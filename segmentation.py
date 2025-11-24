import pandas as pd
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "preprocessed_data.csv"
OUTPUT_FILE = "segmented_data.csv"

def segment_data(df):
    """
    Performs Segmentation by isolating and combining the relevant textual elements.
    
    Rationale:
    According to Lecture 2, segmentation involves "Extracting the elements... 
    out of header, body, title". 
    Here, we fuse the 'CleanedTitle' and 'CleanedBody' into a single semantic 
    unit ('Final_Text') for the Feature Extraction stage, ensuring the model 
    sees the complete context.
    """
    # Fail Fast: Ensure required columns exist from the pre-processing stage
    required_cols = ['CleanedTitle', 'CleanedBody', 'Label']
    for col in required_cols:
        assert col in df.columns, f"Missing column '{col}'. Run pre_processing.py first."

    # Fill NaNs with empty strings to prevent concatenation errors
    df['CleanedTitle'] = df['CleanedTitle'].fillna("")
    df['CleanedBody'] = df['CleanedBody'].fillna("")

    # --- The Segmentation Logic ---
    # We isolate the "content" from the "structure". 
    # We treat Title + Body as one continuous text segment.
    df['Segmented_Content'] = df['CleanedTitle'] + " " + df['CleanedBody']
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    # Fail Fast: Check for empty segments
    # If both title and body were empty/scrubbed, we have nothing to learn from.
    df = df[df['Segmented_Content'].str.strip().astype(bool)]
    
    # Select ONLY the columns needed for the next step (Feature Extraction)
    # We strictly exclude metadata or intermediate columns to prevent Data Leakage.
    output_df = df[['Label', 'Segmented_Content']]
    
    return output_df

def main():
    input_path = Path(INPUT_FILE)

    print(f"Loading data from {input_path.name}...")
    df = pd.read_csv(input_path)
    
    # Apply Segmentation
    segmented_df = segment_data(df)
    
    # Save
    segmented_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print(f"\nSuccessfully created {OUTPUT_FILE} with {len(segmented_df)} rows.")
    print("Columns: 'Label', 'Segmented_Content'")
    print("Ready for Feature Extraction.")

if __name__ == "__main__":
    main()