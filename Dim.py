import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "selected_features.csv"
OUTPUT_FILE_2D = "pca_data_2d.csv"

def perform_pca_2d(df):
    """
    Performs Dimensionality Reduction using PCA (2 Components).
    Ref: Lecture 2, Slide 51 ("Allows to visualize the data in 2 or 3 dimensions")
    """
    # Fail Fast: Check for data
    if 'Label' not in df.columns:
        raise ValueError("Input data missing 'Label' column.")
    
    # Separate Features and Label
    feature_cols = [col for col in df.columns if col != 'Label']
    X = df[feature_cols]
    y = df['Label']

    # --- PCA Implementation ---
    # We choose 2 components for a 2D plot.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame with the new Principal Components
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    final_df = pd.concat([y, pca_df], axis=1)
    
    # Print Explained Variance Ratio
    # This tells us how much information we preserved in just 2D
    print(f"Explained Variance Ratio (PC1, PC2): {pca.explained_variance_ratio_}")
    print(f"Total Information Preserved: {sum(pca.explained_variance_ratio_):.2%}")
    
    return final_df

def visualize_2d(df):
    """
    Creates a 2D Scatter Plot of the Principal Components.
    Ref: Lecture 2, Slide 54 (Shows a 2D projection example)
    """
    plt.figure(figsize=(10, 8))
    
    # Use Seaborn for an easy and attractive 2D scatter plot with labels
    sns.scatterplot(
        data=df, 
        x='PC1', 
        y='PC2', 
        hue='Label', 
        palette='bright', # Vibrant colors
        s=100,            # Marker size
        alpha=0.7         # Transparency
    )
    
    plt.title('2D Visualization of Guardian Articles (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Category')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found. Please run feature_selection.py first.")
        return

    print(f"Loading data from {input_path.name}...")
    df = pd.read_csv(input_path)
    
    # 1. Perform PCA (2D)
    pca_df = perform_pca_2d(df)
    
    # 2. Visualize
    visualize_2d(pca_df)
    
    # 3. Save
    pca_df.to_csv(OUTPUT_FILE_2D, index=False)
    print(f"2D PCA data saved to {OUTPUT_FILE_2D}")

if __name__ == "__main__":
    main()