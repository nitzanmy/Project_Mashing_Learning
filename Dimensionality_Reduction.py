import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "selected_features.csv"
OUTPUT_FILE = "pca_data.csv"

def perform_pca(df):
    """
    Performs Dimensionality Reduction using PCA.
    Ref: Lecture 2, Slide 51 ("Allows to visualize the data in 2 or 3 dimensions")
    """
    # Fail Fast: Check for data
    if 'Label' not in df.columns:
        raise ValueError("Input data missing 'Label' column.")
    
    # Separate Features and Label
    # PCA works on the numeric features only
    feature_cols = [col for col in df.columns if col != 'Label']
    X = df[feature_cols]
    y = df['Label']

    # Validate we have enough features to reduce
    if X.shape[1] < 3:
        print("Warning: Dataset has fewer than 3 features. PCA for 3D viz is trivial.")
    
    # --- PCA Implementation ---
    # Ref: Lecture 2, Slide 53 ("Project the samples onto the first PCs")
    # We choose 3 components to create a 3D visualization.
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame with the new Principal Components
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    final_df = pd.concat([y, pca_df], axis=1)
    
    # Print Explained Variance Ratio
    # This tells us how much information (variance) we preserved.
    # Ref: Slide 53 (PCA preserves "important information and variations")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Information Preserved: {sum(pca.explained_variance_ratio_):.2%}")
    
    return final_df

def visualize_3d(df):
    """
    Creates a 3D Scatter Plot of the Principal Components.
    Ref: Lecture 2, Slide 51 ("visualize the data in 2 or 3 dimensions")
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Map labels to colors for clear distinction
    # We assume standard labels; adding fallback for robustness
    unique_labels = df['Label'].unique()
    colors = ['r', 'g', 'b', 'orange', 'purple', 'cyan']
    
    for i, label in enumerate(unique_labels):
        subset = df[df['Label'] == label]
        # Use modulo to cycle through colors if there are more labels than colors
        color = colors[i % len(colors)]
        ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], 
                   label=label, c=color, s=50, alpha=0.6)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Visualization of Guardian Articles (PCA)')
    ax.legend()
    
    plt.show()

def main():
    input_path = Path(INPUT_FILE)

    print(f"Loading data from {input_path.name}...")
    df = pd.read_csv(input_path)
    
    # 1. Perform PCA
    pca_df = perform_pca(df)
    
    # 2. Visualize
    visualize_3d(pca_df)
    
    # 3. Save (Optional, if we wanted to use PCA features for training)
    # Note: For this assignment, we mostly use this for visualization as requested.
    pca_df.to_csv(OUTPUT_FILE, index=False)
    print(f"PCA data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()