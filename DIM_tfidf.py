import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import numpy as np
import warnings
# Suppress matplotlib font warning
warnings.filterwarnings("ignore", category=UserWarning) 

# --- Configuration ---
INPUT_FILE = "selected_features.csv"
OUTPUT_FILE = "pca_data.csv"
N_COMPONENTS = 3  # Required for the 3D visualization (Lecture 2, Slide 51)

def perform_pca(df):
    """
    Performs Dimensionality Reduction using PCA (Principal Component Analysis).
    Ref: Lecture 2, Slide 51 (PCA is a widely used method)
    """
    # 1. Prepare Data
    feature_cols = [col for col in df.columns if col != 'Label']
    X = df[feature_cols].values
    y = df['Label']

    # 2. Encode Labels (Needed for visualization coloring and grouping)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. PCA Execution
    # Ref: Lecture 2, Slide 51 (Creates new features that incorporate several others)
    pca = PCA(n_components=N_COMPONENTS)
    X_pca = pca.fit_transform(X)
    
    # 4. Reporting Variance
    # This explains how much information is preserved by the projection.
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Information Preserved in {N_COMPONENTS} PCs: {sum(pca.explained_variance_ratio_):.2%}")
    
    # 5. Create final DataFrame
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(N_COMPONENTS)])
    pca_df['Label'] = y.values
    
    return pca_df, y_encoded, le.classes_

def visualize_3d(df, y_encoded, classes):
    """
    Creates a 3D Scatter Plot of the projected samples.
    Ref: Lecture 2, Slide 51 ("Allows to visualize the data in 2 or 3 dimensions")
    """
    print("\n--- Generating 3D Visualization ---")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a distinct color map based on the number of classes
    cmap = plt.cm.get_cmap('Spectral', len(classes))
    
    for i, target_class in enumerate(classes):
        indices = y_encoded == i
        # Use a list for 'c' to ensure consistency with matplotlob versions
        ax.scatter(df.loc[indices, 'PC1'], 
                   df.loc[indices, 'PC2'], 
                   df.loc[indices, 'PC3'], 
                   label=target_class, 
                   c=[cmap(i)], 
                   s=50, alpha=0.7)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Visualization of Articles (PCA)')
    ax.legend(title='Category')
    
    plt.show()

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found. Please run feature_selection.py first.")
        return

    df = pd.read_csv(input_path)
    
    # 1. Perform PCA (Dimensionality Reduction)
    # Ref: Lecture 2, Slide 809, 810
    pca_df, y_encoded, classes = perform_pca(df)
    
    # 2. Visualize the projection (The core requirement)
    visualize_3d(pca_df, y_encoded, classes)
    
    # 3. Save the result
    pca_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved PCA data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()