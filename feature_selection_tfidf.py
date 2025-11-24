import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "represented_data.csv"
OUTPUT_FILE = "selected_features.csv"
K_BEST_FEATURES = 200

def quantitative_selection(df):
    """
    Quantitative Evaluation using Filter Method (Fisher Score).
    """
    print("\n--- Quantitative Evaluation (Filter Method) ---")
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    print(f"Original Feature Count: {X.shape[1]}")
    
    # Select Top 200 features based on ANOVA F-value (Fisher Score)
    selector = SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)
    selector.fit(X, y)
    
    cols_idxs = selector.get_support(indices=True)
    selected_features_names = X.columns[cols_idxs].tolist()
    
    X_selected = X[selected_features_names]
    final_df = pd.concat([y, X_selected], axis=1)
    
    # Report scores
    scores_df = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)
    
    print(f"\nTop 10 Features by Fisher Score:\n{scores_df.head(10)}")
    
    return final_df, scores_df

def qualitative_selection(df, best_feature_name, worst_feature_name):
    """
    Qualitative Evaluation using Histograms (PDF).
    ** ZOOMED to 0-0.4 as requested **
    """
    print("\n--- Qualitative Evaluation (Visual Strategy) ---")
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: The Best Feature
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=best_feature_name, hue='Label', kde=True, element="step", common_norm=False)
    plt.title(f"Best Feature: '{best_feature_name}' (High Discrimination)")
    plt.xlim(0, 0.4)  # <--- ZOOM HERE
    
    # Plot 2: The Worst Feature
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=worst_feature_name, hue='Label', kde=True, element="step", common_norm=False)
    plt.title(f"Worst Feature: '{worst_feature_name}' (High Overlap)")
    plt.xlim(0, 0.4)  # <--- ZOOM HERE
    
    plt.tight_layout()
    plt.show()

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found. Please run feature_representation.py first.")
        return

    print(f"Loading data from {input_path.name}...")
    df = pd.read_csv(input_path)
    
    # 1. Quantitative Step
    final_df, scores_df = quantitative_selection(df)
    
    # 2. Qualitative Step (Visualize with Zoom)
    best_feat = scores_df.iloc[0]['Feature']
    worst_feat = scores_df.iloc[-1]['Feature']
    
    # We use the original dataframe for visualization to compare
    qualitative_selection(df, best_feat, worst_feat)
    
    # 3. Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved {len(final_df.columns)-1} selected features to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()