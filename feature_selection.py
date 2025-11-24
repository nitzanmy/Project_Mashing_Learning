import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "represented_data.csv"
OUTPUT_FILE = "selected_features.csv"
# We will drop the feature with the lowest score to demonstrate selection
NUM_FEATURES_TO_DROP = 1 

def quantitative_selection(df):
    """
    Quantitative Evaluation using Filter Method (Fisher Score / ANOVA).
    Ref: Lecture 2, Slide 41 (Fisher Score formula) & Slide 40 (Filters)
    
    Calculates a score for each feature based on the ratio of 
    between-class variance to within-class variance.
    """
    print("\n--- Quantitative Evaluation (Filter Method) ---")
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # f_classif calculates the ANOVA F-value, which is mathematically 
    # equivalent to the Fisher Score concept (Between Var / Within Var)
    # mentioned in Slide 41[cite: 707].
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    # Create a DataFrame to view scores
    scores_df = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)
    
    print(scores_df)
    
    # Identify features to drop (the ones with the lowest scores)
    features_to_drop = scores_df.tail(NUM_FEATURES_TO_DROP)['Feature'].tolist()
    print(f"\n>> Decision: Dropping {features_to_drop} due to low discriminative power.")
    
    return scores_df, features_to_drop

def qualitative_selection(df, best_feature, worst_feature):
    """
    Qualitative Evaluation using Visualizations (Histograms/Density Plots).
    Ref: Lecture 2, Slide 47 (Probability Density Function)
    
    Visualizes why the best feature obtained a high score (good separation)
    vs why the worst feature got a low score (high overlap).
    """
    print("\n--- Qualitative Evaluation (Visual Strategy) ---")
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: The Best Feature (High Discrimination)
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=best_feature, hue='Label', kde=True, element="step")
    plt.title(f"Best Feature: '{best_feature}' (Good Separation)")
    plt.xlim(0, 0.5)  # Limit x-axis to 0.5
    
    # Plot 2: The Worst Feature (Low Discrimination)
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=worst_feature, hue='Label', kde=True, element="step")
    plt.title(f"Worst Feature: '{worst_feature}' (High Overlap)")
    plt.xlim(0, 0.5)  # Limit x-axis to 0.5
    
    plt.tight_layout()
    plt.show()

def main():
    input_path = Path(INPUT_FILE)

    df = pd.read_csv(input_path)
    
    # 1. Quantitative Step
    scores_df, features_to_drop = quantitative_selection(df)
    
    # 2. Qualitative Step (Visualize the contrast)
    best_feature = scores_df.iloc[0]['Feature']
    worst_feature = scores_df.iloc[-1]['Feature']
    qualitative_selection(df, best_feature, worst_feature)
    
    # 3. Drop the selected features
    final_df = df.drop(columns=features_to_drop)
    
    # 4. Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved dataset to {OUTPUT_FILE}")
    print(f"Original Feature Count: {len(df.columns) - 1}")
    print(f"Final Feature Count: {len(final_df.columns) - 1}")

if __name__ == "__main__":
    main()