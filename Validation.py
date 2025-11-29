import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path

#  Constants & Configuration 
INPUT_FILE = "selected_features.csv" 
TEST_SIZE_RATIO = 0.20  # 20% validation set 
SEED = 42               # For reproducibility

def perform_validation(df):
    """
    Implements Holdout Validation, trains a Logistic Regression model, and evaluates performance.
    """
    print(f"Starting Holdout Validation (Train/Test Split: {1 - TEST_SIZE_RATIO}/{TEST_SIZE_RATIO})...")
    
    # 1. Prepare Data
    # Encode the categorical 'Label' into numerical format for the model
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    X = df.drop(columns=['Label', 'Label_Encoded']).values
    y = df['Label_Encoded'].values
    
    # 2. Split Data (Holdout Validation)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, 
        test_size=TEST_SIZE_RATIO, 
        random_state=SEED,
        stratify=y # Ensures equal distribution of classes in train/validation sets
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_validation)} samples")
    
    # 3. Model Training
    # Logistic Regression Model
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)
    
    # 4. Model Evaluation (Prediction)
    y_pred = model.predict(X_validation)
    
    # 5. Measure Performance (Metric)
    accuracy = accuracy_score(y_validation, y_pred)
    
    print("\n Validation Results ")
    print(f"Chosen Metric: Accuracy")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Additional report for better insights
    print("\nDetailed Classification Report:")
    print(classification_report(y_validation, y_pred, target_names=le.classes_))
    
    return accuracy

def main():
    # Ensure reproducibility
    np.random.seed(SEED)
    
    input_path = Path(INPUT_FILE)

    df = pd.read_csv(input_path)
    
    # Check if the dataframe is empty after all filtering steps
    if df.empty:
        print("Error: Dataset is empty after feature selection.")
        return

    perform_validation(df)

if __name__ == "__main__":
    main()
