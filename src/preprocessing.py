import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from data_loader import load_pima_data

def preprocess_data(file_path, target_column, test_size=0.2, random_state=42):
    data = load_pima_data(file_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

