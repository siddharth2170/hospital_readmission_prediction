import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.fillna("Unknown")
    df = pd.get_dummies(df, drop_first=True)
    y = df['readmitted']
    X = df.drop(columns=['readmitted'])
    smote_enn = SMOTEENN(random_state=42)
    X_bal, y_bal = smote_enn.fit_resample(X, y)
    return X_bal, y_bal

def split_data(X, y):
    return train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
