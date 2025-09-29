import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import load_data, preprocess_data, split_data

df = load_data("data/diabetes_data.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(),
    "ExtraTrees": ExtraTreesClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print(f"=== {name} ===")
    print(classification_report(y_test, preds))
    print(f"AUC: {auc:.4f}\n")
