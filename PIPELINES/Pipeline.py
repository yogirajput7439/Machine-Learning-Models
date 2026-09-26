import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load data
df = pd.read_csv("credit_risk_dataset.csv")


# 2. X and y
X = df.drop("loan_status", axis=1)
y = df["loan_status"]


# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Columns
numerical_cols = [
    # your numerical columns
]

categorical_cols = [
    # your categorical columns
]


# 5. Numerical Pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 6. Categorical Pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 7. Column Transformer
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# 8. Model
model = RandomForestClassifier(
    random_state=42
)


# 9. Final Pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# 10. Training
final_pipeline.fit(X_train, y_train)


# 11. Prediction
y_pred = final_pipeline.predict(X_test)


# 12. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# 13. Save complete pipeline
joblib.dump(final_pipeline, "model.pkl")

print("Model saved successfully!")
