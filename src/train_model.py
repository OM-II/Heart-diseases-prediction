import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from src.config import MODEL_PATH
from src.Load_data import load_data
from src.preprocessing import preprocess_data  
# -----------------------------
# Load processed data
# -----------------------------
X_train, X_test, y_train, y_test, feature_names = preprocess_data()

# -----------------------------
# Train Random Forest model
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# -----------------------------
# Feature importance
# -----------------------------
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(rf_model, MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")