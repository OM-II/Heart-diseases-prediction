import pandas as pd
from src.config import TRAIN_DATA_PATH
from src.Load_data import load_data
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# -----------------------------
# Load data
# -----------------------------
df = load_data(TRAIN_DATA_PATH)

# -----------------------------
# Check null values
# -----------------------------
print("Null values per column:")
print(df.isnull().sum())

# -----------------------------
# Check mixed-type issues (exclude target)
# -----------------------------
for col in df.drop(columns=["Heart Disease"]).columns:
    if df[col].dtype == "object":
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = converted.isna() & df[col].notna()
        if invalid.any():
            print(f"\nMixed-type values in column: {col}")
            print(df.loc[invalid, col])

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_data(df):

    df = df.drop(columns=["id"])

    df["Heart Disease"] = df["Heart Disease"].map({
        "Presence": 1,
        "Absence": 0
    })

    numerical_features = [
        "Age", "BP", "Cholesterol",
        "Max HR", "ST depression",
        "Number of vessels fluro"
    ]

    binary_features = [
        "Sex", "FBS over 120", "Exercise angina"
    ]

    nominal_features = [
        "Chest pain type", "EKG results", "Thallium"
    ]

    ordinal_features = ["Slope of ST"]
    ordinal_categories = [[1, 2, 3]]

    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("bin", "passthrough", binary_features),
            ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_features),
            ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        numerical_features,
        binary_features,
        nominal_features,
        ordinal_features,
        X_train.index,
        X_test.index
    )

# -----------------------------
# Call preprocessing
# -----------------------------
(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    preprocessor,
    numerical_features,
    binary_features,
    nominal_features,
    ordinal_features,
    train_index,
    test_index
) = preprocess_data(df)

# -----------------------------
# Convert processed data to DataFrame
# -----------------------------
nominal_feature_names = (
    preprocessor
    .named_transformers_["nom"]
    .get_feature_names_out(nominal_features)
)

all_feature_names = (
    numerical_features
    + binary_features
    + list(nominal_feature_names)
    + ordinal_features
)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()

X_train_df = pd.DataFrame(
    X_train_processed,
    columns=all_feature_names,
    index=train_index
)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=all_feature_names,
    index=test_index
)

print(X_train_df.head())