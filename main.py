# --- Imports ---

import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- Robust CSV loader with optional kwargs (handles .zip) ---
def load_csv_safely(file_path, max_attempts=3, **read_csv_kwargs):
    """Safely load CSV (supports kwargs like compression='zip')."""
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1} to load {file_path}")
            df = pd.read_csv(file_path, **read_csv_kwargs)
            print(f"✅ Successfully loaded! Shape: {df.shape}")
            return df
        except PermissionError:
            print("❌ Permission denied. Waiting 2 seconds and trying again...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    print("❌ Failed to load file after all attempts")
    return None

# --- Helper: disease categorization (define BEFORE using) ---
def categorize_disease(topic: str) -> str:
    topic = str(topic).lower()
    if "cancer" in topic:
        return "Cancer"
    elif "cardio" in topic or "heart" in topic:
        return "Cardiovascular"
    elif "kidney" in topic:
        return "Chronic Kidney"
    elif "pulmonary" in topic or "respiratory" in topic:
        return "Chronic Pulmonary"
    else:
        return "Non-chronic"

# --- 1) Load data ---
file_path = r"C:\Users\ma_da\Downloads\U.S._Chronic_Disease_Indicators__CDI___2023_Release.csv.zip"
df = load_csv_safely(file_path, compression="zip")
if df is None:
    raise SystemExit("Could not load data. Close Excel/OneDrive locks and try again.")

# --- 2) Create target label ---
df["DiseaseCategory"] = df["Topic"].apply(categorize_disease)

# --- 3) Handle missing values ---
# Make DataValue numeric and impute
df["DataValue"] = pd.to_numeric(df["DataValue"], errors="coerce")
df["DataValue"].fillna(df["DataValue"].median(), inplace=True)

# Drop rows missing core categoricals
df.dropna(subset=["LocationDesc", "Question", "DataValueUnit", "Stratification1"], inplace=True)

# --- 4) Encode categoricals ---
label_encoders = {}
for col in ["LocationDesc", "Question", "DataValueUnit", "Stratification1"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("✅ Preprocessing complete")
print(df.head())
print(df.shape)
# --- Step 5: Split data into features (X) and target (y) ---
from sklearn.model_selection import train_test_split

# Select useful features
X = df[["YearStart", "LocationDesc", "Question", "DataValueUnit", "DataValue", "Stratification1"]]
y = df["DiseaseCategory"]

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data split complete")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
# --- Step 6: Train a Gradient Boosting Classifier ---
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n🚀 Training Gradient Boosting Classifier...")

model = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# --- Step 7: Evaluate on test data ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# --- Random Forest ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,      # let trees grow; you can cap (e.g., 20) if runtime is high
    min_samples_split=2,
    n_jobs=-1,           # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"✅ RF accuracy: {rf_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, rf_pred))
# --- Decision Tree ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n🌳 Training Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=20,        # tweak as you wish; None lets it grow fully (risk of overfitting)
    min_samples_split=5,
    random_state=42
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"✅ Decision Tree accuracy: {dt_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, dt_pred))
# --- Logistic Regression ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("\n🧮 Training Logistic Regression...")
lr = LogisticRegression(
    multi_class="multinomial",
    solver="saga",       # good for large datasets
    max_iter=1000,
    n_jobs=-1,           # available with saga
    random_state=42
)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"✅ Logistic Regression accuracy: {lr_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, lr_pred))
# --- AdaBoost ---
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n⚡ Training AdaBoost...")
ada = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)
print(f"✅ AdaBoost accuracy: {ada_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, ada_pred))
# --- Naive Bayes (Gaussian) ---
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

print("\n🧪 Training Naive Bayes (GaussianNB)...")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"✅ Naive Bayes accuracy: {nb_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, nb_pred))
# --- Extra Trees ---
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n🌲🌲 Training Extra Trees...")
et = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
et.fit(X_train, y_train)
et_pred = et.predict(X_test)
et_acc = accuracy_score(y_test, et_pred)
print(f"✅ Extra Trees accuracy: {et_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, et_pred))





