import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load data
train_data =pd.read_csv("C:/Users/K.Vijay/Downloads/customer-churn-ng-intern/train.csv")

test_data = pd.read_csv("C:/Users/K.Vijay/Downloads/customer-churn-ng-intern/test.csv")

sample_submission = pd.read_csv("C:/Users/K.Vijay/Downloads/customer-churn-ng-intern/sample_submission.csv")

# Split features and target
X = train_data.drop(columns=["CustomerId", "Surname", "Exited"])
y = train_data["Exited"]
X_test = test_data.drop(columns=["CustomerId", "Surname"])

# Encode categorical variables
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model (note: eval_metric here, not in .fit)
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

# Fit the model without early stopping
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate on validation set
y_val_pred = model.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, y_val_pred))

# Predict on test set
test_preds = model.predict_proba(X_test_scaled)[:, 1]

# Format predictions to 2 decimal places
sample_submission["Exited"] = test_preds.round(2)

# Save submission
import os
output_path = os.path.abspath("my_churn_submission.csv")
sample_submission.to_csv(output_path, index=False)
print("Saved at:", output_path)
