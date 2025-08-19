import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib


param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Load the dataset
print("Reading dataset...")
df = pd.read_csv("Housing.csv")

# --- CLEANING ---
print("Cleaning dataset...")
# Handle missing values (drop for simplicity)
df.dropna(inplace=True)

# Convert "yes"/"no" columns to booleans
bool_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[bool_cols] = df[bool_cols].applymap(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Check and clean furnishingstatus (categorical)
df['furnishingstatus'] = df['furnishingstatus'].str.strip().str.lower()

# Target variable
y = df['price']

# ----------- MODEL 1: All Features Except Price -----------
print("Preparing Model 1 (all features)...")

X1 = df.drop(columns=['price'])

# Columns to one-hot encode
categorical_cols = ['furnishingstatus']

# Preprocessing: encode categorical columns
preprocessor1 = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# Build pipeline
model1 = Pipeline(steps=[
    ('preprocessor', preprocessor1),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.2, random_state=42)

# Fit model
print("Training Model 1...")
base_model1 = RandomForestRegressor(random_state=42)
search1 = RandomizedSearchCV(
    estimator=base_model1,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

pipeline1 = Pipeline(steps=[
    ('preprocessor', preprocessor1),
    ('regressor', search1)
])

print("Auto-tuning Model 1...")
pipeline1.fit(X1_train, y1_train)
best_model1 = pipeline1

# Evaluate
y1_pred = best_model1.predict(X1_test)
print(f"Model 1 - MSE: {mean_squared_error(y1_test, y1_pred):.2f}")
print(f"Model 1 - R2: {r2_score(y1_test, y1_pred):.2f}")

# Save model
joblib.dump(best_model1, "model_all_features.pkl")

# ----------- MODEL 2: area, bedrooms, bathrooms, stories -----------
print("\nPreparing Model 2 (selected features)...")

selected_features = ['area', 'bedrooms', 'bathrooms', 'stories']
X2 = df[selected_features]

# Train/test split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)

# Train model (no encoding needed)
model2 = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training Model 2...")
base_model2 = RandomForestRegressor(random_state=42)
search2 = RandomizedSearchCV(
    estimator=base_model2,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Auto-tuning Model 2...")
search2.fit(X2_train, y2_train)
best_model2 = search2.best_estimator_

# Evaluate
y2_pred = best_model2.predict(X2_test)
print(f"Model 2 - MSE: {mean_squared_error(y2_test, y2_pred):.2f}")
print(f"Model 2 - R2: {r2_score(y2_test, y2_pred):.2f}")

# Save model
joblib.dump(best_model2, "model_selected_features.pkl")

print("Both models trained and saved successfully!")
