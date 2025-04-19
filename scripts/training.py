import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import ast

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")  # Assuming it now has embeddings and keyword_overlap_score

# Convert string representations of embeddings back to lists/arrays if needed
df['resume_embedding'] = df['resume_embedding'].apply(ast.literal_eval)
df['job_description_embedding'] = df['job_description_embedding'].apply(ast.literal_eval)

# Split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare features including keyword overlap
X_train_embeddings = np.array([np.concatenate([row['resume_embedding'], row['job_description_embedding']]) for _, row in train_df.iterrows()])
X_train_keywords = train_df['keyword_overlap_score'].values.reshape(-1, 1)  # Reshape for hstack
X_train = np.hstack((X_train_embeddings, X_train_keywords))

X_test_embeddings = np.array([np.concatenate([row['resume_embedding'], row['job_description_embedding']]) for _, row in test_df.iterrows()])
X_test_keywords = test_df['keyword_overlap_score'].values.reshape(-1, 1)
X_test = np.hstack((X_test_embeddings, X_test_keywords))

y_train = train_df['match_score'].values  # Target variable
y_test = test_df['match_score'].values

# Train a Random Forest Regressor
print("Training the Random Forest Regressor...")
model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the test results
test_df['predicted_match_score'] = y_pred
test_df[['resume', 'job_description', 'predicted_match_score', 'keyword_overlap_score']].to_csv(
    "data/test_results.csv", index=False
)
print("Test results saved to data/test_results.csv")

# Save the trained model
print("Saving the trained model...")
joblib.dump(model, "models/resume_job_match_model.pkl")
print("Model saved to models/resume_job_match_model.pkl")