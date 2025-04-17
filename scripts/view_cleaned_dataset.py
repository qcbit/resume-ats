import pandas as pd

# Load the cleaned dataset
file_path = "data/cleaned_dataset.csv"
try:
    df = pd.read_csv(file_path)
    print("Cleaned Dataset Preview:")
    print(df.head())  # Display the first 5 rows of the dataset
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the preprocessing script has been run.")