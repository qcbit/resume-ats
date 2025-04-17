from datasets import load_dataset
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def extract_keywords(text):
    words = word_tokenize(text)
    keywords = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    return ", ".join(keywords[:5])  # Limit to top 5 keywords

# Load the dataset
dataset = load_dataset("cnamuangtoun/resume-job-description-fit")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Save the raw dataset to a CSV file
df.to_csv("data/raw_dataset.csv", index=False)
print("Raw dataset saved to data/raw_dataset.csv")

# Load the raw dataset
df = pd.read_csv("data/raw_dataset.csv")

# Drop unnecessary columns (if any)
df = df[['resume_text', 'job_description_text', 'label']]

# Rename columns for clarity
df.rename(columns={
    'resume_text': 'resume',
    'job_description_text': 'job_description',
    'label': 'match_score'
}, inplace=True)


df['keywords'] = df['job_description'].apply(extract_keywords)

# Clean text data (e.g., remove special characters, extra spaces)
df['resume'] = df['resume'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['job_description'] = df['job_description'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Save the cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)
print("Cleaned dataset saved to data/cleaned_dataset.csv")