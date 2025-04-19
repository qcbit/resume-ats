from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the MiniLM model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define custom stop words
stop_words = {"the", "and", "is", "in", "to", "of", "a", "with", "for", "on", "at", "by"}

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stop words."""
    tokens = tokenizer.tokenize(text.lower())
    return " ".join([token for token in tokens if token.isalnum() and token not in stop_words])

def extract_keywords(text, top_n=20):
    """Extract keywords from text."""
    tokens = tokenizer.tokenize(text.lower())
    keywords = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ", ".join(keywords[:top_n])  # Limit to top N keywords

def get_embeddings_with_sliding_window(text, window_size=512, stride=256):
    """Generate embeddings for a given text using a sliding window approach."""
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+window_size] for i in range(0, len(tokens), stride)]
    embeddings = []

    for chunk in chunks:
        # Convert tokens to string and tokenize with padding/truncation
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=window_size,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of the last hidden state as the embedding for the chunk
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(chunk_embedding)

    # Aggregate embeddings (e.g., take the mean of all chunk embeddings)
    return np.mean(embeddings, axis=0)

def calculate_match_score(resume_embedding, job_description_embedding):
    """Calculate the match score using cosine similarity."""
    return cosine_similarity([resume_embedding], [job_description_embedding])[0][0]

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
df = df[['resume_text', 'job_description_text']]

# Rename columns for clarity
df.rename(columns={
    'resume_text': 'resume',
    'job_description_text': 'job_description',
}, inplace=True)

# Extract keywords dynamically from job descriptions
df['keywords'] = df['job_description'].apply(extract_keywords)

# Extract keywords dynamically
print("Extracting keywords from job descriptions...")
df['job_keywords'] = df['job_description'].apply(lambda x: set(extract_keywords(x).split(', '))) # Store as set for efficiency
print("Extracting keywords from resumes...")
df['resume_keywords'] = df['resume'].apply(lambda x: set(extract_keywords(x).split(', '))) # Store as set

# Remove empty strings resulting from split if necessary
df['job_keywords'] = df['job_keywords'].apply(lambda s: s - {''})
df['resume_keywords'] = df['resume_keywords'].apply(lambda s: s - {''})

# Generate embeddings for resumes and job descriptions using sliding window
df['resume_embedding'] = df['resume'].apply(lambda x: get_embeddings_with_sliding_window(x).tolist())
df['job_description_embedding'] = df['job_description'].apply(lambda x: get_embeddings_with_sliding_window(x).tolist())

# Calculate match scores
df['match_score'] = df.apply(
    lambda row: calculate_match_score(
        np.array(row['resume_embedding']),
        np.array(row['job_description_embedding'])
    ),
    axis=1
)

def calculate_keyword_overlap(resume_kws, job_kws):
    """Calculates Jaccard similarity between two sets of keywords."""
    intersection = len(resume_kws.intersection(job_kws))
    union = len(resume_kws.union(job_kws))
    return intersection / union if union > 0 else 0.0

# Calculate keyword overlap score
print("Calculating keyword overlap...")
df['keyword_overlap_score'] = df.apply(
    lambda row: calculate_keyword_overlap(row['resume_keywords'], row['job_keywords']),
    axis=1
)

# Clean text data (e.g., remove special characters, extra spaces)
df['resume'] = df['resume'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['job_description'] = df['job_description'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Drop the embedding columns (optional, if you don't need them in the final dataset)
df.drop(columns=['resume_embedding', 'job_description_embedding'], inplace=True)

# Save the cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)
print("Cleaned dataset saved to data/cleaned_dataset.csv")