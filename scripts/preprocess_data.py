from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# Load the MiniLM model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize KeyBERT
kw_model = KeyBERT(model_name)

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stop words."""
    tokens = tokenizer.tokenize(text.lower())
    return " ".join([token for token in tokens if token.isalnum()])

# Increase top_n to extract more keywords
def extract_keywords(text, top_n=15): 
    """Extract keywords from text using KeyBERT."""
    # Ensure text is a string and not empty/NaN
    if not isinstance(text, str) or text.strip() == "":
        return ""
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3), # Consider adjusting this range
            stop_words="english",
            top_n=top_n
        )
        # Filter out low-relevance keywords if needed (e.g., based on score)
        # keywords = [kw for kw in keywords if kw[1] > 0.3] # Example score threshold
        return ", ".join([kw[0] for kw in keywords])
    except Exception as e:
        print(f"Error extracting keywords for text: {text[:100]}... Error: {e}")
        return ""

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

def calculate_keyword_overlap(resume_kws, job_kws):
    """Calculates Jaccard similarity between two sets of keywords."""
    intersection = len(resume_kws.intersection(job_kws))
    union = len(resume_kws.union(job_kws))
    return intersection / union if union > 0 else 0.0

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

# Extract keywords dynamically
print("Extracting keywords from job descriptions...")
df['job_keywords'] = df['job_description'].apply(lambda x: set(extract_keywords(x, top_n=10).split(', '))) # Store as set for efficiency
print("Extracting keywords from resumes...")
df['resume_keywords'] = df['resume'].apply(lambda x: set(extract_keywords(x, top_n=10).split(', '))) # Store as set

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