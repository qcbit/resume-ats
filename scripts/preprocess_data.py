from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Load the MiniLM model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define custom stop words
stop_words = {"the", "and", "is", "in", "to", "of", "a", "with", "for", "on", "at", "by"}

activation_words = [
    "responsibilities", "key responsibilities", "responsible", "skills", "skilled",
    "seek", "seeking", "looking for", "role", "must haves", "preferred", "duties",
    "required", "day", "experience", "qualifications", "requirements",
]

# POS tags to exclude
EXCLUDE_POS = {
    'PRP', 'PRP$', 'WP', 'WP$',  # Pronouns
    'JJ', 'JJR', 'JJS',          # Adjectives
    'DT',                        # Determiners (includes articles)
    'RB', 'RBR', 'RBS', 'WRB',   # Adverbs
    'IN',                        # Prepositions, subordinating conjunctions
    'CC',                        # Coordinating conjunctions
    'UH',                        # Interjections
    'WDT',                       # Wh-determiner
    'PDT',                       # Predeterminer
    'POS',                       # Possessive ending
    'CD',                        # Cardinal number (quantifiers)
    'EX',                        # Existential there
    'FW',                        # Foreign word
    'LS',                        # List item marker
    'MD',                        # Modal
    'SYM',                       # Symbol
    'TO',                        # to
}

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stop words."""
    tokens = tokenizer.tokenize(text.lower())
    return " ".join([token for token in tokens if token.isalnum() and token not in stop_words])

def extract_keywords(text, top_n=20):
    """Extract keywords from text using activation words or fallback to top_n unique words, excluding unwanted POS."""
    if not isinstance(text, str):
        return set()
    text_lower = text.lower()
    pattern = r'(' + '|'.join(re.escape(word) for word in activation_words) + r')'
    match = re.search(pattern, text_lower)
    keywords = set()
    activation_words_set = set(activation_words)
    if match:
        # Activation word found: extract from relevant section
        start_idx = match.end()
        relevant_text = text_lower[start_idx:]
        sentences = re.split(r'[.\n]', relevant_text)
        for sentence in sentences:
            # Stop if we hit another activation word
            if re.search(pattern, sentence):
                break
            # Tokenize and POS tag
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            for word, pos in tagged:
                # Skip activation words and excluded POS
                if word in activation_words_set or pos in EXCLUDE_POS:
                    continue
                if word.isalpha():
                    keywords.add(word)
            if len(keywords) >= top_n:
                break
        return set(list(keywords)[:top_n])
    else:
        # No activation word: fallback to top_n unique words from the whole text
        tokens = nltk.word_tokenize(text_lower)
        tagged = nltk.pos_tag(tokens)
        unique_words = []
        seen = set()
        for word, pos in tagged:
            if word in activation_words_set or pos in EXCLUDE_POS:
                continue
            if word.isalpha() and word not in seen:
                unique_words.append(word)
                seen.add(word)
            if len(unique_words) >= top_n:
                break
        return set(unique_words)

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

# Extract keywords dynamically from job descriptions
print("Extracting keywords from job descriptions (activation word-based)...")
df['job_keywords'] = df['job_description'].apply(lambda x: extract_keywords(x))
print("Extracting keywords from resumes...")
df['resume_keywords'] = df['resume'].apply(lambda x: extract_keywords(x))

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

# Save the cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)
print("Cleaned dataset saved to data/cleaned_dataset.csv")