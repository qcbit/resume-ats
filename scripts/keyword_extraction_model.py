from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support

def parse_conll_file(conll_file_path):
    """
    Parse a CoNLL-like file into tokens and labels.
    :param conll_file_path: Path to the CoNLL-like dataset file.
    :return: A dictionary with 'tokens' and 'labels'.
    """
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []

    with open(conll_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line indicates end of a sentence
                if current_tokens and current_labels:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                current_tokens = []
                current_labels = []
            else:
                # Split the line and handle unexpected formatting
                parts = line.split()
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line}")
                    continue
                token, label = parts
                current_tokens.append(token)
                current_labels.append(int(label))  # Convert label to integer

    # Add the last sentence if the file doesn't end with a blank line
    if current_tokens and current_labels:
        tokens.append(current_tokens)
        labels.append(current_labels)

    return {"tokens": tokens, "labels": labels}

# Path to the labeled dataset
conll_file_path = "data/labeled_dataset.conll"

# Parse the CoNLL-like file
parsed_data = parse_conll_file(conll_file_path)

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(parsed_data)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Save the dataset for later use
tokenized_dataset.save_to_disk("data/huggingface_keyword_dataset")

print("Dataset successfully converted and saved to 'data/huggingface_keyword_dataset'.")

# Load pre-trained BERT model for token classification
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define metrics
def compute_metrics(pred):
    """
    Compute precision, recall, and F1 score for the token classification task.
    """
    # Flatten predictions and labels
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()

    # Filter out ignored labels (-100)
    valid_indices = labels != -100
    labels = labels[valid_indices]
    preds = preds[valid_indices]

    # Compute metrics for the multiclass problem
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"precision": precision, "recall": recall, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/bert-keyword-extraction",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./models/bert-keyword-extraction")
tokenizer.save_pretrained("./models/bert-keyword-extraction")