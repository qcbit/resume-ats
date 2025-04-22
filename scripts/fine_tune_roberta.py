import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load the dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Prepare the dataset for training
df["text"] = df["resume"] + " [SEP] " + df["job_description"]  # Combine resume and job description
df = df[["text", "match_score"]]  # Keep only the text and label columns
df["match_score"] = df["match_score"].astype(float)  # Ensure the label is a float

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["match_score"], test_size=0.2, random_state=42
)

# Load the tokenizer and model
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Regression task

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True
)

# Define a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("models/roberta")
tokenizer.save_pretrained("models/roberta")
print("Fine-tuned model saved to models/roberta")