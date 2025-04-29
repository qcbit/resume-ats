from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file
import os

app = Flask(__name__)

# Load the model config and instantiate the model
model_name = "model/roberta"

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_config(config)

# Load adapter weights from safetensors
print("Loading adapter weights from safetensors...")
adapter_state_dict = load_file(f"{model_name}/adapter_model.safetensors")
model.load_state_dict(adapter_state_dict, strict=False)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

def get_preds(sentence1, sentence2, classes=["not equivalent", "equivalent"]):
    inputs = tokenizer(
        sentence1,
        sentence2,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    scores = torch.softmax(logits, dim=1).tolist()[0]
    return {
        "not_equivalent": scores[0],
        "equivalent": scores[1],
        "predicted_class": classes[int(scores[1] > scores[0])]
    }

def sliding_window_preds(sentence1, sentence2, window_size=512, stride=384, classes=["not equivalent", "equivalent"]):
    # Tokenize sentence1 (resume)
    tokens = tokenizer.encode(sentence1, add_special_tokens=False)
    results = []
    for start in range(0, len(tokens), stride):
        window_tokens = tokens[start:start+window_size]
        if not window_tokens:
            break
        window_text = tokenizer.decode(window_tokens)
        # Let tokenizer handle truncation and special tokens
        inputs = tokenizer(
            window_text,
            sentence2,
            truncation='only_first',
            padding="max_length",
            max_length=window_size,
            return_tensors="pt"
        ).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        scores = torch.softmax(logits, dim=1).tolist()[0]
        results.append({
            "not_equivalent": scores[0],
            "equivalent": scores[1]
        })
        if start + window_size >= len(tokens):
            break
    # Aggregate: average the "equivalent" and "not_equivalent" scores
    if results:
        avg_equiv = sum(r["equivalent"] for r in results) / len(results)
        avg_not_equiv = sum(r["not_equivalent"] for r in results) / len(results)
        return {
            "not_equivalent": avg_not_equiv,
            "equivalent": avg_equiv,
            "predicted_class": classes[int(avg_equiv > avg_not_equiv)]
        }
    else:
        return get_preds(sentence1[:1000], sentence2, classes)  # fallback

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence1 = data.get("sentence1")
    sentence2 = data.get("sentence2")
    if not sentence1 or not sentence2:
        return jsonify({"error": "Both 'sentence1' and 'sentence2' are required."}), 400
    result = sliding_window_preds(sentence1, sentence2)
    return jsonify(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Match Scorer Service")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on (default: 5000)")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
