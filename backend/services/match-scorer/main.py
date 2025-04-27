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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence1 = data.get("sentence1")
    sentence2 = data.get("sentence2")
    if not sentence1 or not sentence2:
        return jsonify({"error": "Both 'sentence1' and 'sentence2' are required."}), 400
    result = get_preds(sentence1, sentence2)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
