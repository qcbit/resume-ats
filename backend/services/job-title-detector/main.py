import argparse
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import process
from loguru import logger
import json

# Initialize Flask app
app = Flask(__name__)

# Configure Loguru
logger.add("job_title_detector.log", rotation="10 MB", level="INFO", format="{time} - {level} - {message}")

# Log service startup
logger.info("Starting Job Title Detector Service...")

# Load job titles from job_title_industry.json
try:
    with open("../data/job_title_industry.json", "r") as f:
        job_title_to_industry = json.load(f)
    job_titles = list(job_title_to_industry.keys())
    logger.info("Loaded job titles successfully.")
except Exception as e:
    logger.error(f"Failed to load job titles: {e}")
    raise

# Load a pre-trained Sentence-BERT model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded Sentence-BERT model successfully.")
except Exception as e:
    logger.error(f"Failed to load Sentence-BERT model: {e}")
    raise

def detect_job_title(job_description, job_titles, threshold=0.75):
    """
    Detect a job title from the job description using a combination of embedding similarity and fuzzy matching.
    :param job_description: The job description text.
    :param job_titles: A list of known job titles.
    :param threshold: Similarity threshold for embedding similarity.
    :return: Detected job title or None.
    """
    try:
        # Step 1: Compute embedding similarity
        job_description_embedding = embedding_model.encode(job_description, convert_to_tensor=True)
        job_title_embeddings = embedding_model.encode(job_titles, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(job_description_embedding, job_title_embeddings)
        
        # Find the most similar job title based on embeddings
        best_match_idx = similarities.argmax().item()
        best_match_score = similarities[0][best_match_idx].item()
        best_match_title = job_titles[best_match_idx]

        # Step 2: Apply fuzzy matching as a fallback
        if best_match_score < threshold:
            # Use fuzzy matching if embedding similarity is below the threshold
            best_match_title, fuzzy_score = process.extractOne(job_description, job_titles)
            logger.info(f"Fuzzy match applied. Fuzzy score: {fuzzy_score}")

        return best_match_title
    except Exception as e:
        logger.error(f"Error in detect_job_title: {e}")
        return None

# Define the API endpoint
@app.route("/detect-job-title", methods=["POST"])
def detect_job_title_endpoint():
    """
    API endpoint to detect a job title from a job description.
    """
    try:
        data = request.get_json()
        if "job_description" not in data:
            logger.warning("Missing 'job_description' in request.")
            return jsonify({"error": "Missing 'job_description' in request"}), 400

        job_description = data["job_description"]
        detected_title = detect_job_title(job_description, job_titles)
        logger.info(f"Job title detected: {detected_title}")
        return jsonify({"job_title": detected_title})
    except Exception as e:
        logger.error(f"Error in API endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Job Title Detector Service")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on (default: 5000)")
    args = parser.parse_args()

    try:
        logger.info(f"Job Title Detector Service is running on http://0.0.0.0:{args.port}")
        app.run(host="0.0.0.0", port=args.port)
    except KeyboardInterrupt:
        logger.info("Job Title Detector Service is shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")