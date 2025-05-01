from flask import Flask, request, jsonify
import os
import logging
import argparse
from openai import OpenAI
import json # Import json library for parsing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Default configuration
DEFAULT_PORT = 5000
# CHANGE the default model to one supporting JSON mode
DEFAULT_OPENAI_MODEL = "gpt-4o" # Or "gpt-4-turbo", "gpt-3.5-turbo-1106" etc.

# Get OpenAI API Key and Model from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# This will now default to "gpt-4o" if OPENAI_MODEL env var is not set
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

# Initialize OpenAI client - moved after the check
# client = OpenAI(api_key=OPENAI_API_KEY)

# --- Check for API Key ---
if not OPENAI_API_KEY:
    # Raise ValueError instead of just warning
    logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
    raise ValueError("OPENAI_API_KEY environment variable is required but was not found.")
else:
    # Initialize client only if key exists
    client = OpenAI(api_key=OPENAI_API_KEY)


def get_openai_match_score(resume_text, job_description):
    """
    Use OpenAI API to determine the match score and analysis for frontend display.
    """
    # --- Updated System Prompt ---
    system_prompt = """
    You are an expert ATS (Applicant Tracking System) analyzer. Your task is to evaluate how well a resume matches a job description.
    Provide a detailed analysis focusing on skills, experience, education, and overall alignment.

    Return ONLY a JSON object (no introductory text, no markdown formatting) with the following fields:
    - score: a float between 0 and 100 representing the overall match score.
    - categories: an object containing scores (0-100) for the following categories:
        - skills: { "score": float }
        - experience: { "score": float }
        - education: { "score": float }
        - achievements: { "score": float }
    - feedback: a brief explanation of the score and suggestions for improvement (1-3 sentences).
    """
    user_prompt = f"""
    Analyze the match between the following resume and job description:

    RESUME:
    ---
    {resume_text}
    ---

    JOB DESCRIPTION:
    ---
    {job_description}
    ---

    Provide your analysis as a JSON object according to the system instructions. Ensure all scores are between 0 and 100.
    """

    try:
        logger.info(f"Sending request to OpenAI API with model {OPENAI_MODEL}")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        logger.info("Received response from OpenAI API")

        content = response.choices[0].message.content
        logger.debug(f"Raw OpenAI response content: {content}")

        # Attempt to parse the JSON string from the content
        try:
            # Use json.loads for more robust parsing
            result_json = json.loads(content)
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse JSON response from OpenAI: {json_err}")
            logger.error(f"Invalid JSON string: {content}")
            raise ValueError(f"OpenAI returned invalid JSON: {json_err}")


        # --- Updated Validation ---
        required_keys = ["score", "categories", "feedback"]
        if not all(k in result_json for k in required_keys):
             logger.error(f"OpenAI response missing required keys: {result_json}")
             raise ValueError("OpenAI response missing required keys.")

        required_categories = ["skills", "experience", "education", "achievements"]
        if not isinstance(result_json.get("categories"), dict) or \
           not all(cat in result_json["categories"] for cat in required_categories) or \
           not all(isinstance(result_json["categories"][cat], dict) and "score" in result_json["categories"][cat] for cat in required_categories):
             logger.error(f"OpenAI response 'categories' format is incorrect: {result_json.get('categories')}")
             raise ValueError("OpenAI response 'categories' format is incorrect.")

        # Add default relevance if needed by frontend (or adjust frontend later)
        for cat in required_categories:
             if "relevance" not in result_json["categories"][cat]:
                  result_json["categories"][cat]["relevance"] = 25 # Default relevance

        logger.info(f"Processed result: {result_json}")
        return result_json

    except Exception as e:
        logger.error(f"Error calling OpenAI API or processing response: {e}")
        return {"error": f"Failed to get analysis from OpenAI: {e}"}, 500

# --- /predict route remains the same ---
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    resume_text = data.get("resume_text")
    job_description = data.get("job_description")

    if not resume_text or not job_description:
        return jsonify({
            "error": "Both 'resume_text' and 'job_description' are required."
        }), 400

    result = get_openai_match_score(resume_text, job_description)

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int) and result[1] >= 500:
        return jsonify(result[0]), result[1]

    return jsonify(result)

# --- Main execution block ---
if __name__ == "__main__":
    # --- Check for API Key at startup ---
    if not OPENAI_API_KEY:
         # Log and exit cleanly if running as main script and key is missing
         logger.error("FATAL: OPENAI_API_KEY environment variable not set. Exiting.")
         exit(1) # Exit if API key is missing when running directly

    parser = argparse.ArgumentParser(description="Match Scorer Service using OpenAI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to run the Flask app on (default: {DEFAULT_PORT})")
    parser.add_argument("--openai-model", type=str, help="OpenAI model to use (e.g., gpt-4)")

    args = parser.parse_args()

    if args.openai_model:
        OPENAI_MODEL = args.openai_model

    logger.info(f"Starting Match Scorer service (OpenAI) on port {args.port}")
    logger.info(f"Using OpenAI model: {OPENAI_MODEL}")

    # Client initialization is now guaranteed to happen before app.run if key exists

    app.run(host="0.0.0.0", port=args.port)