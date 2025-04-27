from flask import Flask, request, jsonify
from fuzzywuzzy import process
import json
from loguru import logger
import openai
import os
import ast

# Initialize Flask app
app = Flask(__name__)

# Configure Loguru
logger.add("keyword_extractor.log", rotation="10 MB", level="INFO", format="{time} - {level} - {message}")

# Set your OpenAI API key (set this as an environment variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY")
logger.info("OpenAI API key has been successfully set.")

# Load skills and job titles
try:
    with open("../data/industry_skills.json", "r") as f:
        industry_skills = json.load(f)
    with open("../data/job_title_industry.json", "r") as j:
        job_titles = json.load(j)
    logger.info("Loaded industry skills and job titles successfully.")
except Exception as e:
    logger.error(f"Failed to load skills or job titles: {e}")
    raise

def extract_keywords_with_gpt4(job_description):
    prompt = (
        "Extract the most relevant keywords (skills, technologies, tools, and concepts) from the following job description. "
        "Return them as a Python list of strings.\n\n"
        f"Job Description:\n{job_description}\n\nKeywords:"
    )
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for extracting keywords from job descriptions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=128,
        temperature=0.2,
    )
    # Parse the response to get the list of keywords
    content = response.choices[0].message.content
    try:
        # Use ast.literal_eval for safety instead of eval
        keywords = ast.literal_eval(content) if content.strip().startswith("[") else []
    except Exception:
        keywords = []
    return keywords

def match_skills_with_keywords(extracted_keywords, skills, threshold=80):
    matched_skills = []
    if not skills:
        return matched_skills
    for keyword in extracted_keywords:
        result = process.extractOne(keyword, skills)
        if result:
            match, score = result
            if score >= threshold:
                matched_skills.append((keyword, match, score))
    return set([match for _, match, _ in matched_skills])

@app.route("/extract-keywords", methods=["POST"])
def extract_keywords():
    try:
        data = request.get_json()
        if "job_description" not in data or "job_title" not in data:
            logger.warning("Missing 'job_description' or 'job_title' in request.")
            return jsonify({"error": "Missing 'job_description' or 'job_title' in request"}), 400

        job_description = data["job_description"]
        job_title = data["job_title"]

        # Step 1: Get skills for the job title
        skills = industry_skills.get(job_titles.get(job_title, ""), [])
        if not skills:
            logger.warning(f"No skills found for job title '{job_title}'.")
            return jsonify({"error": f"No skills found for job title '{job_title}'"}), 404

        # Step 2: Extract keywords using GPT-4
        extracted_keywords = extract_keywords_with_gpt4(job_description)
        logger.info(f"Extracted Keywords: {extracted_keywords}")

        # Step 3: Match extracted keywords with skills
        matched_skills = match_skills_with_keywords(extracted_keywords, skills)
        logger.info(f"Matched Skills: {matched_skills}")

        return jsonify({
            "job_title": job_title,
            "extracted_keywords": extracted_keywords,
            "matched_skills": list(matched_skills)
        })
    except Exception as e:
        logger.error(f"Error in API endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Keyword Extractor Service (OpenAI GPT-4)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on (default: 5000)")
    args = parser.parse_args()
    try:
        logger.info(f"Keyword Extractor Service is running on http://0.0.0.0:{args.port}")
        app.run(host="0.0.0.0", port=args.port)
    except KeyboardInterrupt:
        logger.info("Keyword Extractor Service is shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")