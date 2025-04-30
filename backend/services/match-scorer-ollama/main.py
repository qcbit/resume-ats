from flask import Flask, request, jsonify
import requests
import os
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Default configuration
DEFAULT_PORT = 5000
DEFAULT_OLLAMA_SERVICE_URL = "http://localhost:8000/generate"
DEFAULT_MODEL = "llama3:8b"

# Set Ollama URL from environment or use default
OLLAMA_SERVICE_URL = os.environ.get("OLLAMA_SERVICE_URL", DEFAULT_OLLAMA_SERVICE_URL)
MODEL = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

def get_match_score(resume_text, job_description):
    """
    Use Ollama service to determine the match score between a resume and job description
    """
    system_prompt = """
    You are an expert ATS (Applicant Tracking System) analyzer. Your task is to evaluate how well a resume matches a job description.
    Provide a detailed analysis of the match between the candidate's resume and the job requirements.
    Focus on skills, experience, education, and overall alignment.
    
    Return your response as a JSON object with the following fields:
    - equivalent: a float between 0 and 1 representing the overall match score
    - not_equivalent: a float that equals 1 - equivalent
    - predicted_class: "equivalent" if the match score > 0.5, otherwise "not_equivalent"
    - analysis: a brief explanation of the score
    """
    
    prompt = f"""
    === JOB DESCRIPTION ===
    {job_description}
    
    === RESUME ===
    {resume_text}
    
    Based on the above resume and job description, provide a match analysis.
    Remember to format your response as a JSON object.
    """
    
    try:
        # Log URL for debugging
        logger.info(f"Sending request to Ollama service at: {OLLAMA_SERVICE_URL}")
        
        # Make the request to Ollama service
        response = requests.post(
            OLLAMA_SERVICE_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "system": system_prompt
            }
        )
        
        # Log the full response for debugging
        logger.info(f"Ollama service response: {response.text[:500]}...")  # Truncate if too long
        
        # Check if request was successful
        response.raise_for_status()
        
        # Extract the generated text
        result = response.json()
        response_text = result.get("response", "")
        
        logger.info(f"LLM response text: {response_text[:500]}...")
        
        # Parse the JSON response from the LLM
        import json
        import re
        
        try:
            # First, try to parse the entire response as JSON
            try:
                parsed_result = json.loads(response_text)
                logger.info("Successfully parsed complete response as JSON")
            except json.JSONDecodeError:
                # If that fails, try to extract JSON using regex
                logger.info("Response is not valid JSON, trying to extract JSON object")
                json_pattern = r'\{[\s\S]*\}'
                match = re.search(json_pattern, response_text)
                
                if match:
                    json_str = match.group(0)
                    try:
                        parsed_result = json.loads(json_str)
                        logger.info("Successfully extracted and parsed JSON from response")
                    except json.JSONDecodeError:
                        raise ValueError("Extracted text is not valid JSON")
                else:
                    raise ValueError("No JSON-like structure found in response")
            
            # Ensure all required fields are present
            if "equivalent" not in parsed_result:
                if "match_score" in parsed_result:
                    parsed_result["equivalent"] = parsed_result["match_score"]
                else:
                    parsed_result["equivalent"] = 0.5
                    logger.warning("No 'equivalent' or 'match_score' field found, using default 0.5")
            
            # Calculate not_equivalent if missing
            if "not_equivalent" not in parsed_result:
                parsed_result["not_equivalent"] = 1 - float(parsed_result["equivalent"])
            
            # Set predicted_class if missing
            if "predicted_class" not in parsed_result:
                parsed_result["predicted_class"] = "equivalent" if float(parsed_result["equivalent"]) > 0.5 else "not_equivalent"
            
            # Ensure analysis is present
            if "analysis" not in parsed_result:
                if "explanation" in parsed_result:
                    parsed_result["analysis"] = parsed_result["explanation"]
                elif "reasoning" in parsed_result:
                    parsed_result["analysis"] = parsed_result["reasoning"]
                else:
                    parsed_result["analysis"] = "No detailed analysis provided by model."
            
            logger.info(f"Processed result: {parsed_result}")
            return parsed_result
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return a default response
            return {
                "equivalent": 0.5,
                "not_equivalent": 0.5,
                "predicted_class": "not_equivalent",
                "analysis": "Error processing match score - could not parse LLM response"
            }
            
    except requests.RequestException as e:
        logger.error(f"Error calling Ollama service: {e}")
        return {
            "equivalent": 0,
            "not_equivalent": 1,
            "predicted_class": "not_equivalent",
            "analysis": f"Error: Failed to connect to Ollama service - {str(e)}"
        }
    except Exception as e:
        logger.exception(f"Unexpected error processing match score: {e}")
        return {
            "equivalent": 0.5,
            "not_equivalent": 0.5,
            "predicted_class": "not_equivalent",
            "analysis": f"Error processing match score - {str(e)}"
        }

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    resume_text = data.get("sentence1")
    job_description = data.get("sentence2")
    
    if not resume_text or not job_description:
        return jsonify({
            "error": "Both resume and job description are required."
        }), 400
    
    # Get match score from Ollama
    result = get_match_score(resume_text, job_description)
    return jsonify(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match Scorer Service using Ollama")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to run the Flask app on (default: {DEFAULT_PORT})")
    parser.add_argument("--ollama-url", type=str, help="URL of the Ollama service")
    parser.add_argument("--model", type=str, help="Model to use for matching")
    
    args = parser.parse_args()
    
    # Override defaults with command line arguments if provided
    if args.ollama_url:
        OLLAMA_SERVICE_URL = args.ollama_url
    
    if args.model:
        MODEL = args.model
    
    logger.info(f"Starting Match Scorer service on port {args.port}")
    logger.info(f"Using Ollama service at: {OLLAMA_SERVICE_URL}")
    logger.info(f"Using model: {MODEL}")
    
    app.run(host="0.0.0.0", port=args.port)
