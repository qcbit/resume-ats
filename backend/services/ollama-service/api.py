# A simple Flask wrapper for Ollama
from flask import Flask, request, jsonify
import requests
import argparse
import os
import yaml
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Default configuration
DEFAULT_PORT = 8000
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3:8b"

def load_config_from_yaml(config_path):
    """Load configuration from a YAML file"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    # Get model from request or use default
    model = data.get('model', DEFAULT_MODEL)
    
    try:
        # Make the request to Ollama
        ollama_response = requests.post(app.config['OLLAMA_URL'], json={
            'model': model,
            'prompt': data['prompt'],
            'system': data.get('system', ''),
            'context': data.get('context', [])
        })
        
        # Log the raw response for debugging
        logger.info(f"Raw Ollama response: {ollama_response.text[:200]}...")
        
        # Check if request was successful
        ollama_response.raise_for_status()
        
        # Handle streaming response format - extract and combine all text chunks
        try:
            lines = ollama_response.text.strip().split('\n')
            combined_response = ""
            
            for line in lines:
                try:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        combined_response += chunk['response']
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line as JSON: {line[:50]}...")
            
            logger.info(f"Combined response: {combined_response[:200]}...")
            
            # Return the combined text
            return jsonify({
                "response": combined_response,
                "error": None
            })
            
        except Exception as e:
            logger.error(f"Error processing Ollama response: {e}")
            return jsonify({
                "response": "Error processing model response",
                "error": str(e)
            }), 500
            
    except requests.RequestException as e:
        logger.error(f"Error connecting to Ollama service: {e}")
        return jsonify({"error": str(e)}), 500
    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        return jsonify({"error": f"Missing required parameter: {e}"}), 400

if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Ollama API Service')
    parser.add_argument('--port', type=int, help='Port to run the service on')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--ollama-url', type=str, help='URL of the Ollama service')
    args = parser.parse_args()

    # Load config from YAML if specified
    config = {}
    if args.config:
        config = load_config_from_yaml(args.config)
    
    # Set port (priority: CLI arg > YAML > environment var > default)
    port = args.port or config.get('port') or int(os.environ.get('SERVICE_PORT', DEFAULT_PORT))
    
    # Set Ollama URL (priority: CLI arg > YAML > environment var > default)
    ollama_url = args.ollama_url or config.get('ollama_url') or os.environ.get('OLLAMA_URL', DEFAULT_OLLAMA_URL)
    
    # Update app config
    app.config['OLLAMA_URL'] = ollama_url

    logger.info(f"Starting Ollama service on port {port}, connecting to {ollama_url}")
    app.run(host='0.0.0.0', port=port)