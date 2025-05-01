# Match Scorer Service (OpenAI)

## Overview

This service provides an analysis of how well a given resume matches a job description using the OpenAI API (specifically, models like GPT-4o, GPT-4 Turbo, or GPT-3.5 Turbo that support JSON mode). It returns a structured JSON response containing an overall match score, category-specific scores, and textual feedback.

## API Endpoint

### `/predict`

*   **Method:** `POST`
*   **Description:** Analyzes the match between a resume and a job description.
*   **Request Body:** JSON object containing:
    *   `resume_text` (string): The full text content of the resume.
    *   `job_description` (string): The full text content of the job description.
    ```json
    {
      "resume_text": "...",
      "job_description": "..."
    }
    ```
*   **Response Body (Success - 200 OK):** JSON object containing:
    *   `score` (float): Overall match score (0-100).
    *   `categories` (object): Scores for different sections.
        *   `skills` (object): `{ "score": float, "relevance": float }`
        *   `experience` (object): `{ "score": float, "relevance": float }`
        *   `education` (object): `{ "score": float, "relevance": float }`
        *   `achievements` (object): `{ "score": float, "relevance": float }`
    *   `feedback` (string): Textual analysis and suggestions.
    ```json
    {
      "score": 85.5,
      "categories": {
        "skills": { "score": 90.0, "relevance": 25 },
        "experience": { "score": 80.0, "relevance": 25 },
        "education": { "score": 85.0, "relevance": 25 },
        "achievements": { "score": 87.0, "relevance": 25 }
      },
      "feedback": "The resume shows a strong alignment with the required skills and experience..."
    }
    ```
*   **Response Body (Error):** JSON object with an `error` key.

## Dependencies

*   Python 3.x
*   Flask
*   OpenAI Python library (`openai`)
*   Logging library (e.g., `loguru` or standard `logging`)

See `requirements.txt` for specific versions.

## Configuration

The service is configured using environment variables:

*   `OPENAI_API_KEY`: **Required.** Your secret API key for accessing the OpenAI API. This is typically injected via a Kubernetes secret (`openai-secret`).
*   `OPENAI_MODEL`: The specific OpenAI model to use (e.g., `gpt-4o`, `gpt-4-turbo`). Defaults to `gpt-4o`. Must be a model supporting JSON mode.
*   `PORT`: The port the Flask application will listen on. Defaults to `5000`.

## Running

### Locally (for development)

1.  **Set Environment Variables:**
    ```bash
    export OPENAI_API_KEY="YOUR_API_KEY"
    # Optionally set OPENAI_MODEL and PORT
    # export OPENAI_MODEL="gpt-4o"
    # export PORT=5001
    ```
3.  **Run the Flask App:**
    ```bash
    cd match-scorer-openai
    uv run main.py [--port <port_number>] [--openai-model <model_name>]
    ```

### With Docker & Kubernetes

1.  **Build the Docker Image:** (Use the main backend Makefile)
    ```bash
    # From the backend directory
    make build-match-scorer-openai
    ```
3.  **Deploy to Kubernetes:** (Use the main backend Makefile)
    ```bash
    # From the backend directory
    make deploy-match-scorer-openai
    ```
    This uses the `backend/deployment/dev/match-scorer-openai.yaml` deployment file, which references the `openai-secret`.

## Notes

*   Ensure the selected `OPENAI_MODEL` supports the `response_format={"type": "json_object"}` parameter required by this service.
*   The service expects the `OPENAI_API_KEY` environment variable to be set; it will fail to start otherwise.