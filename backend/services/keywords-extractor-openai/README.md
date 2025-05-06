# Job Keywords Extractor (OpenAI)

## Overview

The Job Keywords Extractor service uses OpenAI's language models to extract relevant industry skills and keywords from job descriptions. This microservice helps identify critical terms for resume matching and analysis within the resume-ATS ecosystem.

## Features

- Semantic extraction of skills and keywords from job descriptions
- Leverages OpenAI's advanced language models
- Intelligent categorization of technical and soft skills
- Customizable relevance scoring
- RESTful API interface
- Seamless integration with the resume-ATS system

## Prerequisites

- Python 3.8+
- Flask
- OpenAI Python client
- Access to OpenAI API (API key required)

## Usage

Environment Setup

Create a .env file with your OpenAI API key:

```sh
OPENAI_API_KEY=your_api_key_here
```

Starting the Service

```sh
# Using uv
cd services/keywords-extractor-openai
uv run main.py [--port PORT]

# Or using python directly
python main.py [--port PORT]
```

## API Endpoints

**POST /extract-keywords**

Extracts relevant keywords and skills from a job description.

## Request Format

```json
{
  "job_description": "We are looking for an ERP Business Analyst with experience in Dynamics 365 and SAP.",
  "job_title": "Business Analyst"
}
```

## Response Format

```json
{
  "keywords": [
    "ERP", 
    "Business Analyst", 
    "Dynamics 365", 
    "SAP"
  ]
}
```

## Testing

You can test the service using curl:

```sh
curl -X POST http://127.0.0.1:5000/extract-keywords \
-H "Content-Type: application/json" \
-d '{
  "job_description": "We are looking for an ERP Business Analyst with experience in Dynamics 365 and SAP.",
  "job_title": "Business Analyst"
}'
```

## Deployment

See Makefile

## Links

[Frontend Documentation](../../../frontend/README.md)

[Backend Services Documentation](../../README.md)