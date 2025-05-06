# Job Title Detector

The Job Title Detector service identifies job titles from job descriptions using a combination of embedding similarity and fuzzy matching algorithms. 

## Features

- Accurate extraction of job titles from unstructured text
- Embedding-based semantic similarity matching
- Fuzzy string matching for handling variations in titles
- Comprehensive reference database of common job titles
- RESTful API interface
- Integration with the broader resume-ATS system

## Prerequisites

- Python 3.8+
- Flask
- Sentence Transformers
- Fuzzy-Wuzzy
- A pre-trained embedding model

## Usage

```sh
uv run services/job-title-detector/main.py [--port PORT]
```

## API Endpoints

**POST /detect-job-title**

Extracts the job title from a provided job description.

## Request Format

```json
{
  "job_description": "Alexander Technology Group is looking for a direct hire ERP Business Analyst."
}
```

## Response Format

```json
{
  "job_title": "ERP Business Analyst"
}
```

## Testing

You can test the service using curl:

```sh
curl -X POST http://127.0.0.1:5000/detect-job-title \
-H "Content-Type: application/json" \
-d '{"job_description": "Alexander Technology Group is looking for a direct hire ERP Business Analyst."}'
```

## Deployment

See Makefile.

The job title detector helps categorize job descriptions, enabling more accurate matching and analysis.

## Links

[Frontend Documentation](../../../frontend/README.md)

[Backend Services Documentation](../../backend/README.md)
