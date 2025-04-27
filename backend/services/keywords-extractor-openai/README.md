# Job Keywords Extractor

The Job Keywords Extractor uses semantic similarity to extract keywords that relate to industry skills for a job title.

## To Run

uv run services/keywords-extractor/main.py

## Test

Run the following on the terminal:

curl -X POST http://127.0.0.1:5001/extract-keywords \
-H "Content-Type: application/json" \
-d '{
  "job_description": "We are looking for an ERP Business Analyst with experience in Dynamics 365 and SAP.",
  "job_title": "Business Analyst"
}'
