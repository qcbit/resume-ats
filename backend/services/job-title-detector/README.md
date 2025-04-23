# Job Title Detector

The Job Title Detector uses a combination of embedding similarity and fuzzy matching to detect the job title from the job description. It references a list of job titles and does a best match the determine the job title.

## To Run

uv run services/job-title-detector/main.py

## Test

Run the following on the terminal:

curl -X POST http://127.0.0.1:5000/detect-job-title \
-H "Content-Type: application/json" \
-d '{"job_description": "Alexander Technology Group is looking for a direct hire ERP Business Analyst."}'
