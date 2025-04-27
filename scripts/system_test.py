import requests
import json

# Update these URLs/ports as needed for your deployment
JOB_TITLE_DETECTOR_URL = "http://localhost:5003/detect-job-title"
KEYWORDS_EXTRACTOR_URL = "http://localhost:5001/extract-keywords"
MATCH_SCORER_URL = "http://localhost:5002/predict"

def test_job_title_detector(job_description):
    print("Testing job-title-detector service...")
    payload = {"job_description": job_description}
    try:
        response = requests.post(JOB_TITLE_DETECTOR_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("Job Title Detector Response:", data)
        return data.get("job_title")
    except Exception as e:
        print(f"Job title detector failed: {e}")
        return None

def test_keywords_extractor(text, job_title=""):
    print("Testing keywords-extractor service...")
    payload = {"job_description": text, "job_title": job_title}
    try:
        response = requests.post(KEYWORDS_EXTRACTOR_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("Keywords Extractor Response:", data)
        return data.get("extracted_keywords", [])
    except Exception as e:
        print(f"Keywords extractor failed: {e}")
        return []

def test_match_scorer(resume, job_description):
    print("Testing match-scorer service...")
    payload = {"sentence1": resume, "sentence2": job_description}
    try:
        response = requests.post(MATCH_SCORER_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("Match Scorer Response:", data)
        return data
    except Exception as e:
        print(f"Match scorer failed: {e}")
        return None

if __name__ == "__main__":

    # Example test data
    job_description = "We are looking for a Senior Data Analyst with experience in SQL, Tableau, and business process improvement."
    resume = "Experienced Data Analyst skilled in SQL, Tableau, and data visualization. Proven track record in business process optimization."

    resume = "Experienced software engineer skilled in Python, machine learning, and cloud computing."
    job_description = "We are looking for a software engineer with expertise in Python, machine learning, and cloud computing."

    resume = "Experienced Machine Learning Engineer skilled in Python, TensorFlow, and data analysis."
    job_description = "We are looking for an experienced Machine Learning Engineer with expertise in Python, TensorFlow, and data analysis."

    print("=== Microservices Integration Test ===\n")

    # 1. Test job-title-detector
    detected_title = test_job_title_detector(job_description)
    print(f"Detected Job Title: {detected_title}\n")

    # 2. Test keywords-extractor for job description and resume
    jd_keywords = test_keywords_extractor(job_description, detected_title)
    print(f"Extracted Job Description Keywords: {jd_keywords}\n")

    resume_keywords = test_keywords_extractor(resume, detected_title)
    print(f"Extracted Resume Keywords: {resume_keywords}\n")

    # 3. Test match-scorer
    # Concatenate extracted keywords to the original texts
    resume_with_keywords = resume + " " + " ".join(resume_keywords)
    job_description_with_keywords = job_description + " " + " ".join(jd_keywords)

    match_score = test_match_scorer(resume_with_keywords, job_description_with_keywords)
    print(f"Match Score: {match_score}\n")

    print("=== Test Complete ===")