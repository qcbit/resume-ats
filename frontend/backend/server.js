const express = require('express');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const fs = require('fs');
const cors = require('cors');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(express.json());

// Ensure these point to the correct K8s service names and ports
const JOB_TITLE_SERVICE_URL = process.env.JOB_TITLE_SERVICE_URL || 'http://job-title-detector:5000/detect-job-title';
const KEYWORDS_SERVICE_URL = process.env.KEYWORDS_SERVICE_URL || 'http://keywords-extractor-openai:5000/extract-keywords';
const PREDICT_SERVICE_URL = process.env.PREDICT_SERVICE_URL || 'http://match-scorer-openai:5000/predict';

// Helper: Extract text from file
async function extractText(file) {
  if (file.mimetype === 'application/pdf') {
    const data = fs.readFileSync(file.path);
    const pdfData = await pdfParse(data);
    return pdfData.text;
  } else if (
    file.mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    file.mimetype === 'application/msword'
  ) {
    const result = await mammoth.extractRawText({ path: file.path });
    return result.value;
  } else {
    // Plain text
    return fs.readFileSync(file.path, 'utf8');
  }
}

// POST /api/analyze endpoint
app.post('/api/analyze', upload.single('resume'), async (req, res) => {
  console.log("Received request for /api/analyze");
  const file = req.file;
  const { jobDescription } = req.body;

  if (!file || !jobDescription) {
    console.error("Missing file or job description");
    return res.status(400).json({ error: 'Missing resume file or job description' });
  }

  let analysisResult = {}; // To store results from different services

  try {
    console.log("Extracting text from:", file.originalname);
    const resumeText = await extractText(file);
    console.log("Text extracted, length:", resumeText.length);
    analysisResult.resumeText = resumeText; // Add resumeText to the results

    // --- Call Microservices ---
    // 1. Job title detection
    console.log('Sending request to JOB_TITLE_SERVICE_URL:', JOB_TITLE_SERVICE_URL);
    const jobTitleRes = await fetch(JOB_TITLE_SERVICE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_description: jobDescription }),
    });
    if (!jobTitleRes.ok) throw new Error(`Job title service failed: ${jobTitleRes.statusText}`);
    const jobTitleData = await jobTitleRes.json();
    if (!jobTitleData || !jobTitleData.job_title) {
      throw new Error('Job title property is missing in the response from JOB_TITLE_SERVICE_URL');
    }
    analysisResult.jobTitle = jobTitleData.job_title;
    console.log('Job Title received:', analysisResult.jobTitle);

    // 2. Keyword extraction (Parallel)
    console.log('Sending requests to KEYWORDS_SERVICE_URL:', KEYWORDS_SERVICE_URL);
    const [jdKeywordsRes] = await Promise.all([
      fetch(KEYWORDS_SERVICE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: jobDescription }), // Assuming KW service expects 'text'
      }),
      // Removed resume keyword extraction
    ]);
    if (!jdKeywordsRes.ok) throw new Error(`JD Keywords service failed: ${jdKeywordsRes.statusText}`);

    const jdKeywordsJson = await jdKeywordsRes.json();

    analysisResult.jdKeywords = Array.isArray(jdKeywordsJson.keywords) ? jdKeywordsJson.keywords : [];
    analysisResult.resumeKeywords = []; // Set to empty array as resume keywords are no longer fetched
    console.log('JD Keywords received:', analysisResult.jdKeywords);

    // 3. Prediction/Scoring using match-scorer-openai
    console.log('Sending request to PREDICT_SERVICE_URL:', PREDICT_SERVICE_URL);
    const predictTimeoutMs = 180000; // 3 minutes timeout for OpenAI
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), predictTimeoutMs);

    let predictionData = {}; // To store prediction results

    try {
      const predictRes = await fetch(PREDICT_SERVICE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resume_text: resumeText,
          job_description: jobDescription,
          // No longer sending skills/keywords here unless the scorer needs them
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId); // Clear timeout if fetch completes

      if (!predictRes.ok) {
        const errorBody = await predictRes.text();
        console.error(`Prediction service failed with status ${predictRes.status}: ${errorBody}`);
        throw new Error(`Prediction service failed: ${predictRes.statusText} - ${errorBody}`);
      }

      predictionData = await predictRes.json();
      console.log('Prediction received:', predictionData);

      // Merge prediction data into the main result object
      analysisResult.score = predictionData.score; // Overall score (0-100)
      analysisResult.categories = predictionData.categories; // Category scores/relevance
      analysisResult.feedback = predictionData.feedback; // Feedback/Analysis text

    } catch (error) {
      clearTimeout(timeoutId); // Ensure timeout is cleared on error too
      // Define a default categories structure
      const defaultCategories = {
        skills: { score: 0, relevance: 0 },
        experience: { score: 0, relevance: 0 },
        education: { score: 0, relevance: 0 },
        achievements: { score: 0, relevance: 0 }
      };

      if (error.name === 'AbortError') {
        console.error(`Prediction service timed out after ${predictTimeoutMs / 1000} seconds.`);
        analysisResult.score = 0;
        analysisResult.feedback = "Analysis timed out. Could not retrieve detailed scores.";
        // Use the default structure
        analysisResult.categories = defaultCategories;
        analysisResult.error = 'Prediction service timed out.';
      } else {
        console.error('Error calling prediction service:', error);
        analysisResult.score = 0;
        analysisResult.feedback = "Failed to get analysis score due to an error.";
        // Use the default structure
        analysisResult.categories = defaultCategories;
        analysisResult.error = `Prediction service failed: ${error.message}`;
      }
    }

    // --- Respond with combined results ---
    console.log("Sending final response:", analysisResult);
    res.json(analysisResult);

  } catch (error) {
    console.error('Error processing /api/analyze:', error);
    res.status(500).json({
        error: 'Failed to analyze resume',
        details: error.message // Send back the specific error message
    });
  } finally {
    // Clean up uploaded file regardless of success or failure
    if (file && file.path) {
      try {
        fs.unlinkSync(file.path);
        console.log("Cleaned up file:", file.path);
      } catch (cleanupError) {
        console.error("Error cleaning up file:", cleanupError);
      }
    }
  }
});

// Parse --port argument from command line
let cliPort = null;
const portArgIndex = process.argv.indexOf('--port');
if (portArgIndex !== -1 && process.argv[portArgIndex + 1]) {
  const parsed = parseInt(process.argv[portArgIndex + 1], 10);
  if (!isNaN(parsed)) {
    cliPort = parsed;
  }
}

const PORT = cliPort || process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => { // Ensure listening on 0.0.0.0
  console.log(`Backend server listening on port ${PORT}`);
});