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

const JOB_TITLE_SERVICE_URL = process.env.JOB_TITLE_SERVICE_URL || 'http://localhost:5001/detect-job-title';
const KEYWORDS_SERVICE_URL = process.env.KEYWORDS_SERVICE_URL || 'http://localhost:5002/extract-keywords';
const PREDICT_SERVICE_URL = process.env.PREDICT_SERVICE_URL || 'http://localhost:5003/predict';

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
  console.log("Received request for /api/analyze"); // Add log
  const file = req.file;
  const { jobDescription } = req.body;

  if (!file || !jobDescription) {
    console.error("Missing file or job description");
    return res.status(400).json({ error: 'Missing resume file or job description' });
  }

  try {
    console.log("Extracting text from:", file.originalname);
    const resumeText = await extractText(file);
    console.log("Text extracted, length:", resumeText.length);

    // 2.1 Job title detection (only if not provided by client)
    let job_title = req.body.jobTitle;
    if (!job_title) {
      const jobTitleRes = await fetch(JOB_TITLE_SERVICE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jobDescription }),
      });
      const jobTitleData = await jobTitleRes.json();
      job_title = jobTitleData.job_title;
    }

    // 2.2 Keyword extraction
    const [jdKeywordsRes, resumeKeywordsRes] = await Promise.all([
      fetch(KEYWORDS_SERVICE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jobDescription, job_title }),
      }),
      fetch(KEYWORDS_SERVICE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: resumeText, job_title }),
      }),
    ]);
    const jdKeywordsJson = await jdKeywordsRes.json();
    const resumeKeywordsJson = await resumeKeywordsRes.json();

    const jdKeywords = Array.isArray(jdKeywordsJson.extracted_keywords)
      ? jdKeywordsJson.extracted_keywords
      : [];
    const resumeKeywords = Array.isArray(resumeKeywordsJson.extracted_keywords)
      ? resumeKeywordsJson.extracted_keywords
      : [];

    const jdMatchedSkills = Array.isArray(jdKeywordsJson.matched_skills)
      ? jdKeywordsJson.matched_skills
      : [];
    const resumeMatchedSkills = Array.isArray(resumeKeywordsJson.matched_skills)
      ? resumeKeywordsJson.matched_skills
      : [];
    console.log('JD Matched Skills:', jdMatchedSkills);
    console.log('Resume Matched Skills:', resumeMatchedSkills);
    // Defensive: fallback to empty string if keywords are empty
    const resumeMatchedSkillsStr = resumeMatchedSkills.length
      ? resumeMatchedSkills.join(' ')
      : '';
    const jdMatchedSkillsStr = jdMatchedSkills.length
      ? jdMatchedSkills.join(' ')
      : '';
    console.log('JD Matched Skills Str:', jdMatchedSkillsStr);
    console.log('Resume Matched Skills Str:', resumeMatchedSkillsStr);

    console.log('JD Keywords:', jdKeywords);
    console.log('Resume Keywords:', resumeKeywords);

    // Defensive: fallback to empty string if keywords are empty
    const resumeKeywordsStr = resumeKeywords.length ? resumeKeywords.join(' ') : '';
    const jdKeywordsStr = jdKeywords.length ? jdKeywords.join(' ') : '';

    // 2.3 Match scoring
    console.log('Sending to predict:', {
      sentence1: resumeText + (resumeMatchedSkillsStr ? ' ' + resumeMatchedSkillsStr : ''),
      sentence2: jobDescription + (jdMatchedSkillsStr ? ' ' + jdMatchedSkillsStr : ''),
    });
    const matchRes = await fetch(PREDICT_SERVICE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sentence1: resumeText + (resumeMatchedSkillsStr ? ' ' + resumeMatchedSkillsStr : ''),
        sentence2: jobDescription + (jdMatchedSkillsStr ? ' ' + jdMatchedSkillsStr : ''),
      }),
    });
    const matchText = await matchRes.text();
    console.log('Predict service response:', matchText);
    let matchData;
    try {
      matchData = JSON.parse(matchText);
    } catch (err) {
      console.error('Failed to parse JSON from predict service');
      const sanitizedResponse = matchText.length > 100 ? matchText.substring(0, 100) + '...' : matchText;
      console.error('Sanitized response:', sanitizedResponse);
      throw err;
    }

    // Extract the analysis from the match data for richer feedback
    const analysis = matchData.analysis || '';

    // 3. Generate feedback (enhanced with LLM analysis)
    const score = Math.round((matchData.equivalent || 0) * 100);

    // 4. Respond with enhanced data
    res.json({
      skills: [],
      jobTitle: job_title,
      jdKeywords,
      resumeKeywords,
      score,
      analysis,
      feedback:
        score > 85
          ? 'Excellent match!'
          : score > 70
          ? 'Good match, but some improvements possible.'
          : 'Consider improving your resume for this job.',
    });

    // 5. Clean up uploaded file
    fs.unlinkSync(file.path);
    console.log("Cleaned up file:", file.path);
  } catch (error) {
    console.error('Error processing /api/analyze:', error);
    res.status(500).json({ error: 'Failed to analyze resume', details: error.message });
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