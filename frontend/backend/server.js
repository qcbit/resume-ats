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

// POST /analyze endpoint
app.post('/analyze', upload.single('resume'), async (req, res) => {
  try {
    const file = req.file;
    const jobDescription = req.body.jobDescription;
    // Accept jobTitle optionally from the request
    const jobTitleFromClient = req.body.jobTitle;

    if (!file || !jobDescription) {
      return res.status(400).json({ error: 'Missing resume or job description' });
    }

    // 1. Extract text from resume
    const resumeText = await extractText(file);

    // 2.1 Job title detection (only if not provided by client)
    let job_title = jobTitleFromClient;
    if (!job_title) {
      const jobTitleRes = await fetch('http://localhost:5003/detect-job-title', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jobDescription }),
      });
      const jobTitleData = await jobTitleRes.json();
      job_title = jobTitleData.job_title;
    }

    // 2.2 Keyword extraction
    const [jdKeywordsRes, resumeKeywordsRes] = await Promise.all([
      fetch('http://localhost:5001/extract-keywords', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jobDescription, job_title }),
      }),
      fetch('http://localhost:5001/extract-keywords', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: resumeText, job_title }),
      }),
    ]);
        // Example mock response for jdKeywordsRes
    // const jdKeywordsRes = {
    //   json: async () => ({
    //     job_title: "Software Engineer",
    //     extracted_keywords: ["Java", "Python", "C++"],
    //     matched_skills: ["Java", "Python"]
    //   })
    // };
    // const resumeKeywordsRes = {
    //   json: async () => ({
    //     job_title: "Software Engineer",
    //     extracted_keywords: ["Java", "Python", "C++"],
    //     matched_skills: ["Java", "Python"]
    //   })
    // };
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
    console.log('JD Matched Skills:', jdMatchedSkills);
    console.log('Resume Matched Skills:', resumeMatchedSkills);

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
    const matchRes = await fetch('http://localhost:5002/predict', {
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
      throw err;
    }

    // 3. Generate feedback (simple)
    const score = Math.round((matchData.equivalent || 0) * 100);

    // 4. Respond
    res.json({
      jobTitle: job_title,
      jdKeywords,
      resumeKeywords,
      score,
      feedback:
        score > 85
          ? 'Excellent match!'
          : score > 70
          ? 'Good match, but some improvements possible.'
          : 'Consider improving your resume for this job.',
    });

    // 5. Clean up uploaded file
    fs.unlinkSync(file.path);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Processing failed' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});