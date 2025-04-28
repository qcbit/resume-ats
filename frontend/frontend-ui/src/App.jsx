import React from 'react';
import Header from './components/Header';
import ResumeUploader from './components/ResumeUploader';
import JobDetails from './components/JobDetails';
import AnalysisResults from './components/AnalysisResults';
import Footer from './components/Footer';
import './App.css';

function App() {
  const [resumeText, setResumeText] = React.useState('');
  const [jobTitle, setJobTitle] = React.useState('');
  const [jobDescription, setJobDescription] = React.useState('');
  const [analysisResults, setAnalysisResults] = React.useState(null);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);

  const handleResumeUpload = (text) => {
    setResumeText(text);
  };

  const handleAnalyze = () => {
    if (!resumeText || !jobDescription) {
      alert('Please provide both a resume and job description');
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate analysis with a timeout
    setTimeout(() => {
      const results = analyzeResume(resumeText, jobTitle, jobDescription);
      setAnalysisResults(results);
      setIsAnalyzing(false);
    }, 1500);
  };

  return (
    <div className="app">
      <Header />
      <main className="container">
        <div className="app-content">
          <div className="input-section">
            <ResumeUploader onResumeUpload={handleResumeUpload} />
            <JobDetails 
              jobTitle={jobTitle}
              jobDescription={jobDescription}
              onJobTitleChange={setJobTitle}
              onJobDescriptionChange={setJobDescription}
            />
            <button 
              className="analyze-button" 
              onClick={handleAnalyze}
              disabled={isAnalyzing || !resumeText || !jobDescription}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Match'}
            </button>
          </div>
          
          {analysisResults && (
            <AnalysisResults results={analysisResults} />
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
}

// Function to analyze resume against job description
function analyzeResume(resumeText, jobTitle, jobDescription) {
  // This is a simplified analysis algorithm
  // In a real application, you would use NLP or ML techniques
  
  const jobKeywords = extractKeywords(jobDescription);
  const resumeKeywords = extractKeywords(resumeText);
  
  // Calculate overall match score
  const matchedKeywords = resumeKeywords.filter(keyword => 
    jobKeywords.includes(keyword)
  );
  
  const overallScore = Math.min(
    Math.round((matchedKeywords.length / Math.max(jobKeywords.length, 1)) * 100),
    100
  );
  
  // Calculate category scores
  const categories = {
    skills: calculateCategoryScore(resumeText, jobDescription, ['skill', 'technology', 'software', 'proficient']),
    experience: calculateCategoryScore(resumeText, jobDescription, ['experience', 'year', 'work', 'position']),
    education: calculateCategoryScore(resumeText, jobDescription, ['education', 'degree', 'university', 'college']),
    achievements: calculateCategoryScore(resumeText, jobDescription, ['achievement', 'award', 'recognition', 'accomplish'])
  };
  
  // Generate improvement suggestions
  const suggestions = generateSuggestions(resumeText, jobDescription, jobTitle, categories);
  
  return {
    overallScore,
    categories,
    suggestions,
    matchedKeywords,
    missingKeywords: jobKeywords.filter(keyword => !resumeKeywords.includes(keyword))
  };
}

function extractKeywords(text) {
  // Simple keyword extraction (would be more sophisticated in a real app)
  const words = text.toLowerCase().match(/\b\w{3,}\b/g) || [];
  const stopWords = ['and', 'the', 'for', 'with', 'that', 'this', 'are', 'from'];
  return [...new Set(words.filter(word => !stopWords.includes(word)))];
}

function calculateCategoryScore(resumeText, jobDescription, categoryKeywords) {
  const relevantJobContent = categoryKeywords.some(keyword => 
    jobDescription.toLowerCase().includes(keyword)
  ) ? 1 : 0;
  
  if (!relevantJobContent) return { score: 0, relevance: 'low' };
  
  const categoryMatches = categoryKeywords.filter(keyword => 
    resumeText.toLowerCase().includes(keyword)
  ).length;
  
  const score = Math.min(Math.round((categoryMatches / categoryKeywords.length) * 100), 100);
  
  let relevance = 'low';
  if (score > 70) relevance = 'high';
  else if (score > 40) relevance = 'medium';
  
  return { score, relevance };
}

function generateSuggestions(resumeText, jobDescription, jobTitle, categories) {
  const suggestions = [];
  
  // Add general suggestions
  if (categories.skills.score < 60) {
    suggestions.push("Consider adding more relevant technical skills that match the job requirements.");
  }
  
  if (categories.experience.score < 60) {
    suggestions.push("Your experience section could be better aligned with the job requirements. Focus on relevant work history.");
  }
  
  if (categories.education.score < 60) {
    suggestions.push("Consider highlighting educational qualifications that are relevant to this position.");
  }
  
  // Add specific keyword suggestions
  const jobKeywords = extractKeywords(jobDescription);
  const resumeKeywords = extractKeywords(resumeText);
  const missingKeywords = jobKeywords.filter(keyword => 
    !resumeKeywords.includes(keyword) && 
    keyword.length > 3 && 
    !['with', 'from', 'have', 'that', 'this', 'will', 'about'].includes(keyword)
  );
  
  if (missingKeywords.length > 0) {
    const keywordsToShow = missingKeywords.slice(0, 5);
    suggestions.push(`Consider incorporating these keywords from the job description: ${keywordsToShow.join(', ')}${missingKeywords.length > 5 ? '...' : ''}`);
  }
  
  // Add job title specific suggestion
  if (jobTitle && !resumeText.toLowerCase().includes(jobTitle.toLowerCase())) {
    suggestions.push(`The job title "${jobTitle}" doesn't appear in your resume. Consider including it if you have relevant experience.`);
  }
  
  return suggestions;
}

export default App;
