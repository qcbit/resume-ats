import React from 'react';
import Header from './components/Header';
import ResumeUploader from './components/ResumeUploader';
import JobDetails from './components/JobDetails';
import AnalysisResults from './components/AnalysisResults';
import Footer from './components/Footer';
import './App.css';

const API_URL = import.meta.env.VITE_REACT_APP_API_URL || '/api';

function App() {
  const [resumeText, setResumeText] = React.useState('');
  const [jobTitle, setJobTitle] = React.useState('');
  const [jobDescription, setJobDescription] = React.useState('');
  const [analysisResults, setAnalysisResults] = React.useState(null);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);
  const [resumeFile, setResumeFile] = React.useState(null);

  console.log('API URL:', API_URL);

  const handleResumeUpload = (file) => {
    setResumeFile(file);
  };

  const handleAnalyze = async () => {
    if (!resumeFile || !jobDescription) {
      alert('Please provide both a resume and job description');
      return;
    }

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('resume', resumeFile);
      formData.append('jobDescription', jobDescription);
      if (jobTitle) formData.append('jobTitle', jobTitle);

      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze resume');
      }

      const results = await response.json();
      setAnalysisResults(results);
    } catch (error) {
      alert('Error analyzing resume: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
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
              disabled={isAnalyzing || !resumeFile || !jobDescription}
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

export default App;
