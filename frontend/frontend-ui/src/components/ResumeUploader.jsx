import React, { useState } from 'react';
import './ResumeUploader.css';

function ResumeUploader({ onResumeUpload }) {
  const [resumeContent, setResumeContent] = useState('');
  const [fileName, setFileName] = useState('');
  
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileName(file.name);

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      setResumeContent(text);
      onResumeUpload(text);
    };
    reader.onerror = (e) => {
      console.error('File reading error:', e);
      setResumeContent('');
      onResumeUpload('');
      alert('Failed to read the file. Please try a different file or format.');
    };
    reader.readAsText(file);
  };
  
  const handleTextChange = (e) => {
    setResumeContent(e.target.value);
    onResumeUpload(e.target.value);
  };
  
  return (
    <div className="resume-uploader">
      <h2>Upload Your Resume</h2>
      
      <div className="file-upload">
        <label htmlFor="resume-file" className="file-label">
          <span className="file-icon">📄</span>
          <span>{fileName || 'Choose a file'}</span>
        </label>
        <input 
          type="file" 
          id="resume-file" 
          accept=".txt,.pdf,.doc,.docx" 
          onChange={handleFileUpload}
        />
        <p className="file-help">Supported formats: .txt, .pdf, .doc, .docx</p>
      </div>
      
      <div className="text-input">
        <label htmlFor="resume-text">Or paste your resume text:</label>
        <textarea
          id="resume-text"
          value={resumeContent}
          onChange={handleTextChange}
          placeholder="Paste your resume content here..."
          rows="10"
        ></textarea>
      </div>
    </div>
  );
}

export default ResumeUploader;
