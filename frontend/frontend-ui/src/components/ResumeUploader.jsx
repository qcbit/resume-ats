import React, { useState, useCallback } from 'react';
import './ResumeUploader.css';

function ResumeUploader({ onResumeUpload }) {
  const [fileName, setFileName] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(''); // State for inline error message

  // Wrap handleFileChange in useCallback
  const handleFileChange = useCallback((file) => {
    setError(''); // Clear previous errors on new attempt
    if (file) {
      const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
      if (allowedTypes.includes(file.type)) {
        setFileName(file.name);
        onResumeUpload(file); // Dependency
      } else {
        // Set inline error message instead of alert
        setError('Unsupported file type. Please upload .txt, .pdf, .doc, or .docx');
        setFileName(''); // Reset file name
        onResumeUpload(null); // Notify parent that upload is invalid/cleared
      }
    }
  }, [onResumeUpload]); // Add onResumeUpload as a dependency

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
    setError(''); // Clear error when dragging over
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  // Update handleDrop dependencies
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    setError(''); // Clear error on drop attempt
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileChange(files[0]); // Call the memoized version
      if (e.dataTransfer.items) {
        e.dataTransfer.items.clear();
      } else {
        e.dataTransfer.clearData();
      }
    }
  }, [handleFileChange]); // Correct dependency

  return (
    <div
      className={`resume-uploader ${isDragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <h2>Upload Your Resume</h2>
      <div className="file-upload">
        <label htmlFor="resume-file" className="file-label">
          <span className="file-icon">📄</span>
          <span>{fileName || (isDragging ? 'Drop file here' : 'Choose or drop a file')}</span>
        </label>
        <input
          type="file"
          id="resume-file"
          accept=".pdf,.doc,.docx,.txt"
          style={{ display: 'none' }}
          onChange={e => {
            const file = e.target.files[0];
            // Clear input value to allow re-uploading the same file after an error
            e.target.value = null;
            handleFileChange(file); // Call the memoized version
          }}
        />
        {/* Conditionally render the error message */}
        {error && <p className="file-error">{error}</p>}
        <p className="file-help">Supported formats: .txt, .pdf, .doc, .docx</p>
      </div>
    </div>
  );
}

export default ResumeUploader;
