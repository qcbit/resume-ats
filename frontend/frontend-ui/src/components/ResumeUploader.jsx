import React, { useState, useCallback } from 'react';
import './ResumeUploader.css';

function ResumeUploader({ onResumeUpload }) {
  const [fileName, setFileName] = useState('');
  const [isDragging, setIsDragging] = useState(false); // State for drag feedback

  const handleFileChange = (file) => {
    if (file) {
      // Basic validation for allowed types (can be more robust)
      const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
      if (allowedTypes.includes(file.type)) {
        setFileName(file.name);
        onResumeUpload(file);
      } else {
        alert('Unsupported file type. Please upload .txt, .pdf, .doc, or .docx');
        setFileName(''); // Reset file name if invalid
      }
    }
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault(); // Prevent default behavior (opening file)
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileChange(files[0]); // Handle the first dropped file
      // Clear the dataTransfer buffer (important for some browsers)
      if (e.dataTransfer.items) {
        e.dataTransfer.items.clear();
      } else {
        e.dataTransfer.clearData();
      }
    }
  }, [onResumeUpload]); // Include onResumeUpload in dependency array if it could change

  return (
    <div
      className={`resume-uploader ${isDragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <h2>Upload Your Resume</h2>
      <div className="file-upload">
        {/* Label now acts as the drop zone visually */}
        <label htmlFor="resume-file" className="file-label">
          <span className="file-icon">📄</span>
          <span>{fileName || (isDragging ? 'Drop file here' : 'Choose or drop a file')}</span>
        </label>
        <input
          type="file"
          id="resume-file"
          accept=".pdf,.doc,.docx,.txt"
          style={{ display: 'none' }} // Keep hidden, label handles interaction
          onChange={e => {
            const file = e.target.files[0];
            handleFileChange(file);
          }}
        />
        <p className="file-help">Supported formats: .txt, .pdf, .doc, .docx</p>
      </div>
    </div>
  );
}

export default ResumeUploader;
