import React, { useState } from 'react';
import './ResumeUploader.css';

function ResumeUploader({ onResumeUpload }) {
  const [fileName, setFileName] = useState('');

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
          accept=".pdf,.doc,.docx,.txt"
          style={{ display: 'none' }}
          onChange={e => {
            const file = e.target.files[0];
            if (file) {
              setFileName(file.name);
              onResumeUpload(file);
            }
          }}
        />
        <p className="file-help">Supported formats: .txt, .pdf, .doc, .docx</p>
      </div>
    </div>
  );
}

export default ResumeUploader;
