import React from 'react';
import './JobDetails.css';

function JobDetails({ jobTitle, jobDescription, onJobTitleChange, onJobDescriptionChange }) {
  return (
    <div className="job-details">
      <h2>Job Details</h2>
      
      <div className="form-group">
        <label htmlFor="job-title">Job Title (Optional):</label>
        <input
          type="text"
          id="job-title"
          value={jobTitle}
          onChange={(e) => onJobTitleChange(e.target.value)}
          placeholder="e.g. Frontend Developer"
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="job-description">
          Job Description: <span className="required">*</span>
        </label>
        <textarea
          id="job-description"
          value={jobDescription}
          onChange={(e) => onJobDescriptionChange(e.target.value)}
          placeholder="Paste the job description here..."
          rows="10"
          required
        ></textarea>
      </div>
    </div>
  );
}

export default JobDetails;
