import React from 'react';
import './SuggestionsList.css';

function SuggestionsList({ suggestions }) {
  if (!suggestions || suggestions.length === 0) {
    return (
      <div className="no-suggestions">
        <p>No specific suggestions at this time. Your resume appears to be well-aligned with the job description.</p>
      </div>
    );
  }
  
  return (
    <ul className="suggestions-list">
      {suggestions.map((suggestion, index) => (
        <li key={index} className="suggestion-item">
          <div className="suggestion-icon">💡</div>
          <div className="suggestion-text">{suggestion}</div>
        </li>
      ))}
    </ul>
  );
}

export default SuggestionsList;
