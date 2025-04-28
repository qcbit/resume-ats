import React from 'react';
import './KeywordMatch.css';

function KeywordMatch({ matchedKeywords, missingKeywords }) {
  return (
    <div className="keyword-match">
      <div className="keyword-section">
        <h4>
          <span className="keyword-icon matched">✓</span>
          Matched Keywords ({matchedKeywords.length})
        </h4>
        <div className="keyword-tags">
          {matchedKeywords.length > 0 ? (
            matchedKeywords.map((keyword, index) => (
              <span key={index} className="keyword-tag matched">
                {keyword}
              </span>
            ))
          ) : (
            <p className="no-keywords">No matched keywords found.</p>
          )}
        </div>
      </div>
      
      <div className="keyword-section">
        <h4>
          <span className="keyword-icon missing">✗</span>
          Missing Keywords ({missingKeywords.length})
        </h4>
        <div className="keyword-tags">
          {missingKeywords.length > 0 ? (
            missingKeywords.map((keyword, index) => (
              <span key={index} className="keyword-tag missing">
                {keyword}
              </span>
            ))
          ) : (
            <p className="no-keywords">No missing keywords found. Great job!</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default KeywordMatch;
