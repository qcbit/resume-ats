import React from 'react';
import './CategoryScore.css';

function CategoryScore({ title, score, relevance }) {
  // Determine color based on score
  let scoreColor = '#dc3545'; // red for low scores
  if (score >= 70) {
    scoreColor = '#28a745'; // green for high scores
  } else if (score >= 40) {
    scoreColor = '#ffc107'; // yellow for medium scores
  }
  
  // Determine relevance badge color
  let relevanceColor = '#6c757d'; // gray for low relevance
  if (relevance === 'high') {
    relevanceColor = '#28a745'; // green for high relevance
  } else if (relevance === 'medium') {
    relevanceColor = '#ffc107'; // yellow for medium relevance
  }
  
  return (
    <div className="category-score">
      <div className="category-header">
        <h4>{title}</h4>
        <span 
          className="relevance-badge"
          style={{ backgroundColor: relevanceColor }}
        >
          {relevance} relevance
        </span>
      </div>
      <div className="score-bar-container">
        <div 
          className="score-bar" 
          style={{ 
            width: `${score}%`, 
            backgroundColor: scoreColor 
          }}
        ></div>
      </div>
      <div className="score-value">{score}%</div>
    </div>
  );
}

export default CategoryScore;
