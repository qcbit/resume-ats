import React from 'react';
import './ScoreGauge.css';

function ScoreGauge({ score }) {
  // Determine color based on score
  let scoreColor = '#dc3545'; // red for low scores
  if (score >= 70) {
    scoreColor = '#28a745'; // green for high scores
  } else if (score >= 40) {
    scoreColor = '#ffc107'; // yellow for medium scores
  }
  
  // Calculate rotation for gauge needle
  const rotation = (score / 100) * 180;
  
  return (
    <div className="score-gauge">
      <div className="gauge">
        <div className="gauge-body">
          <div className="gauge-fill" style={{ transform: `rotate(${rotation}deg)` }}></div>
          <div className="gauge-cover" style={{ color: scoreColor }}>{score}%</div>
        </div>
        <div className="gauge-labels">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
      <div className="score-interpretation">
        {score >= 80 ? (
          <p className="excellent">Excellent match! Your resume is well-aligned with this job.</p>
        ) : score >= 60 ? (
          <p className="good">Good match. With some adjustments, your resume could be a great fit.</p>
        ) : score >= 40 ? (
          <p className="fair">Fair match. Consider the suggestions below to improve your chances.</p>
        ) : (
          <p className="poor">Low match. Your resume may need significant revisions for this position.</p>
        )}
      </div>
    </div>
  );
}

export default ScoreGauge;
