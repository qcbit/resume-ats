import React from 'react';
import './AnalysisResults.css';
import ScoreGauge from './ScoreGauge';
import CategoryScore from './CategoryScore';
import SuggestionsList from './SuggestionsList';
import KeywordMatch from './KeywordMatch';

function AnalysisResults({ results }) {
  if (!results) return null;
  
  // Check if results has the expected structure or use defaults
  const overallScore = results.score || 0;
  const matchedKeywords = results.resumeKeywords || [];
  const missingKeywords = results.jdKeywords?.filter(kw => 
    !results.resumeKeywords?.includes(kw)) || [];
  
  // Create categories object if it doesn't exist
  const categories = results.categories || {
    skills: { score: 50, relevance: 25 },
    experience: { score: 50, relevance: 25 },
    education: { score: 50, relevance: 25 },
    achievements: { score: 50, relevance: 25 }
  };
  
  // Use feedback as suggestions or create default
  const suggestions = results.feedback ? [results.feedback] : [];

  return (
    <div className="analysis-results">
      <h2>Analysis Results</h2>
      
      <div className="overall-score">
        <h3>Overall Match</h3>
        <ScoreGauge score={overallScore} />
      </div>
      
      <div className="category-scores">
        <h3>Category Breakdown</h3>
        <div className="categories-grid">
          <CategoryScore 
            title="Skills" 
            score={categories.skills.score} 
            relevance={categories.skills.relevance} 
          />
          <CategoryScore 
            title="Experience" 
            score={categories.experience.score} 
            relevance={categories.experience.relevance} 
          />
          <CategoryScore 
            title="Education" 
            score={categories.education.score} 
            relevance={categories.education.relevance} 
          />
          <CategoryScore 
            title="Achievements" 
            score={categories.achievements.score} 
            relevance={categories.achievements.relevance} 
          />
        </div>
      </div>
      
      <div className="keyword-analysis">
        <h3>Keyword Analysis</h3>
        <KeywordMatch 
          matchedKeywords={matchedKeywords} 
          missingKeywords={missingKeywords} 
        />
      </div>
      
      <div className="improvement-suggestions">
        <h3>Improvement Suggestions</h3>
        <SuggestionsList suggestions={suggestions} />
      </div>
    </div>
  );
}

export default AnalysisResults;
