import React from 'react';
import './AnalysisResults.css';
import ScoreGauge from './ScoreGauge';
import CategoryScore from './CategoryScore';
import SuggestionsList from './SuggestionsList';
import KeywordMatch from './KeywordMatch';

function AnalysisResults({ results }) {
  if (!results) return null;

  // results.jdKeywords: Array of keywords extracted from the job description
  // results.resumeText: The full text content of the resume

  const jdKeywordsList = results.jdKeywords || [];
  const resumeText = results.resumeText || ""; // Get the full resume text

  // --- Correct Calculation ---
  // Matched: Keywords from jdKeywordsList found in resumeText
  const matchedKeywords = jdKeywordsList.filter(jdKw =>
    resumeText.toLowerCase().includes(jdKw.toLowerCase()) // Case-insensitive check in resume text
  );

  // Missing: Keywords from jdKeywordsList NOT found in resumeText
  const missingKeywords = jdKeywordsList.filter(jdKw =>
    !resumeText.toLowerCase().includes(jdKw.toLowerCase()) // Case-insensitive check in resume text
  );
  // --- End Correct Calculation ---

  // Check if results has the expected structure or use defaults for other parts
  const overallScore = results.score || 0;
  const categories = results.categories || {
    skills: { score: 0, relevance: 25 },
    experience: { score: 0, relevance: 25 },
    education: { score: 0, relevance: 25 },
    achievements: { score: 0, relevance: 25 }
  };
  const suggestions = results.feedback ? [results.feedback] : ["Review missing keywords and tailor your resume."]; // Default suggestion

  return (
    <div className="analysis-results">
      <h2>Analysis Results</h2>

      <div className="overall-score">
        <h3>Overall Match</h3>
        <ScoreGauge score={overallScore} />
      </div>

      <div className="category-scores">
        {/* ... CategoryScore components ... */}
         <h3>Category Breakdown</h3>
        <div className="categories-grid">
          <CategoryScore
            title="Skills"
            score={categories.skills?.score || 0} // Add safe access
            relevance={categories.skills?.relevance || 0}
          />
          <CategoryScore
            title="Experience"
            score={categories.experience?.score || 0}
            relevance={categories.experience?.relevance || 0}
          />
          <CategoryScore
            title="Education"
            score={categories.education?.score || 0}
            relevance={categories.education?.relevance || 0}
          />
          <CategoryScore
            title="Achievements"
            score={categories.achievements?.score || 0}
            relevance={categories.achievements?.relevance || 0}
          />
        </div>
      </div>

      <div className="keyword-analysis">
        <h3>Keyword Analysis</h3>
        {/* Pass the correctly calculated lists */}
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
