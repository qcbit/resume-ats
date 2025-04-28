import React from 'react';
import './Footer.css';

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <p>Resume Job Matcher &copy; {new Date().getFullYear()}</p>
        <p className="disclaimer">
          This tool provides an automated analysis and should be used as a guide only. 
          The actual hiring process involves many factors beyond keyword matching.
        </p>
      </div>
    </footer>
  );
}

export default Footer;
