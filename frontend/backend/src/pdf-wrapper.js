import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Use require to import pdf-parse to avoid ES module issues
const pdfParse = require('pdf-parse');

export default pdfParse;

