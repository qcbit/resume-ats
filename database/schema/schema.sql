-- Example Schema for Resume ATS

-- Job Descriptions Table
CREATE TABLE job_descriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, -- User who submitted this JD for analysis
    title VARCHAR(255),
    company VARCHAR(255),
    description_text TEXT NOT NULL,
    description_vector VECTOR(768), -- Example dimension, adjust based on embedding model
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Resumes Table
CREATE TABLE resumes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL, -- Path to the stored resume file (e.g., S3 URL or local path)
    resume_text TEXT, -- Extracted text content of the resume
    version_name VARCHAR(100), -- e.g., "Tailored for Google", "General Tech"
    resume_vector VECTOR(768), -- Example dimension, adjust based on embedding model
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analysis History Table
CREATE TABLE analysis_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    resume_id INTEGER NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    job_description_id INTEGER NOT NULL REFERENCES job_descriptions(id) ON DELETE CASCADE,
    analysis_results JSONB, -- Store the full JSON analysis (score, categories, feedback, keywords)
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_analysis UNIQUE (user_id, resume_id, job_description_id, analysis_date) -- Prevent duplicate entries for the exact same analysis at the same time
);

-- Optional: Indexes for faster searching
-- Index on description vectors for similarity search
CREATE INDEX IF NOT EXISTS idx_job_description_vector ON job_descriptions USING ivfflat (description_vector vector_l2_ops) WITH (lists = 100);
-- Index on resume vectors for similarity search
CREATE INDEX IF NOT EXISTS idx_resume_vector ON resumes USING ivfflat (resume_vector vector_l2_ops) WITH (lists = 100);

-- Index for quickly finding results by job or resume
CREATE INDEX IF NOT EXISTS idx_results_job_id ON analysis_results (job_id);
CREATE INDEX IF NOT EXISTS idx_results_resume_id ON analysis_results (resume_id);

-- Note: The vector dimension (e.g., 768) depends on the embedding model you choose.
-- Note: Index types (like ivfflat) and parameters (like lists) might need tuning based on data size and performance needs.

-- Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255), -- For email/password login
    social_provider VARCHAR(50), -- e.g., 'google', 'github'
    social_id VARCHAR(255), -- User ID from the social provider
    profile_picture_url TEXT,
    bio TEXT,
    contact_information TEXT, -- Could be a JSONB field for more structured data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_social_login UNIQUE (social_provider, social_id)
);

-- Cover Letters Table
CREATE TABLE cover_letters (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    cover_letter_text TEXT,
    version_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Job Applications Table
CREATE TABLE job_applications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_title VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    application_date DATE NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'applied', 'interview', 'offer', 'rejected'
    response_date DATE,
    source TEXT, -- LinkedIn URL, job board link, email, etc.
    resume_id INTEGER REFERENCES resumes(id),
    cover_letter_id INTEGER REFERENCES cover_letters(id),
    analysis_id INTEGER REFERENCES analysis_history(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);



-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_resumes_user_id ON resumes(user_id);
CREATE INDEX idx_job_descriptions_user_id ON job_descriptions(user_id);
CREATE INDEX idx_analysis_history_user_id ON analysis_history(user_id);
CREATE INDEX idx_analysis_history_resume_id ON analysis_history(resume_id);
CREATE INDEX idx_analysis_history_job_description_id ON analysis_history(job_description_id);

-- Optional: Trigger to update 'updated_at' timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_resumes_updated_at
BEFORE UPDATE ON resumes
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();