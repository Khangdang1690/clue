-- Initialize database with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better performance
-- These will be created after tables are created by SQLAlchemy

-- Grant permissions (optional, for production)
GRANT ALL PRIVILEGES ON DATABASE ai_analyst TO postgres;

-- Set optimization parameters for pgvector
SET maintenance_work_mem = '512MB';
SET max_parallel_workers_per_gather = 4;
SET max_parallel_workers = 8;
SET max_parallel_maintenance_workers = 4;

-- Log successful initialization
DO $$
BEGIN
  RAISE NOTICE 'Database ai_analyst initialized with pgvector extension';
END $$;