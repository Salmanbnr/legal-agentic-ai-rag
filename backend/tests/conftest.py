"""Shared test environment.

Values are set before test modules import the application so configuration
validation succeeds without a developer-local .env file.
"""

from __future__ import annotations

import os

TEST_ENVIRONMENT = {
    "APP_ENV": "test",
    "API_PREFIX": "/api/v1",
    "SUPABASE_URL": "https://test-project.supabase.co",
    "SUPABASE_PUBLISHABLE_KEY": "test-publishable-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "OPENAI_API_KEY": "test-openai-key",
    "OPENAI_MODEL": "gpt-4.1-mini-test",
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_INDEX_NAME": "legal-evidence-test",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "EMBEDDING_DEVICE": "cpu",
    "MODEL_CACHE_DIR": ".cache/huggingface-test",
    "TEMP_DIR": ".tmp-test",
    "MAX_UPLOAD_MB": "50",
    "MAX_PDF_PAGES": "1000",
    "AGENT_MAX_STEPS": "12",
    "AGENT_MAX_RETRIEVAL_RETRIES": "2",
    "AGENT_MAX_LLM_CALLS": "12",
    "FRONTEND_ORIGIN": "http://localhost:3000",
}

for key, value in TEST_ENVIRONMENT.items():
    os.environ.setdefault(key, value)
