"""Service for generating embeddings using Google Gemini."""

import google.generativeai as genai
from typing import List, Optional
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbeddingService:
    """Handles embeddings using Google's Gemini embedding model."""

    def __init__(self):
        """Initialize Gemini embedding service."""
        # Configure with existing API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            # Also check for GOOGLE_API_KEY for compatibility
            api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=api_key)
        self.model = 'models/embedding-001'
        self.dimension = 768  # Gemini embedding dimension

        print(f"[EMBEDDINGS] Using Gemini {self.model} with {self.dimension} dimensions")

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        Cached for efficiency.

        Args:
            text: Text to embed

        Returns:
            List of 768 floats
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"[ERROR] Gemini embedding failed: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimension

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def create_dataset_description_embedding(self, metadata: dict) -> List[float]:
        """
        Create embedding for dataset description.
        Combines domain, entities, description into single text.

        Args:
            metadata: Dataset metadata dictionary

        Returns:
            Embedding vector
        """
        text_parts = [
            f"Domain: {metadata.get('domain', 'Unknown')}",
            f"Department: {metadata.get('department', 'Unknown')}",
            f"Description: {metadata.get('description', '')}",
            f"Entities: {', '.join(metadata.get('entities', []))}",
        ]

        combined_text = " | ".join(text_parts)
        return self.get_embedding(combined_text)

    def create_schema_embedding(self, column_names: List[str], column_types: List[str]) -> List[float]:
        """
        Create embedding for table schema.
        Useful for finding structurally similar tables.

        Args:
            column_names: List of column names
            column_types: List of column data types

        Returns:
            Embedding vector
        """
        schema_text = " ".join([
            f"{col}:{dtype}" for col, dtype in zip(column_names, column_types)
        ])
        return self.get_embedding(schema_text)

    def create_column_semantic_embedding(
        self,
        column_name: str,
        business_meaning: str,
        semantic_type: str
    ) -> List[float]:
        """
        Create embedding for column semantic meaning.
        Useful for matching columns across tables.

        Args:
            column_name: Name of the column
            business_meaning: Business description of the column
            semantic_type: Type (dimension, measure, key, etc.)

        Returns:
            Embedding vector
        """
        text = f"Column: {column_name} | Type: {semantic_type} | Meaning: {business_meaning}"
        return self.get_embedding(text)

    def create_kpi_embedding(self, kpi_name: str, description: str, domain: str) -> List[float]:
        """
        Create embedding for KPI definition.

        Args:
            kpi_name: Name of the KPI
            description: Description of what the KPI measures
            domain: Business domain

        Returns:
            Embedding vector
        """
        text = f"KPI: {kpi_name} | Domain: {domain} | Description: {description}"
        return self.get_embedding(text)

    def create_insight_embedding(self, insight_text: str, domain: str) -> List[float]:
        """
        Create embedding for an insight pattern.

        Args:
            insight_text: The insight description
            domain: Business domain

        Returns:
            Embedding vector
        """
        text = f"Domain: {domain} | Insight: {insight_text}"
        return self.get_embedding(text)

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        import numpy as np

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Ensure value is between 0 and 1
        return max(0.0, min(1.0, similarity))


# Singleton instance
_embedding_service = None


def get_embedding_service() -> GeminiEmbeddingService:
    """Get singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = GeminiEmbeddingService()
    return _embedding_service