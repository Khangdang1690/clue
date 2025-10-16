"""ChromaDB manager for business context storage."""

import json
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


class ChromaDBManager:
    """Manages ChromaDB operations for business context storage."""

    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Initialize ChromaDB manager.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Initialize collections
        self.business_context_collection = self._get_or_create_collection(
            "business_context",
            metadata={"description": "Stores business context information"}
        )

        self.challenges_collection = self._get_or_create_collection(
            "challenges",
            metadata={"description": "Stores identified challenges"}
        )

    def _get_or_create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Get or create a collection."""
        try:
            return self.client.get_collection(name=name)
        except Exception:
            return self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )

    def store_business_context(self, context_dict: Dict, context_id: str = "main_context"):
        """
        Store business context in ChromaDB.

        Args:
            context_dict: Dictionary containing business context
            context_id: Unique identifier for this context
        """
        # Store as document with metadata
        self.business_context_collection.upsert(
            ids=[context_id],
            documents=[json.dumps(context_dict, indent=2)],
            metadatas=[{
                "type": "business_context",
                "company_name": context_dict.get("company_name", ""),
            }]
        )

    def get_business_context(self, context_id: str = "main_context") -> Optional[Dict]:
        """
        Retrieve business context from ChromaDB.

        Args:
            context_id: Unique identifier for the context

        Returns:
            Dictionary containing business context or None if not found
        """
        try:
            results = self.business_context_collection.get(ids=[context_id])
            if results and results['documents']:
                return json.loads(results['documents'][0])
            return None
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return None

    def store_challenge(self, challenge_dict: Dict, challenge_id: str):
        """
        Store a challenge in ChromaDB.

        Args:
            challenge_dict: Dictionary containing challenge data
            challenge_id: Unique identifier for this challenge
        """
        # Convert department list to comma-separated string for metadata
        dept = challenge_dict.get("department", [])
        if isinstance(dept, list):
            dept_str = ", ".join(dept)
        else:
            dept_str = str(dept)

        self.challenges_collection.upsert(
            ids=[challenge_id],
            documents=[json.dumps(challenge_dict, indent=2)],
            metadatas=[{
                "type": "challenge",
                "department": dept_str,
                "priority_level": challenge_dict.get("priority_level", ""),
                "priority_score": challenge_dict.get("priority_score", 0),
            }]
        )

    def get_all_challenges(self) -> List[Dict]:
        """
        Retrieve all challenges from ChromaDB.

        Returns:
            List of challenge dictionaries
        """
        try:
            results = self.challenges_collection.get()
            if results and results['documents']:
                return [json.loads(doc) for doc in results['documents']]
            return []
        except Exception as e:
            print(f"Error retrieving challenges: {e}")
            return []

    def query_context(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Query business context using semantic search.

        Args:
            query_text: Text to query
            n_results: Number of results to return

        Returns:
            List of relevant context documents
        """
        results = self.business_context_collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        if results and results['documents']:
            return [json.loads(doc) for doc in results['documents'][0]]
        return []

    def reset_database(self):
        """Reset the entire database (use with caution)."""
        self.client.reset()
        # Reinitialize collections
        self.__init__(str(self.persist_directory))
