"""Detects relationships between tables using statistical and semantic analysis with embeddings."""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from src.utils.llm_client import get_llm
from src.utils.embedding_service import get_embedding_service
from langchain_core.prompts import ChatPromptTemplate
import json


class RelationshipDetector:
    """Detects foreign key relationships between tables using embeddings and statistics."""

    def __init__(self, confidence_threshold: float = 0.8):
        """
        Args:
            confidence_threshold: Minimum confidence to accept a relationship
        """
        self.confidence_threshold = confidence_threshold
        self.llm = get_llm(temperature=0.0, model="gemini-2.0-flash")
        self.embedding_service = get_embedding_service()

    def detect_relationships(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Detect relationships between multiple datasets.

        Args:
            datasets: {dataset_id: DataFrame}
            metadata: {dataset_id: semantic_analysis_dict}

        Returns:
            List of relationship dicts with confidence scores
        """
        relationships = []
        dataset_ids = list(datasets.keys())

        print("\n" + "="*80)
        print("[RELATIONSHIP DETECTION]")
        print("="*80)

        # 1. Name-based matching (fuzzy)
        name_matches = self._detect_by_column_names(datasets, metadata)
        print(f"[STEP 1] Name matching: {len(name_matches)} potential relationships")

        # 2. Embedding-based matching (semantic similarity)
        embedding_matches = self._detect_by_embeddings(metadata)
        print(f"[STEP 2] Embedding similarity: {len(embedding_matches)} semantic matches")

        # 3. Combine and deduplicate matches
        all_matches = self._combine_matches(name_matches, embedding_matches)
        print(f"[STEP 3] Combined: {len(all_matches)} unique candidates")

        # 4. Statistical validation
        validated_matches = []
        for match in all_matches:
            stats = self._validate_statistically(
                datasets[match['from_dataset_id']],
                datasets[match['to_dataset_id']],
                match['from_column'],
                match['to_column']
            )

            # Combine confidence scores
            final_confidence = self._calculate_final_confidence(match, stats)

            if final_confidence >= self.confidence_threshold:
                match.update(stats)
                match['confidence'] = final_confidence
                validated_matches.append(match)

        print(f"[STEP 4] Statistical validation: {len(validated_matches)} high-confidence matches")

        # 5. LLM semantic validation (final check)
        if validated_matches:
            final_relationships = self._validate_semantically(validated_matches, metadata)
        else:
            final_relationships = []

        print(f"\n[OK] Found {len(final_relationships)} confirmed relationships")

        return final_relationships

    def _detect_by_column_names(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Detect potential relationships by column name similarity.

        Patterns:
        - Exact match: customer_id ↔ customer_id
        - Suffix match: customer_id ↔ cust_id
        - Fuzzy match: CustomerID ↔ customer_id
        """
        matches = []
        dataset_ids = list(datasets.keys())

        for i, from_id in enumerate(dataset_ids):
            from_df = datasets[from_id]

            for to_id in dataset_ids[i+1:]:
                to_df = datasets[to_id]

                # Compare all column pairs
                for from_col in from_df.columns:
                    for to_col in to_df.columns:
                        similarity = self._column_name_similarity(from_col, to_col)

                        if similarity >= 0.6:  # Potential match
                            matches.append({
                                'from_dataset_id': from_id,
                                'to_dataset_id': to_id,
                                'from_column': from_col,
                                'to_column': to_col,
                                'name_similarity': similarity
                            })

        return matches

    def _detect_by_embeddings(self, metadata: Dict[str, Dict]) -> List[Dict]:
        """
        Detect relationships using embedding similarity.
        More accurate than fuzzy string matching for semantic relationships.
        """
        matches = []

        # Collect all column embeddings
        column_data = []
        for dataset_id, meta in metadata.items():
            column_semantics = meta.get('column_semantics', {})
            for col_name, col_meta in column_semantics.items():
                embedding = col_meta.get('semantic_embedding')
                if embedding:
                    column_data.append({
                        'dataset_id': dataset_id,
                        'column': col_name,
                        'embedding': embedding,
                        'semantic_type': col_meta.get('semantic_type'),
                        'is_foreign_key': col_meta.get('is_foreign_key', False),
                        'business_meaning': col_meta.get('business_meaning', '')
                    })

        # Find similar columns across different datasets
        for i, col1 in enumerate(column_data):
            for col2 in column_data[i+1:]:
                # Skip if same dataset
                if col1['dataset_id'] == col2['dataset_id']:
                    continue

                # Calculate similarity
                similarity = self.embedding_service.calculate_similarity(
                    col1['embedding'],
                    col2['embedding']
                )

                # High similarity threshold for relationships (0.85)
                if similarity >= 0.85:
                    # Bonus if both are marked as keys
                    if 'key' in col1.get('semantic_type', '') or 'key' in col2.get('semantic_type', ''):
                        similarity += 0.05

                    matches.append({
                        'from_dataset_id': col1['dataset_id'],
                        'to_dataset_id': col2['dataset_id'],
                        'from_column': col1['column'],
                        'to_column': col2['column'],
                        'embedding_similarity': min(1.0, similarity)
                    })

        return matches

    def _combine_matches(self, name_matches: List[Dict], embedding_matches: List[Dict]) -> List[Dict]:
        """Combine and deduplicate matches from different detection methods."""
        combined = {}

        # Add name matches
        for match in name_matches:
            key = (
                match['from_dataset_id'],
                match['to_dataset_id'],
                match['from_column'],
                match['to_column']
            )
            combined[key] = match

        # Add or update with embedding matches
        for match in embedding_matches:
            key = (
                match['from_dataset_id'],
                match['to_dataset_id'],
                match['from_column'],
                match['to_column']
            )

            if key in combined:
                # Merge scores
                combined[key]['embedding_similarity'] = match.get('embedding_similarity', 0)
            else:
                combined[key] = match

        return list(combined.values())

    def _column_name_similarity(self, col1: str, col2: str) -> float:
        """Calculate similarity between column names."""
        # Normalize
        c1 = col1.lower().replace('_', '').replace('-', '')
        c2 = col2.lower().replace('_', '').replace('-', '')

        # Exact match
        if c1 == c2:
            return 1.0

        # Common ID patterns
        if 'id' in c1 and 'id' in c2:
            # Remove 'id' and compare base names
            base1 = c1.replace('id', '')
            base2 = c2.replace('id', '')
            if base1 and base2:
                return SequenceMatcher(None, base1, base2).ratio()

        # General fuzzy match
        return SequenceMatcher(None, c1, c2).ratio()

    def _validate_statistically(
        self,
        from_df: pd.DataFrame,
        to_df: pd.DataFrame,
        from_col: str,
        to_col: str
    ) -> Dict:
        """
        Validate relationship using statistical analysis.

        Returns:
            Dictionary with validation metrics
        """
        # Get column data
        from_values = set(from_df[from_col].dropna().astype(str).unique())
        to_values = set(to_df[to_col].dropna().astype(str).unique())

        if len(from_values) == 0 or len(to_values) == 0:
            return {'match_percentage': 0.0, 'relationship_type': 'unknown'}

        # Calculate overlap
        intersection = from_values & to_values
        match_pct = len(intersection) / len(from_values) * 100 if len(from_values) > 0 else 0

        # Determine relationship type
        from_is_unique = len(from_values) == len(from_df)
        to_is_unique = len(to_values) == len(to_df)

        if from_is_unique and to_is_unique:
            rel_type = 'one-to-one'
        elif to_is_unique:
            rel_type = 'many-to-one'
        elif from_is_unique:
            rel_type = 'one-to-many'
        else:
            rel_type = 'many-to-many'

        # Recommend join strategy
        if match_pct >= 95:
            join_strategy = 'inner'
        elif match_pct >= 70:
            join_strategy = 'left'
        else:
            join_strategy = 'left'

        return {
            'match_percentage': match_pct,
            'relationship_type': rel_type,
            'join_strategy': join_strategy,
            'from_unique_count': len(from_values),
            'to_unique_count': len(to_values),
            'intersection_count': len(intersection)
        }

    def _calculate_final_confidence(self, match: Dict, stats: Dict) -> float:
        """
        Calculate final confidence score combining all signals.

        Args:
            match: Match dictionary with similarity scores
            stats: Statistical validation results

        Returns:
            Confidence score between 0 and 1
        """
        confidence_components = []

        # Name similarity (weight: 0.2)
        if 'name_similarity' in match:
            confidence_components.append(match['name_similarity'] * 0.2)

        # Embedding similarity (weight: 0.4 - most important)
        if 'embedding_similarity' in match:
            confidence_components.append(match['embedding_similarity'] * 0.4)

        # Statistical match percentage (weight: 0.4)
        if 'match_percentage' in stats:
            confidence_components.append((stats['match_percentage'] / 100) * 0.4)

        # Calculate weighted average
        if confidence_components:
            confidence = sum(confidence_components)
        else:
            confidence = 0.0

        # Bonus for good relationship types
        if stats.get('relationship_type') in ['one-to-many', 'many-to-one']:
            confidence = min(1.0, confidence + 0.05)

        return confidence

    def _validate_semantically(
        self,
        candidate_relationships: List[Dict],
        metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Use LLM to validate relationships semantically.
        Filter out false positives based on business context.
        """
        if not candidate_relationships:
            return []

        # Prepare context for LLM
        context = []
        for rel in candidate_relationships:
            from_meta = metadata.get(rel['from_dataset_id'], {})
            to_meta = metadata.get(rel['to_dataset_id'], {})

            # Get column semantic info
            from_col_meta = from_meta.get('column_semantics', {}).get(rel['from_column'], {})
            to_col_meta = to_meta.get('column_semantics', {}).get(rel['to_column'], {})

            context.append({
                'index': len(context),
                'from_table': from_meta.get('table_name', 'unknown'),
                'from_domain': from_meta.get('domain', 'Unknown'),
                'from_column': rel['from_column'],
                'from_meaning': from_col_meta.get('business_meaning', ''),
                'to_table': to_meta.get('table_name', 'unknown'),
                'to_domain': to_meta.get('domain', 'Unknown'),
                'to_column': rel['to_column'],
                'to_meaning': to_col_meta.get('business_meaning', ''),
                'confidence': rel['confidence'],
                'match_pct': rel.get('match_percentage', 0)
            })

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are validating database relationships.

Given candidate relationships detected by name/statistical/embedding matching, validate which ones make semantic sense.

Return a JSON array of indices (0-based) for relationships that are VALID.

Rules:
- Accept relationships that make business sense
- customer_id should relate to customer tables
- product_id should relate to product tables
- order_id should relate to order/transaction tables
- Reject relationships between completely unrelated domains (e.g., HR employee_id ↔ Product inventory_id)
- Consider the business context and domain
- High confidence (>0.9) relationships are likely correct

Return ONLY a JSON array of valid indices, e.g., [0, 2, 5]"""),
            ("user", """Candidate relationships:

{relationships}

Return valid relationship indices:""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "relationships": json.dumps(context, indent=2)
        })

        try:
            valid_indices = json.loads(result.content.strip())
            validated = [candidate_relationships[i] for i in valid_indices if i < len(candidate_relationships)]

            print(f"[STEP 5] Semantic validation: {len(validated)}/{len(candidate_relationships)} relationships confirmed")
            return validated

        except json.JSONDecodeError:
            print(f"[WARN] LLM semantic validation failed, accepting all statistical matches")
            return candidate_relationships