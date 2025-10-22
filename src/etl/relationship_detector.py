"""Detects relationships between tables using LLM interpretation and statistical analysis."""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from src.utils.llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
import json


class RelationshipDetector:
    """Detects foreign key relationships between tables using LLM interpretation and statistics."""

    def __init__(self, confidence_threshold: float = 0.8):
        """
        Args:
            confidence_threshold: Minimum confidence to accept a relationship
        """
        self.confidence_threshold = confidence_threshold
        self.llm = get_llm(temperature=0.0, model="gemini-2.5-flash")

    def detect_relationships(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict],
        existing_datasets: Optional[Dict[str, pd.DataFrame]] = None,
        existing_metadata: Optional[Dict[str, Dict]] = None
    ) -> List[Dict]:
        """
        Detect relationships between multiple datasets.

        Args:
            datasets: {dataset_id: DataFrame} - New datasets being processed
            metadata: {dataset_id: semantic_analysis_dict} - New datasets metadata
            existing_datasets: Optional {dataset_id: DataFrame} - Existing datasets in DB
            existing_metadata: Optional {dataset_id: metadata} - Existing datasets metadata

        Returns:
            List of relationship dicts with confidence scores
        """
        # Combine new and existing datasets for relationship detection
        all_datasets = dict(datasets)
        all_metadata = dict(metadata)

        if existing_datasets:
            all_datasets.update(existing_datasets)
            all_metadata.update(existing_metadata or {})
            print(f"\n[INFO] Including {len(existing_datasets)} existing datasets for relationship detection")

        relationships = []
        dataset_ids = list(all_datasets.keys())

        print("\n" + "="*80)
        print("[RELATIONSHIP DETECTION]")
        print(f"Total datasets: {len(all_datasets)} ({len(datasets)} new + {len(existing_datasets or {})} existing)")
        print("="*80)

        # 1. Name-based matching (fuzzy)
        name_matches = self._detect_by_column_names(all_datasets, all_metadata)
        print(f"[STEP 1] Name matching: {len(name_matches)} potential relationships")

        # 2. LLM interpretation (semantic reasoning)
        llm_matches = self._detect_by_llm_interpretation(all_datasets, all_metadata, name_matches)
        print(f"[STEP 2] LLM interpretation: {len(llm_matches)} semantic matches")

        # 3. Combine and deduplicate matches
        all_matches = self._combine_matches(name_matches, llm_matches)
        print(f"[STEP 3] Combined: {len(all_matches)} unique candidates")

        # 4. Statistical validation
        validated_matches = []
        for match in all_matches:
            stats = self._validate_statistically(
                all_datasets[match['from_dataset_id']],
                all_datasets[match['to_dataset_id']],
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
            final_relationships = self._validate_semantically(validated_matches, all_metadata)
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

    def _detect_by_llm_interpretation(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict],
        name_matches: List[Dict]
    ) -> List[Dict]:
        """
        Use LLM to interpret and score relationship likelihood.
        Replaces embedding-based detection with direct LLM reasoning.

        This is more aligned with LLM-driven decision making:
        - LLM sees actual data samples
        - Can understand business logic
        - Provides explainable reasoning
        - No embeddings needed
        """
        matches = []

        # Build candidate list: name matches + all key-to-key pairs
        candidates = list(name_matches)

        # Add all key-to-key relationships (even if names don't match)
        # This allows LLM to detect semantic relationships like client_id ↔ user_id ↔ customer_code
        dataset_ids = list(datasets.keys())
        seen_pairs = set()

        # Track name match pairs to avoid duplicates
        for match in name_matches:
            key = (match['from_dataset_id'], match['to_dataset_id'],
                   match['from_column'], match['to_column'])
            seen_pairs.add(key)

        # Add key-to-key candidates
        for i, from_id in enumerate(dataset_ids):
            from_meta = metadata.get(from_id, {})
            from_col_semantics = from_meta.get('column_semantics', {})

            for to_id in dataset_ids[i+1:]:
                to_meta = metadata.get(to_id, {})
                to_col_semantics = to_meta.get('column_semantics', {})

                # Find all key columns in both datasets
                for from_col, from_col_meta in from_col_semantics.items():
                    if from_col_meta.get('semantic_type') == 'key':
                        for to_col, to_col_meta in to_col_semantics.items():
                            if to_col_meta.get('semantic_type') == 'key':
                                key = (from_id, to_id, from_col, to_col)
                                if key not in seen_pairs:
                                    candidates.append({
                                        'from_dataset_id': from_id,
                                        'to_dataset_id': to_id,
                                        'from_column': from_col,
                                        'to_column': to_col,
                                        'name_similarity': 0.0  # No name match
                                    })
                                    seen_pairs.add(key)

        print(f"    [LLM] Evaluating {len(candidates)} candidates ({len(name_matches)} from name matching, {len(candidates) - len(name_matches)} key-to-key pairs)")

        # PRE-FILTER: Skip obviously unrelated pairs to reduce LLM calls
        filtered_candidates = self._prefilter_candidates(candidates, datasets, metadata)
        if len(filtered_candidates) < len(candidates):
            print(f"    [OPTIMIZATION] Pre-filtered from {len(candidates)} to {len(filtered_candidates)} candidates")

        # OPTIMIZATION: Batch evaluate with LLM instead of sequential calls
        if len(filtered_candidates) <= 10:
            # Small batch: Use batch evaluation (1 LLM call)
            return self._batch_evaluate_llm(filtered_candidates, datasets, metadata)
        else:
            # Large batch: Split into chunks of 10 (reduces total calls significantly)
            matches = []
            for i in range(0, len(filtered_candidates), 10):
                chunk = filtered_candidates[i:i+10]
                print(f"    [LLM] Batch {i//10 + 1}: Evaluating {len(chunk)} candidates...")
                matches.extend(self._batch_evaluate_llm(chunk, datasets, metadata))
            return matches

    def _prefilter_candidates(
        self,
        candidates: List[Dict],
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Pre-filter candidates to skip obviously unrelated pairs.
        Reduces LLM API calls by eliminating incompatible columns.

        Filters based on:
        - Data type compatibility
        - Semantic type compatibility
        - Column completeness (not all null)
        """
        filtered = []

        for candidate in candidates:
            from_dataset_id = candidate['from_dataset_id']
            to_dataset_id = candidate['to_dataset_id']
            from_col = candidate['from_column']
            to_col = candidate['to_column']

            # Get dataframes
            from_df = datasets[from_dataset_id]
            to_df = datasets[to_dataset_id]

            # Skip if either column is mostly null (>90% null)
            from_null_pct = from_df[from_col].isna().sum() / len(from_df)
            to_null_pct = to_df[to_col].isna().sum() / len(to_df)
            if from_null_pct > 0.9 or to_null_pct > 0.9:
                continue

            # Get data types
            from_dtype = from_df[from_col].dtype
            to_dtype = to_df[to_col].dtype

            # Check data type compatibility
            from_is_numeric = from_dtype in ['int64', 'int32', 'float64', 'float32']
            to_is_numeric = to_dtype in ['int64', 'int32', 'float64', 'float32']

            # Skip if one is numeric and other is string (incompatible for FK)
            if from_is_numeric != to_is_numeric:
                continue

            # Get semantic types
            from_meta = metadata.get(from_dataset_id, {})
            to_meta = metadata.get(to_dataset_id, {})
            from_col_meta = from_meta.get('column_semantics', {}).get(from_col, {})
            to_col_meta = to_meta.get('column_semantics', {}).get(to_col, {})

            from_semantic_type = from_col_meta.get('semantic_type', '')
            to_semantic_type = to_col_meta.get('semantic_type', '')

            # Skip if one is a measure and other is a key (very unlikely relationship)
            if (from_semantic_type == 'measure' and to_semantic_type == 'key') or \
               (from_semantic_type == 'key' and to_semantic_type == 'measure'):
                continue

            # Skip if both are measures (measures don't relate to each other)
            if from_semantic_type == 'measure' and to_semantic_type == 'measure':
                continue

            # Passed all filters
            filtered.append(candidate)

        return filtered

    def _batch_evaluate_llm(
        self,
        candidates: List[Dict],
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Evaluate multiple candidates in a single LLM call.
        Significantly reduces API calls from N to 1 (or N/10 for large batches).

        Args:
            candidates: List of candidate relationships to evaluate
            datasets: Dictionary of dataframes
            metadata: Dictionary of metadata

        Returns:
            List of matches that scored >= 0.7
        """
        if not candidates:
            return []

        # Build batch context
        batch_context = []
        for i, candidate in enumerate(candidates):
            from_dataset_id = candidate['from_dataset_id']
            to_dataset_id = candidate['to_dataset_id']
            from_col = candidate['from_column']
            to_col = candidate['to_column']

            # Get metadata
            from_meta = metadata.get(from_dataset_id, {})
            to_meta = metadata.get(to_dataset_id, {})
            from_col_meta = from_meta.get('column_semantics', {}).get(from_col, {})
            to_col_meta = to_meta.get('column_semantics', {}).get(to_col, {})

            # Get sample data
            from_df = datasets[from_dataset_id]
            to_df = datasets[to_dataset_id]
            from_samples = from_df[from_col].dropna().head(5).tolist()
            to_samples = to_df[to_col].dropna().head(5).tolist()

            batch_context.append({
                'index': i,
                'from_table': from_meta.get('table_name', 'unknown'),
                'from_domain': from_meta.get('domain', 'Unknown'),
                'from_column': from_col,
                'from_type': str(from_df[from_col].dtype),
                'from_semantic_type': from_col_meta.get('semantic_type', 'unknown'),
                'from_meaning': from_col_meta.get('business_meaning', 'N/A'),
                'from_samples': str(from_samples[:3]),
                'to_table': to_meta.get('table_name', 'unknown'),
                'to_domain': to_meta.get('domain', 'Unknown'),
                'to_column': to_col,
                'to_type': str(to_df[to_col].dtype),
                'to_semantic_type': to_col_meta.get('semantic_type', 'unknown'),
                'to_meaning': to_col_meta.get('business_meaning', 'N/A'),
                'to_samples': str(to_samples[:3])
            })

        # Batch prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database relationship expert. Evaluate multiple potential foreign key relationships at once.

For each candidate, score 0.0 to 1.0 based on:
- Column name similarity
- Data type compatibility
- Sample value patterns
- Business meaning alignment
- Semantic types (key, dimension, measure)

Return ONLY a JSON array of objects, one per candidate:
[
  {{"index": 0, "score": 0.95, "reasoning": "brief explanation"}},
  {{"index": 1, "score": 0.3, "reasoning": "brief explanation"}},
  ...
]

Be strict - only score >= 0.7 if you're confident it's a real relationship."""),
            ("user", """Evaluate these {count} potential relationships:

{candidates}

Return JSON array of scores:""")
        ])

        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "count": len(candidates),
                "candidates": json.dumps(batch_context, indent=2)
            })

            # Parse LLM response
            response_text = result.content.strip()
            # Remove markdown code blocks if present
            if '```' in response_text:
                parts = response_text.split('```')
                response_text = parts[1] if len(parts) > 1 else parts[0]
                if response_text.startswith('json'):
                    response_text = response_text[4:]

            llm_results = json.loads(response_text.strip())

            # Build matches from results
            matches = []
            for result_item in llm_results:
                idx = result_item.get('index')
                score = float(result_item.get('score', 0.0))
                reasoning = result_item.get('reasoning', '')

                if idx is not None and idx < len(candidates) and score >= 0.7:
                    candidate = candidates[idx]
                    matches.append({
                        'from_dataset_id': candidate['from_dataset_id'],
                        'to_dataset_id': candidate['to_dataset_id'],
                        'from_column': candidate['from_column'],
                        'to_column': candidate['to_column'],
                        'llm_similarity': min(1.0, score),
                        'llm_reasoning': reasoning
                    })
                    print(f"    [LLM] {candidate['from_column']} -> {candidate['to_column']}: score={score:.2f}")

            return matches

        except Exception as e:
            print(f"    [ERROR] Batch LLM evaluation failed: {e}")
            print(f"    [FALLBACK] Using sequential evaluation for {len(candidates)} candidates...")
            # Fallback to sequential evaluation if batch fails
            return self._fallback_sequential_evaluation(candidates, datasets, metadata)

    def _fallback_sequential_evaluation(
        self,
        candidates: List[Dict],
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Fallback to sequential evaluation if batch evaluation fails.
        This is the old code path kept for reliability.
        """
        matches = []

        for candidate in candidates:
            from_dataset_id = candidate['from_dataset_id']
            to_dataset_id = candidate['to_dataset_id']
            from_col = candidate['from_column']
            to_col = candidate['to_column']

            # Get metadata
            from_meta = metadata.get(from_dataset_id, {})
            to_meta = metadata.get(to_dataset_id, {})
            from_col_meta = from_meta.get('column_semantics', {}).get(from_col, {})
            to_col_meta = to_meta.get('column_semantics', {}).get(to_col, {})

            # Get sample data
            from_df = datasets[from_dataset_id]
            to_df = datasets[to_dataset_id]
            from_samples = from_df[from_col].dropna().head(5).tolist()
            to_samples = to_df[to_col].dropna().head(5).tolist()

            # Ask LLM to score the relationship
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a database relationship expert. Analyze if two columns are likely a foreign key relationship.

Score 0.0 to 1.0 based on:
- Column name similarity
- Data type compatibility
- Sample value patterns
- Business meaning alignment
- Semantic types (key, dimension, measure)

Return ONLY a JSON object: {{"score": 0.95, "reasoning": "brief explanation"}}"""),
                ("user", """Analyze this potential relationship:

Table 1: {from_table} (domain: {from_domain})
  Column: {from_col}
  Type: {from_type}
  Semantic type: {from_semantic_type}
  Business meaning: {from_meaning}
  Sample values: {from_samples}

Table 2: {to_table} (domain: {to_domain})
  Column: {to_col}
  Type: {to_type}
  Semantic type: {to_semantic_type}
  Business meaning: {to_meaning}
  Sample values: {to_samples}

Score this as a foreign key relationship:""")
            ])

            try:
                chain = prompt | self.llm
                result = chain.invoke({
                    "from_table": from_meta.get('table_name', 'unknown'),
                    "from_domain": from_meta.get('domain', 'Unknown'),
                    "from_col": from_col,
                    "from_type": str(from_df[from_col].dtype),
                    "from_semantic_type": from_col_meta.get('semantic_type', 'unknown'),
                    "from_meaning": from_col_meta.get('business_meaning', 'N/A'),
                    "from_samples": str(from_samples[:3]),
                    "to_table": to_meta.get('table_name', 'unknown'),
                    "to_domain": to_meta.get('domain', 'Unknown'),
                    "to_col": to_col,
                    "to_type": str(to_df[to_col].dtype),
                    "to_semantic_type": to_col_meta.get('semantic_type', 'unknown'),
                    "to_meaning": to_col_meta.get('business_meaning', 'N/A'),
                    "to_samples": str(to_samples[:3])
                })

                # Parse LLM response
                response_text = result.content.strip()
                # Remove markdown code blocks if present
                if '```' in response_text:
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]

                llm_result = json.loads(response_text)
                score = float(llm_result.get('score', 0.0))
                reasoning = llm_result.get('reasoning', '')

                print(f"    [LLM] {from_col} -> {to_col}: score={score:.2f}, reasoning={reasoning}")

                # Only add if LLM thinks it's likely a relationship (>0.7)
                if score >= 0.7:
                    matches.append({
                        'from_dataset_id': from_dataset_id,
                        'to_dataset_id': to_dataset_id,
                        'from_column': from_col,
                        'to_column': to_col,
                        'llm_similarity': min(1.0, score),
                        'llm_reasoning': reasoning
                    })

            except Exception as e:
                # If LLM fails, skip this candidate
                print(f"    [WARN] LLM interpretation failed for {from_col}->{to_col}: {e}")
                continue

        return matches

    def _combine_matches(self, name_matches: List[Dict], llm_matches: List[Dict]) -> List[Dict]:
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

        # Add or update with LLM matches
        for match in llm_matches:
            key = (
                match['from_dataset_id'],
                match['to_dataset_id'],
                match['from_column'],
                match['to_column']
            )

            if key in combined:
                # Merge scores
                combined[key]['llm_similarity'] = match.get('llm_similarity', 0)
                combined[key]['llm_reasoning'] = match.get('llm_reasoning', '')
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
            'match_percentage': float(match_pct),  # Convert to Python float
            'relationship_type': rel_type,
            'join_strategy': join_strategy,
            'from_unique_count': int(len(from_values)),  # Convert to Python int
            'to_unique_count': int(len(to_values)),  # Convert to Python int
            'intersection_count': int(len(intersection))  # Convert to Python int
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

        # LLM similarity (weight: 0.4 - most important, replaces embedding similarity)
        if 'llm_similarity' in match:
            confidence_components.append(match['llm_similarity'] * 0.4)

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

        # Convert to Python float to avoid numpy type issues
        return float(confidence)

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
            print(f"    [LLM] Semantic validation response: {result.content.strip()}")
            valid_indices = json.loads(result.content.strip())
            validated = [candidate_relationships[i] for i in valid_indices if i < len(candidate_relationships)]

            print(f"[STEP 5] Semantic validation: {len(validated)}/{len(candidate_relationships)} relationships confirmed")
            if len(validated) < len(candidate_relationships):
                rejected = len(candidate_relationships) - len(validated)
                print(f"    Rejected {rejected} relationships as redundant or invalid")
            return validated

        except json.JSONDecodeError:
            print(f"[WARN] LLM semantic validation failed, accepting all statistical matches")
            return candidate_relationships