"""LangGraph workflow for multi-dataset ETL with embeddings."""

from typing import Dict, List, Any, AsyncGenerator
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
import pandas as pd
from src.models.etl_models import ETLState
from src.etl.data_ingestion import DataIngestion
from src.etl.semantic_analyzer import SemanticAnalyzer
from src.etl.relationship_detector import RelationshipDetector
from src.etl.adaptive_cleaner import AdaptiveCleaner
from src.etl.kpi_calculator import KPICalculator
from src.database.connection import DatabaseManager
from src.database.repository import (
    DatasetRepository, RelationshipRepository, CompanyRepository,
    ColumnMetadataRepository
)


class ETLWorkflow:
    """
    Multi-dataset ETL workflow with pgvector and Gemini embeddings.

    Steps:
    1. Ingest → Load files
    2. Semantic Analysis → Understand domain/entities + generate embeddings
    3. Relationship Detection → Find FK relationships using embeddings
    4. Adaptive Cleaning → Clean with FK awareness
    5. KPI Calculation → Add derived KPIs
    6. Store → Save to PostgreSQL with embeddings
    """

    def __init__(self, company_id: str):
        """
        Args:
            company_id: Company UUID for this ETL run
        """
        self.company_id = company_id
        self.semantic_analyzer = SemanticAnalyzer()
        self.relationship_detector = RelationshipDetector(confidence_threshold=0.8)
        self.adaptive_cleaner = AdaptiveCleaner()
        # PERFORMANCE: Disable slow LLM suggestions and embedding creation by default
        # Pre-defined KPIs still work (Finance, Sales, Marketing, HR, Operations)
        # Enable these flags only if you need custom KPIs or KPI embeddings for search
        self.kpi_calculator = KPICalculator(
            enable_custom_suggestions=False,  # Saves ~2-5s per dataset
            enable_embeddings=False  # Saves ~1-3s per KPI
        )

        # Build workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build ETL workflow graph."""
        workflow = StateGraph(ETLState)

        # Add nodes
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("semantic_analysis", self._semantic_analysis_node)
        workflow.add_node("detect_relationships", self._detect_relationships_node)
        workflow.add_node("clean_data", self._clean_data_node)
        workflow.add_node("calculate_kpis", self._calculate_kpis_node)
        workflow.add_node("store_to_db", self._store_to_db_node)

        # Define flow
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "semantic_analysis")
        workflow.add_edge("semantic_analysis", "detect_relationships")
        workflow.add_edge("detect_relationships", "clean_data")
        workflow.add_edge("clean_data", "calculate_kpis")
        workflow.add_edge("calculate_kpis", "store_to_db")
        workflow.add_edge("store_to_db", END)

        return workflow.compile()

    def run_etl(self, file_paths: List[str], mode: str = 'create') -> Dict:
        """
        Run ETL on multiple files.

        Args:
            file_paths: List of file paths to process
            mode: Operation mode - 'create' = add new datasets (error on duplicate)

        Returns:
            ETL result with dataset IDs
        """
        print("\n" + "="*80)
        print(f"[ETL] STARTING ETL FOR COMPANY ID: {self.company_id}")
        print(f"[ETL] MODE: {mode.upper()}")
        print("="*80)
        print(f"Files to process: {len(file_paths)}")

        initial_state: ETLState = {
            'company_name': '',  # Deprecated, kept for compatibility
            'file_paths': file_paths,
            'mode': mode,
            'raw_dataframes': {},
            'file_metadata': {},
            'semantic_metadata': {},
            'relationships': [],
            'cleaned_dataframes': {},
            'cleaning_reports': {},
            'kpi_definitions': {},
            'calculated_kpis': {},
            'dataset_ids': {},
            'company_id': self.company_id,
            'status': 'pending',
            'error_message': '',
            'current_step': 'initialization'
        }

        # Run workflow
        try:
            final_state = self.graph.invoke(initial_state)

            if final_state['status'] == 'completed':
                print("\n" + "="*80)
                print("[SUCCESS] ETL COMPLETED")
                print("="*80)
                print(f"Datasets processed: {len(final_state['dataset_ids'])}")
                print(f"Relationships found: {len(final_state['relationships'])}")

                # Print summary
                for file_id, dataset_id in final_state['dataset_ids'].items():
                    meta = final_state['semantic_metadata'].get(file_id, {})
                    print(f"  - {meta.get('table_name', 'unknown')}: {meta.get('domain', 'Unknown')} domain")

                return final_state
            else:
                raise Exception(final_state.get('error_message', 'ETL failed'))

        except Exception as e:
            print(f"\n[ERROR] ETL failed: {e}")
            raise

    async def astream_etl(self, file_paths: List[str], mode: str = 'create') -> AsyncGenerator[Dict, None]:
        """
        Run ETL on multiple files with progress streaming.

        Streams custom progress events that can be consumed by the service layer
        to provide real-time updates to the frontend via SSE.

        Args:
            file_paths: List of file paths to process
            mode: Operation mode - 'create' = add new datasets

        Yields:
            Custom progress events with step, progress, and message
        """
        print("\n" + "="*80)
        print(f"[ETL] STARTING ETL STREAM FOR COMPANY ID: {self.company_id}")
        print(f"[ETL] MODE: {mode.upper()}")
        print("="*80)
        print(f"Files to process: {len(file_paths)}")

        initial_state: ETLState = {
            'company_name': '',
            'file_paths': file_paths,
            'mode': mode,
            'raw_dataframes': {},
            'file_metadata': {},
            'semantic_metadata': {},
            'relationships': [],
            'cleaned_dataframes': {},
            'cleaning_reports': {},
            'kpi_definitions': {},
            'calculated_kpis': {},
            'dataset_ids': {},
            'company_id': self.company_id,
            'status': 'pending',
            'error_message': '',
            'current_step': 'initialization'
        }

        try:
            # Stream using custom mode to capture progress events from nodes
            async for event in self.graph.astream(initial_state, stream_mode="custom"):
                # event is a tuple: (node_name, custom_data)
                if isinstance(event, tuple) and len(event) == 2:
                    node_name, data = event
                    # Yield the custom data (progress event)
                    yield data

            print("\n" + "="*80)
            print("[SUCCESS] ETL STREAM COMPLETED")
            print("="*80)

        except Exception as e:
            print(f"\n[ERROR] ETL stream failed: {e}")
            # Yield error event
            yield {
                "step": "error",
                "progress": 0,
                "message": f"ETL failed: {str(e)}",
                "error": str(e)
            }
            raise

    # Node implementations

    def _ingest_node(self, state: ETLState) -> ETLState:
        """Node 1: Load all files."""
        state['current_step'] = 'ingestion'
        print("\n[STEP 1/6] INGESTING FILES")

        # Emit progress event
        try:
            writer = get_stream_writer()
            writer({"step": "ingestion", "progress": 15, "message": f"Loading {len(state['file_paths'])} file(s)..."})
        except:
            pass  # Writer only works in streaming mode

        try:
            for i, file_path in enumerate(state['file_paths']):
                print(f"  Loading {i+1}/{len(state['file_paths'])}: {file_path}")

                df, metadata = DataIngestion.load_file(file_path)

                # Validate DataFrame
                validation = DataIngestion.validate_dataframe(df)
                if not validation['is_valid']:
                    print(f"    [WARN] Validation issues: {validation['issues']}")

                file_id = f"file_{i}"
                state['raw_dataframes'][file_id] = df
                state['file_metadata'][file_id] = metadata

            print(f"[OK] Loaded {len(state['raw_dataframes'])} datasets")
            print(f"  File IDs: {list(state['raw_dataframes'].keys())}")

            # Emit completion event
            try:
                writer = get_stream_writer()
                writer({"step": "ingestion_complete", "progress": 28, "message": f"Loaded {len(state['raw_dataframes'])} dataset(s)"})
            except:
                pass

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Ingestion failed: {e}"

        return state

    def _semantic_analysis_node(self, state: ETLState) -> ETLState:
        """Node 2: Analyze semantic meaning of each dataset with embeddings."""
        state['current_step'] = 'semantic_analysis'
        print("\n[STEP 2/6] SEMANTIC ANALYSIS")

        # Emit progress event
        try:
            writer = get_stream_writer()
            writer({"step": "semantic_analysis", "progress": 28, "message": "Analyzing dataset semantics..."})
        except:
            pass

        try:
            for file_id, df in state['raw_dataframes'].items():
                file_meta = state['file_metadata'][file_id]
                print(f"  Analyzing {file_meta['file_name']}...")

                # Generate table name
                table_name = self.semantic_analyzer.generate_table_name(
                    df,
                    file_meta['file_name']
                )

                # Perform semantic analysis (includes embeddings)
                analysis = self.semantic_analyzer.analyze_dataset(df, table_name)
                analysis['table_name'] = table_name

                # Suggest KPIs
                suggested_kpis = self.semantic_analyzer.suggest_kpis(analysis, df)
                analysis['suggested_kpis'] = suggested_kpis

                state['semantic_metadata'][file_id] = analysis

                print(f"    Table: {table_name}")
                print(f"    Domain: {analysis['domain']}")
                print(f"    Entities: {', '.join(analysis.get('entities', []))}")

            print(f"[OK] Analyzed {len(state['semantic_metadata'])} datasets")

        except Exception as e:
            print(f"[WARN] Semantic analysis failed: {e}")

            # Create fallback metadata for missing files
            for file_id, df in state['raw_dataframes'].items():
                if file_id not in state['semantic_metadata']:
                    file_meta = state['file_metadata'][file_id]
                    # Create minimal fallback metadata
                    table_name = Path(file_meta['file_name']).stem.replace('_', ' ').title().replace(' ', '_').lower()
                    state['semantic_metadata'][file_id] = {
                        'table_name': table_name,
                        'domain': 'Unknown',
                        'description': f"Dataset from {file_meta['file_name']}",
                        'entities': [],
                        'column_semantics': {},
                        'suggested_kpis': []
                    }
                    print(f"  [FALLBACK] Created metadata for {table_name}")

            print(f"[WARN] Using fallback metadata for {len(state['raw_dataframes'])} datasets")

        # Emit completion event
        try:
            writer = get_stream_writer()
            writer({"step": "semantic_analysis_complete", "progress": 41, "message": f"Analyzed {len(state['semantic_metadata'])} dataset(s)"})
        except:
            pass

        return state

    def _detect_relationships_node(self, state: ETLState) -> ETLState:
        """Node 3: Detect relationships between datasets using embeddings."""
        state['current_step'] = 'relationship_detection'
        print("\n[STEP 3/6] DETECTING RELATIONSHIPS")

        # Emit progress event
        try:
            writer = get_stream_writer()
            writer({"step": "relationship_detection", "progress": 41, "message": "Detecting relationships between datasets..."})
        except:
            pass

        try:
            # CREATE mode: Load existing datasets for cross-ETL relationship detection
            existing_datasets = None
            existing_metadata = None

            if state['mode'] == 'create':
                print("  [CREATE MODE] Loading existing datasets for cross-ETL relationship detection...")
                with DatabaseManager.get_session() as session:
                    # Use company_id from state (already set in __init__)
                    company_id = state['company_id']
                    print(f"  Company ID: {company_id}")

                    # Load existing datasets
                    existing_datasets, existing_metadata = DatasetRepository.load_existing_datasets_for_company(
                        session, company_id
                    )

                    if existing_datasets:
                        print(f"  Loaded {len(existing_datasets)} existing datasets")

            # Detect relationships
            if len(state['raw_dataframes']) < 2 and not existing_datasets:
                print("  [INFO] Single dataset and no existing datasets - no relationships to detect")
                state['relationships'] = []
            else:
                relationships = self.relationship_detector.detect_relationships(
                    state['raw_dataframes'],
                    state['semantic_metadata'],
                    existing_datasets=existing_datasets,
                    existing_metadata=existing_metadata
                )

                state['relationships'] = relationships
                print(f"[OK] Found {len(relationships)} confirmed relationships")

                # Print detected relationships
                for rel in relationships:
                    # Check if dataset is in new or existing
                    from_meta = state['semantic_metadata'].get(rel['from_dataset_id']) or (existing_metadata or {}).get(rel['from_dataset_id'], {})
                    to_meta = state['semantic_metadata'].get(rel['to_dataset_id']) or (existing_metadata or {}).get(rel['to_dataset_id'], {})

                    from_table = from_meta.get('table_name', 'unknown')
                    to_table = to_meta.get('table_name', 'unknown')
                    print(f"  - {from_table}.{rel['from_column']} → {to_table}.{rel['to_column']} "
                          f"(confidence: {rel['confidence']:.2f})")

                print(f"  State now has {len(state['relationships'])} relationships")

        except Exception as e:
            print(f"[WARN] Relationship detection failed: {e}")
            state['relationships'] = []

        # Emit completion event
        try:
            writer = get_stream_writer()
            writer({"step": "relationship_detection_complete", "progress": 54, "message": f"Found {len(state['relationships'])} relationship(s)"})
        except:
            pass

        return state

    def _clean_data_node(self, state: ETLState) -> ETLState:
        """Node 4: Clean all datasets with FK awareness."""
        state['current_step'] = 'cleaning'
        print("\n[STEP 4/6] CLEANING DATA")

        # Emit progress event
        try:
            writer = get_stream_writer()
            writer({"step": "cleaning", "progress": 54, "message": "Cleaning data..."})
        except:
            pass

        try:
            for file_id, df in state['raw_dataframes'].items():
                # Get relationships for this dataset
                dataset_rels = [
                    r for r in state['relationships']
                    if r.get('from_dataset_id') == file_id or r.get('to_dataset_id') == file_id
                ]

                # Get semantic metadata or create fallback
                if file_id in state['semantic_metadata']:
                    semantic_meta = state['semantic_metadata'][file_id]
                else:
                    # Fallback when semantic analysis fails (e.g., API quota)
                    file_meta = state['file_metadata'][file_id]
                    semantic_meta = {
                        'table_name': Path(file_meta['file_name']).stem.replace('_', ' ').title().replace(' ', '_').lower(),
                        'domain': 'Unknown',
                        'description': f"Dataset from {file_meta['file_name']}",
                        'entities': [],
                        'column_semantics': {}
                    }
                    state['semantic_metadata'][file_id] = semantic_meta
                    print(f"  [WARN] Using fallback metadata for {file_id}")

                # Clean
                cleaned_df, report = self.adaptive_cleaner.clean_dataset(
                    df,
                    semantic_meta,
                    dataset_rels,
                    file_id
                )

                state['cleaned_dataframes'][file_id] = cleaned_df
                state['cleaning_reports'][file_id] = report
                print(f"  Cleaned {semantic_meta['table_name']}: {len(cleaned_df)} rows")

            print(f"[OK] Cleaned {len(state['cleaned_dataframes'])} datasets")

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Data cleaning failed: {e}"
            print(f"[ERROR] Cleaning failed: {e}")
            import traceback
            traceback.print_exc()

        # Emit completion event
        try:
            writer = get_stream_writer()
            writer({"step": "cleaning_complete", "progress": 67, "message": f"Cleaned {len(state['cleaned_dataframes'])} dataset(s)"})
        except:
            pass

        return state

    def _calculate_kpis_node(self, state: ETLState) -> ETLState:
        """Node 5: Calculate KPIs for each dataset."""
        state['current_step'] = 'kpi_calculation'
        print("\n[STEP 5/6] CALCULATING KPIs")

        # Emit progress event
        try:
            writer = get_stream_writer()
            writer({"step": "kpi_calculation", "progress": 67, "message": "Calculating KPIs..."})
        except:
            pass

        try:
            for file_id, df in state['cleaned_dataframes'].items():
                semantic_meta = state['semantic_metadata'][file_id]
                domain = semantic_meta.get('domain', 'Unknown')
                column_semantics = semantic_meta.get('column_semantics', {})

                print(f"  {semantic_meta.get('table_name', 'unknown')} ({domain}):")

                # Identify applicable KPIs
                kpi_defs = self.kpi_calculator.identify_kpis(
                    df,
                    domain,
                    column_semantics
                )

                # Calculate KPIs
                if kpi_defs:
                    df_with_kpis, calculated_values = self.kpi_calculator.calculate_kpis(df, kpi_defs)
                    state['cleaned_dataframes'][file_id] = df_with_kpis
                    state['kpi_definitions'][file_id] = kpi_defs
                    state['calculated_kpis'][file_id] = calculated_values
                else:
                    print("    No KPIs identified")

        except Exception as e:
            print(f"[WARN] KPI calculation failed: {e}")

        # Emit completion event
        try:
            writer = get_stream_writer()
            writer({"step": "kpi_calculation_complete", "progress": 80, "message": "KPI calculation complete"})
        except:
            pass

        return state

    def _store_to_db_node(self, state: ETLState) -> ETLState:
        """Node 6: Store everything to PostgreSQL with embeddings."""
        state['current_step'] = 'storage'
        print("\n[STEP 6/6] STORING TO DATABASE")
        print(f"  Relationships in state: {len(state.get('relationships', []))}")

        # Emit progress event
        try:
            writer = get_stream_writer()
            writer({"step": "storage", "progress": 80, "message": "Storing to database..."})
        except:
            pass

        try:
            with DatabaseManager.get_session() as session:
                # Use company_id from state (already set in __init__)
                company_id = state['company_id']
                print(f"  Company ID: {company_id}")

                # CREATE mode: Check for duplicates
                if state['mode'] == 'create':
                    print("  [CREATE MODE] Checking for duplicate files...")
                    for file_id, df in state['cleaned_dataframes'].items():
                        file_meta = state['file_metadata'][file_id]
                        filename = file_meta['file_name']

                        existing = DatasetRepository.find_dataset_by_filename(
                            session, company_id, filename
                        )

                        if existing:
                            error_msg = f"Dataset '{filename}' already exists for this company"
                            print(f"  [ERROR] {error_msg}")
                            state['status'] = 'error'
                            state['error_message'] = error_msg
                            raise ValueError(error_msg)

                    print("  [OK] No duplicates found")

                # Store each dataset
                print(f"  Files to store: {list(state['cleaned_dataframes'].keys())}")
                for file_id, df in state['cleaned_dataframes'].items():
                    file_meta = state['file_metadata'][file_id]
                    semantic_meta = state['semantic_metadata'][file_id]

                    print(f"  Storing {semantic_meta['table_name']}...")

                    # Generate storage path (not temp path) for the dataset record
                    # This is the permanent storage location (GCS or local storage)
                    from app.services.storage_service import StorageService
                    storage_path = StorageService.get_gcs_path(company_id, file_meta['file_name'])

                    # Create dataset record with embeddings and unified context
                    dataset = DatasetRepository.create_dataset(
                        session=session,
                        company_id=company_id,
                        original_filename=file_meta['file_name'],
                        table_name=semantic_meta['table_name'],
                        file_type=file_meta['file_type'],
                        file_path=storage_path,  # Store permanent storage path, not temp path
                        domain=semantic_meta.get('domain'),
                        department=semantic_meta.get('department'),
                        description=semantic_meta.get('description'),
                        entities=semantic_meta.get('entities'),
                        description_embedding=semantic_meta.get('description_embedding'),
                        schema_embedding=semantic_meta.get('schema_embedding'),
                        # New unified context fields
                        dataset_type=semantic_meta.get('dataset_type'),
                        time_period=semantic_meta.get('time_period'),
                        typical_use_cases=semantic_meta.get('typical_use_cases'),
                        business_context=semantic_meta.get('business_context')
                    )

                    # Store column metadata with embeddings (BULK INSERT for performance)
                    column_semantics = semantic_meta.get('column_semantics', {})
                    columns_data = []
                    for i, col_name in enumerate(df.columns):
                        col_meta = column_semantics.get(col_name, {})
                        columns_data.append({
                            'dataset_id': dataset.id,
                            'column_name': col_name,
                            'original_name': col_name,
                            'position': i,
                            'data_type': str(df[col_name].dtype),
                            'semantic_type': col_meta.get('semantic_type'),
                            'business_meaning': col_meta.get('business_meaning'),
                            'is_primary_key': col_meta.get('is_primary_key', False),
                            'is_foreign_key': col_meta.get('is_foreign_key', False),
                            'semantic_embedding': col_meta.get('semantic_embedding'),
                            'null_count': col_meta.get('null_count'),
                            'null_percentage': col_meta.get('null_percentage'),
                            'unique_count': col_meta.get('unique_count'),
                            'unique_percentage': col_meta.get('unique_percentage')
                        })

                    # Bulk insert all columns at once (much faster than individual inserts)
                    ColumnMetadataRepository.bulk_create_column_metadata(session, columns_data)

                    # Store DataFrame to PostgreSQL
                    table_name = DatasetRepository.store_dataframe(
                        session,
                        dataset.id,
                        df,
                        table_prefix='cleaned'
                    )

                    # Update status
                    DatasetRepository.update_dataset_status(
                        session,
                        dataset.id,
                        status='ready',
                        row_count=len(df),
                        column_count=len(df.columns)
                    )

                    # Store KPI definitions
                    if file_id in state['kpi_definitions']:
                        self.kpi_calculator.store_kpi_definitions(
                            session,
                            semantic_meta.get('domain', 'Unknown'),
                            state['kpi_definitions'][file_id]
                        )

                    state['dataset_ids'][file_id] = dataset.id

                    print(f"    Stored as {table_name}: {len(df):,} rows × {len(df.columns)} columns")

                # Store relationships (BULK INSERT for performance)
                print(f"  Storing {len(state['relationships'])} relationships...")
                print(f"    Dataset ID mapping: {list(state['dataset_ids'].keys())}")

                relationships_data = []
                for i, rel in enumerate(state['relationships']):
                    # The relationships use file IDs (e.g., 'file_0'), not dataset IDs
                    from_file_id = rel['from_dataset_id']
                    to_file_id = rel['to_dataset_id']
                    from_dataset_id = state['dataset_ids'].get(from_file_id)
                    to_dataset_id = state['dataset_ids'].get(to_file_id)

                    if i == 0:  # Debug first relationship
                        print(f"    First rel: from={from_file_id} -> {from_dataset_id}, to={to_file_id} -> {to_dataset_id}")

                    if from_dataset_id and to_dataset_id:
                        # Convert numpy types to Python native types
                        confidence = float(rel['confidence'])
                        match_pct = float(rel.get('match_percentage', 0))

                        relationships_data.append({
                            'from_dataset_id': from_dataset_id,
                            'to_dataset_id': to_dataset_id,
                            'from_column': rel['from_column'],
                            'to_column': rel['to_column'],
                            'relationship_type': rel['relationship_type'],
                            'confidence': confidence,
                            'match_percentage': match_pct,
                            'join_strategy': rel.get('join_strategy', 'left')
                        })
                        print(f"    Queued: {rel['from_column']} -> {rel['to_column']} (conf: {confidence:.2f})")
                    else:
                        print(f"    Skipped relationship: from_id={from_file_id} ({from_dataset_id}), to_id={to_file_id} ({to_dataset_id})")

                # Bulk insert all relationships at once (much faster than individual inserts)
                if relationships_data:
                    RelationshipRepository.bulk_create_relationships(session, relationships_data)
                    print(f"    Bulk inserted {len(relationships_data)} relationships")

            state['status'] = 'completed'
            print("[OK] All data stored successfully")

            # Emit final completion event with dataset IDs for service layer
            try:
                writer = get_stream_writer()
                writer({
                    "step": "completed",
                    "progress": 95,
                    "message": "ETL completed successfully",
                    "dataset_ids": state.get('dataset_ids', {}),
                    "semantic_metadata": state.get('semantic_metadata', {})
                })
            except:
                pass

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Database storage failed: {e}"
            import traceback
            traceback.print_exc()

        return state