"""LangGraph workflow for multi-dataset ETL with embeddings."""

from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
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

    def __init__(self, company_name: str):
        """
        Args:
            company_name: Company identifier for this ETL run
        """
        self.company_name = company_name
        self.semantic_analyzer = SemanticAnalyzer()
        self.relationship_detector = RelationshipDetector(confidence_threshold=0.8)
        self.adaptive_cleaner = AdaptiveCleaner()
        self.kpi_calculator = KPICalculator()

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

    def run_etl(self, file_paths: List[str]) -> Dict:
        """
        Run ETL on multiple files.

        Args:
            file_paths: List of file paths to process

        Returns:
            ETL result with dataset IDs
        """
        print("\n" + "="*80)
        print(f"[ETL] STARTING ETL FOR COMPANY: {self.company_name}")
        print("="*80)
        print(f"Files to process: {len(file_paths)}")

        initial_state: ETLState = {
            'company_name': self.company_name,
            'file_paths': file_paths,
            'raw_dataframes': {},
            'file_metadata': {},
            'semantic_metadata': {},
            'relationships': [],
            'cleaned_dataframes': {},
            'cleaning_reports': {},
            'kpi_definitions': {},
            'calculated_kpis': {},
            'dataset_ids': {},
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

    # Node implementations

    def _ingest_node(self, state: ETLState) -> ETLState:
        """Node 1: Load all files."""
        state['current_step'] = 'ingestion'
        print("\n[STEP 1/6] INGESTING FILES")

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

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Ingestion failed: {e}"

        return state

    def _semantic_analysis_node(self, state: ETLState) -> ETLState:
        """Node 2: Analyze semantic meaning of each dataset with embeddings."""
        state['current_step'] = 'semantic_analysis'
        print("\n[STEP 2/6] SEMANTIC ANALYSIS")

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

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Semantic analysis failed: {e}"

        return state

    def _detect_relationships_node(self, state: ETLState) -> ETLState:
        """Node 3: Detect relationships between datasets using embeddings."""
        state['current_step'] = 'relationship_detection'
        print("\n[STEP 3/6] DETECTING RELATIONSHIPS")

        try:
            if len(state['raw_dataframes']) < 2:
                print("  [INFO] Single dataset - no relationships to detect")
                state['relationships'] = []
            else:
                relationships = self.relationship_detector.detect_relationships(
                    state['raw_dataframes'],
                    state['semantic_metadata']
                )

                state['relationships'] = relationships

                # Print detected relationships
                for rel in relationships:
                    from_table = state['semantic_metadata'][rel['from_dataset_id']].get('table_name', 'unknown')
                    to_table = state['semantic_metadata'][rel['to_dataset_id']].get('table_name', 'unknown')
                    print(f"  Found: {from_table}.{rel['from_column']} → {to_table}.{rel['to_column']} "
                          f"(confidence: {rel['confidence']:.2f})")

        except Exception as e:
            print(f"[WARN] Relationship detection failed: {e}")
            state['relationships'] = []

        return state

    def _clean_data_node(self, state: ETLState) -> ETLState:
        """Node 4: Clean all datasets with FK awareness."""
        state['current_step'] = 'cleaning'
        print("\n[STEP 4/6] CLEANING DATA")

        try:
            for file_id, df in state['raw_dataframes'].items():
                # Get relationships for this dataset
                dataset_rels = [
                    r for r in state['relationships']
                    if r.get('from_dataset_id') == file_id or r.get('to_dataset_id') == file_id
                ]

                # Clean
                cleaned_df, report = self.adaptive_cleaner.clean_dataset(
                    df,
                    state['semantic_metadata'][file_id],
                    dataset_rels,
                    file_id
                )

                state['cleaned_dataframes'][file_id] = cleaned_df
                state['cleaning_reports'][file_id] = report

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Data cleaning failed: {e}"

        return state

    def _calculate_kpis_node(self, state: ETLState) -> ETLState:
        """Node 5: Calculate KPIs for each dataset."""
        state['current_step'] = 'kpi_calculation'
        print("\n[STEP 5/6] CALCULATING KPIs")

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

        return state

    def _store_to_db_node(self, state: ETLState) -> ETLState:
        """Node 6: Store everything to PostgreSQL with embeddings."""
        state['current_step'] = 'storage'
        print("\n[STEP 6/6] STORING TO DATABASE")

        try:
            with DatabaseManager.get_session() as session:
                # Get or create company
                company = CompanyRepository.get_or_create_company(
                    session,
                    state['company_name']
                )
                print(f"  Company: {company.name} (ID: {company.id})")

                # Store each dataset
                for file_id, df in state['cleaned_dataframes'].items():
                    file_meta = state['file_metadata'][file_id]
                    semantic_meta = state['semantic_metadata'][file_id]

                    print(f"  Storing {semantic_meta['table_name']}...")

                    # Create dataset record with embeddings
                    dataset = DatasetRepository.create_dataset(
                        session=session,
                        company_id=company.id,
                        original_filename=file_meta['file_name'],
                        table_name=semantic_meta['table_name'],
                        file_type=file_meta['file_type'],
                        file_path=file_meta['file_path'],
                        domain=semantic_meta.get('domain'),
                        department=semantic_meta.get('department'),
                        description=semantic_meta.get('description'),
                        entities=semantic_meta.get('entities'),
                        description_embedding=semantic_meta.get('description_embedding'),
                        schema_embedding=semantic_meta.get('schema_embedding')
                    )

                    # Store column metadata with embeddings
                    column_semantics = semantic_meta.get('column_semantics', {})
                    for i, col_name in enumerate(df.columns):
                        col_meta = column_semantics.get(col_name, {})

                        ColumnMetadataRepository.create_column_metadata(
                            session=session,
                            dataset_id=dataset.id,
                            column_name=col_name,
                            original_name=col_name,
                            position=i,
                            data_type=str(df[col_name].dtype),
                            semantic_type=col_meta.get('semantic_type'),
                            business_meaning=col_meta.get('business_meaning'),
                            is_primary_key=col_meta.get('is_primary_key', False),
                            is_foreign_key=col_meta.get('is_foreign_key', False),
                            semantic_embedding=col_meta.get('semantic_embedding'),
                            null_count=col_meta.get('null_count'),
                            null_percentage=col_meta.get('null_percentage'),
                            unique_count=col_meta.get('unique_count'),
                            unique_percentage=col_meta.get('unique_percentage')
                        )

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

                # Store relationships
                for rel in state['relationships']:
                    from_dataset_id = state['dataset_ids'].get(rel['from_dataset_id'])
                    to_dataset_id = state['dataset_ids'].get(rel['to_dataset_id'])

                    if from_dataset_id and to_dataset_id:
                        RelationshipRepository.create_relationship(
                            session=session,
                            from_dataset_id=from_dataset_id,
                            to_dataset_id=to_dataset_id,
                            from_column=rel['from_column'],
                            to_column=rel['to_column'],
                            relationship_type=rel['relationship_type'],
                            confidence=rel['confidence'],
                            match_percentage=rel.get('match_percentage', 0),
                            join_strategy=rel.get('join_strategy', 'left')
                        )

            state['status'] = 'completed'
            print("[OK] All data stored successfully")

        except Exception as e:
            state['status'] = 'error'
            state['error_message'] = f"Database storage failed: {e}"
            import traceback
            traceback.print_exc()

        return state