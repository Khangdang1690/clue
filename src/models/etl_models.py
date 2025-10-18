"""Data models for ETL workflow."""

from typing import TypedDict, Dict, List, Any, Optional
import pandas as pd


class ETLState(TypedDict):
    """State for ETL workflow using LangGraph."""

    # Input
    company_name: str
    file_paths: List[str]

    # Step 1: Ingestion
    raw_dataframes: Dict[str, pd.DataFrame]  # {file_id: df}
    file_metadata: Dict[str, Dict]  # {file_id: metadata}

    # Step 2: Semantic Analysis
    semantic_metadata: Dict[str, Dict]  # {file_id: analysis with embeddings}

    # Step 3: Relationships
    relationships: List[Dict]  # List of relationship dicts with confidence

    # Step 4: Cleaning
    cleaned_dataframes: Dict[str, pd.DataFrame]
    cleaning_reports: Dict[str, Dict]

    # Step 5: KPIs
    kpi_definitions: Dict[str, List[Dict]]  # {file_id: [kpi_defs]}
    calculated_kpis: Dict[str, List[Dict]]  # {file_id: [kpi_values]}

    # Step 6: Storage
    dataset_ids: Dict[str, str]  # {file_id: dataset_id in DB}

    # Status tracking
    status: str  # pending, running, completed, error
    error_message: str
    current_step: str  # Current workflow step


class MultiTableDiscoveryState(TypedDict):
    """State for multi-table discovery workflow."""

    # Input
    company_id: str
    dataset_ids: List[str]
    analysis_name: str

    # Data
    datasets: Dict[str, pd.DataFrame]  # {dataset_id: DataFrame}
    metadata: Dict[str, Dict]  # {dataset_id: metadata}
    relationships: List[Dict]  # Relationships between datasets

    # Phase 1: Single table results
    single_table_results: Dict[str, Any]  # {dataset_id: DiscoveryResult}

    # Phase 2: Cross-table results
    joined_dataframe: Optional[pd.DataFrame]
    cross_table_insights: List[Dict]
    suggested_datasets: List[str]  # Additional datasets suggested by similarity

    # Results
    unified_report_path: Optional[str]
    analysis_session_id: Optional[str]

    # Status
    status: str
    error_message: str
    current_phase: str  # single_table, cross_table, reporting