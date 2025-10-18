"""Multi-source data ingestion with automatic file type detection."""

import pandas as pd
import chardet
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class DataIngestion:
    """Handles loading data from multiple file formats."""

    SUPPORTED_FORMATS = {
        'csv': ['.csv'],
        'excel': ['.xlsx', '.xls', '.xlsm'],
        'json': ['.json'],
        'parquet': ['.parquet'],
    }

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(100000))
            return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'

    @staticmethod
    def load_file(file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data from file with automatic format detection.

        Args:
            file_path: Path to the file
            **kwargs: Additional parameters for pandas read functions

        Returns:
            (DataFrame, metadata_dict)
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        metadata = {
            'file_path': str(file_path),
            'file_name': path.name,
            'file_type': None,
            'encoding': None,
            'sheets': None,
        }

        try:
            if suffix == '.csv':
                # Detect encoding for CSV
                encoding = DataIngestion.detect_encoding(file_path)
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                metadata['file_type'] = 'csv'
                metadata['encoding'] = encoding

            elif suffix in ['.xlsx', '.xls', '.xlsm']:
                # For Excel, check if it has multiple sheets
                excel_file = pd.ExcelFile(file_path)
                metadata['file_type'] = 'excel'
                metadata['sheets'] = excel_file.sheet_names

                # If multiple sheets, load the first non-empty one
                df = None
                for sheet_name in excel_file.sheet_names:
                    try:
                        temp_df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
                        if not temp_df.empty:
                            df = temp_df
                            print(f"[INFO] Loaded sheet: {sheet_name}")
                            break
                    except:
                        continue

                if df is None:
                    raise ValueError("No valid data found in Excel file")

            elif suffix == '.json':
                # Try different JSON orientations
                try:
                    df = pd.read_json(file_path, **kwargs)
                except:
                    # Try reading as JSON lines
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = [json.loads(line) for line in f]
                    df = pd.DataFrame(data)
                metadata['file_type'] = 'json'

            elif suffix == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
                metadata['file_type'] = 'parquet'

            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            # Basic validation
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")

            # Clean column names (remove special characters)
            df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]

            print(f"[OK] Loaded {len(df):,} rows × {len(df.columns)} columns from {path.name}")

            return df, metadata

        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            raise

    @staticmethod
    def load_directory(directory_path: str) -> List[Tuple[pd.DataFrame, Dict]]:
        """
        Load all supported files from a directory.

        Args:
            directory_path: Path to directory containing data files

        Returns:
            List of (DataFrame, metadata) tuples
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")

        results = []

        # Get all supported files
        supported_extensions = []
        for exts in DataIngestion.SUPPORTED_FORMATS.values():
            supported_extensions.extend(exts)

        files = [f for f in dir_path.iterdir() if f.suffix.lower() in supported_extensions]

        print(f"\n[INGESTION] Found {len(files)} supported files in {directory_path}")

        for file_path in files:
            try:
                df, metadata = DataIngestion.load_file(str(file_path))
                results.append((df, metadata))
            except Exception as e:
                print(f"[SKIP] Skipping {file_path.name}: {e}")

        return results

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict:
        """
        Validate DataFrame and return quality metrics.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }

        # Check for empty DataFrame
        if df.empty:
            validation['is_valid'] = False
            validation['issues'].append("DataFrame is empty")
            return validation

        # Check for all null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            validation['warnings'].append(f"Columns with all null values: {null_columns}")

        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            validation['issues'].append(f"Duplicate column names: {duplicate_cols}")
            validation['is_valid'] = False

        # Check for reasonable size
        if len(df) > 10_000_000:
            validation['warnings'].append(f"Very large dataset: {len(df):,} rows")

        # Check for mixed types in columns (potential data quality issue)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it might be numeric with errors
                try:
                    numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if 0 < numeric_count < len(df[col]) * 0.9:
                        validation['warnings'].append(f"Column '{col}' has mixed numeric/text values")
                except:
                    pass

        return validation