"""ETL Pipeline for processing data from various sources."""

import pandas as pd
import PyPDF2
from pathlib import Path
from typing import List, Dict, Tuple, Union
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_client import get_llm
import json


class ETLPipeline:
    """Handles Extract, Transform, Load operations for various data formats."""

    def __init__(self, data_directory: str = "data/uploads"):
        """
        Initialize ETL pipeline.

        Args:
            data_directory: Directory containing uploaded data files
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.llm = get_llm(temperature=0.2)
        self.loaded_data: Dict[str, pd.DataFrame] = {}

    def extract(self, department: Union[str, List[str]] = None) -> Dict[str, any]:
        """
        Extract data from various sources.

        Args:
            department: Optional department name(s) to filter files (can be string or list)

        Returns:
            Dictionary containing extracted data and metadata
        """
        extracted_data = {
            "csv_data": [],
            "excel_data": [],
            "pdf_text": [],
            "metadata": []
        }

        # Convert department to list for uniform handling
        departments = []
        if department:
            if isinstance(department, str):
                departments = [department]
            elif isinstance(department, list):
                departments = department

        # Determine search directories
        search_dirs = []
        if departments:
            for dept in departments:
                dept_dir = self.data_directory / dept
                if dept_dir.exists():
                    search_dirs.append(dept_dir)
        else:
            search_dirs = [self.data_directory]

        # Extract from different file types across all search directories
        for search_dir in search_dirs:
            for file_path in search_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                try:
                    if file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path)
                        extracted_data["csv_data"].append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "data": df,
                            "rows": len(df),
                            "columns": list(df.columns)
                        })

                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        # Read all sheets
                        excel_file = pd.ExcelFile(file_path)
                        for sheet_name in excel_file.sheet_names:
                            df = pd.read_excel(file_path, sheet_name=sheet_name)
                            extracted_data["excel_data"].append({
                                "name": f"{file_path.name} - {sheet_name}",
                                "path": str(file_path),
                                "sheet": sheet_name,
                                "data": df,
                                "rows": len(df),
                                "columns": list(df.columns)
                            })

                    elif file_path.suffix.lower() == '.pdf':
                        text = self._extract_pdf_text(file_path)
                        extracted_data["pdf_text"].append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "text": text,
                            "length": len(text)
                        })

                    # Get department name from parent directory if in department folder
                    dept_name = search_dir.name if search_dir != self.data_directory else "general"
                    extracted_data["metadata"].append({
                        "file": file_path.name,
                        "type": file_path.suffix,
                        "size": file_path.stat().st_size,
                        "department": dept_name
                    })

                except Exception as e:
                    print(f"Error extracting {file_path}: {e}")

        return extracted_data

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text

    def transform(self, extracted_data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Transform and clean extracted data using LLM-guided strategies.

        Args:
            extracted_data: Data extracted from various sources

        Returns:
            Dictionary of cleaned DataFrames
        """
        transformed_data = {}

        # Transform CSV data
        for idx, csv_item in enumerate(extracted_data["csv_data"]):
            df = csv_item["data"]
            cleaned_df = self._clean_dataframe(df, csv_item["name"])
            transformed_data[f"csv_{idx}_{csv_item['name']}"] = cleaned_df

        # Transform Excel data
        for idx, excel_item in enumerate(extracted_data["excel_data"]):
            df = excel_item["data"]
            cleaned_df = self._clean_dataframe(df, excel_item["name"])
            transformed_data[f"excel_{idx}_{excel_item['name']}"] = cleaned_df

        # Transform PDF text to structured data if possible
        for idx, pdf_item in enumerate(extracted_data["pdf_text"]):
            structured_data = self._extract_structured_data_from_text(
                pdf_item["text"],
                pdf_item["name"]
            )
            if structured_data is not None:
                transformed_data[f"pdf_{idx}_{pdf_item['name']}"] = structured_data

        return transformed_data

    def _clean_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Clean a DataFrame using standard techniques.

        Args:
            df: DataFrame to clean
            source_name: Name of the data source

        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()

        # Remove completely empty rows and columns
        cleaned_df = cleaned_df.dropna(how='all', axis=0)
        cleaned_df = cleaned_df.dropna(how='all', axis=1)

        # Remove duplicate rows
        cleaned_df = cleaned_df.drop_duplicates()

        # Standardize column names
        cleaned_df.columns = [
            col.strip().lower().replace(' ', '_')
            for col in cleaned_df.columns
        ]

        # Convert data types appropriately
        for col in cleaned_df.columns:
            # Try to convert to numeric
            if cleaned_df[col].dtype == 'object':
                try:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                except:
                    pass

                # Try to convert to datetime
                if cleaned_df[col].dtype == 'object':
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='ignore')
                    except:
                        pass

        # Handle missing values based on column type
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['float64', 'int64']:
                # For numeric columns, fill with median
                if cleaned_df[col].isnull().sum() > 0:
                    median_value = cleaned_df[col].median()
                    if pd.notna(median_value):
                        cleaned_df[col] = cleaned_df[col].fillna(median_value)
            elif cleaned_df[col].dtype == 'object':
                # For categorical columns, fill with mode or 'Unknown'
                if cleaned_df[col].isnull().sum() > 0:
                    mode_value = cleaned_df[col].mode()
                    if len(mode_value) > 0:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna('Unknown')

        return cleaned_df

    def _extract_structured_data_from_text(
        self,
        text: str,
        source_name: str
    ) -> pd.DataFrame:
        """
        Use LLM to extract structured data from text.

        Args:
            text: Text content
            source_name: Name of the source

        Returns:
            DataFrame with extracted data or None
        """
        # Limit text length for LLM
        text_sample = text[:3000] if len(text) > 3000 else text

        extraction_prompt = ChatPromptTemplate.from_template(
            """Analyze this text and determine if it contains structured data (tables, lists, metrics).
If it does, extract the data in JSON format suitable for creating a DataFrame.

Text from {source}:
{text}

If structured data exists, respond with JSON in this format:
{{
    "has_structured_data": true,
    "data": [
        {{"column1": "value1", "column2": "value2"}},
        {{"column1": "value3", "column2": "value4"}}
    ]
}}

If no structured data, respond with:
{{
    "has_structured_data": false,
    "summary": "Brief summary of the text content"
}}
"""
        )

        try:
            chain = extraction_prompt | self.llm
            response = chain.invoke({"source": source_name, "text": text_sample})

            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)

                if result.get("has_structured_data") and result.get("data"):
                    return pd.DataFrame(result["data"])

        except Exception as e:
            print(f"Error extracting structured data from text: {e}")

        return None

    def load(self, transformed_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Load transformed data into final storage.

        Args:
            transformed_data: Dictionary of transformed DataFrames

        Returns:
            Dictionary with load summary
        """
        output_dir = Path("data/outputs/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        load_summary = {
            "loaded_files": [],
            "total_records": 0
        }

        for name, df in transformed_data.items():
            # Store in memory for analysis
            self.loaded_data[name] = df

            # Save to CSV for persistence
            output_path = output_dir / f"{name}_processed.csv"
            df.to_csv(output_path, index=False)

            load_summary["loaded_files"].append(str(output_path))
            load_summary["total_records"] += len(df)

        return load_summary

    def run_etl(self, department: Union[str, List[str]] = None) -> Tuple[Dict, str, str, str]:
        """
        Run the complete ETL pipeline.

        Args:
            department: Optional department name(s) to filter data (can be string or list)

        Returns:
            Tuple of (loaded_data, extraction_summary, transformation_summary, load_summary)
        """
        # Extract
        if department:
            dept_str = ", ".join(department) if isinstance(department, list) else department
            print(f"Extracting data for {dept_str}...")
        else:
            print("Extracting data...")
        extracted_data = self.extract(department)
        extraction_summary = f"""Extracted:
- {len(extracted_data['csv_data'])} CSV files
- {len(extracted_data['excel_data'])} Excel sheets
- {len(extracted_data['pdf_text'])} PDF documents
Total files: {len(extracted_data['metadata'])}"""

        # Transform
        print("Transforming and cleaning data...")
        transformed_data = self.transform(extracted_data)
        transformation_summary = f"""Transformed {len(transformed_data)} datasets:
- Applied data cleaning techniques
- Standardized column names
- Handled missing values
- Converted data types
- Removed duplicates"""

        # Load
        print("Loading data to final destination...")
        load_result = self.load(transformed_data)
        load_summary = f"""Loaded data:
- {len(load_result['loaded_files'])} processed datasets
- {load_result['total_records']} total records
- Stored in memory and persisted to disk"""

        return self.loaded_data, extraction_summary, transformation_summary, load_summary

    def get_loaded_data(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded data."""
        return self.loaded_data

    def get_data_summary(self) -> str:
        """Generate summary of loaded data using LLM."""
        if not self.loaded_data:
            return "No data loaded."

        summary_parts = []
        for name, df in self.loaded_data.items():
            summary_parts.append(f"{name}: {len(df)} rows, {len(df.columns)} columns")

        summary_prompt = ChatPromptTemplate.from_template(
            """As a data analyst, provide a brief overview of these datasets and their potential for analysis.

Datasets:
{datasets}

Provide a 2-3 sentence summary highlighting key characteristics and analysis opportunities.
"""
        )

        chain = summary_prompt | self.llm
        response = chain.invoke({"datasets": "\n".join(summary_parts)})

        return response.content
