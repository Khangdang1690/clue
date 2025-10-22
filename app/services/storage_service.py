"""Storage service for file uploads - supports local (dev) and GCS (production)."""

import os
from typing import List, BinaryIO
from pathlib import Path
from google.cloud import storage
from app.core.config import get_settings


class StorageService:
    """Service for handling file storage - local filesystem or Google Cloud Storage based on ENV."""

    def __init__(self):
        """Initialize storage client based on environment."""
        self.settings = get_settings()
        self.use_gcs = self.settings.is_production

        if self.use_gcs:
            # Production: Use Google Cloud Storage
            self.client = storage.Client()
            self.bucket_name = self.settings.gcs_bucket_name
            self.bucket = self.client.bucket(self.bucket_name)
            print(f"[STORAGE] Using Google Cloud Storage (bucket: {self.bucket_name})")
        else:
            # Development: Use local filesystem
            self.local_storage_root = "data/storage"
            os.makedirs(self.local_storage_root, exist_ok=True)
            print(f"[STORAGE] Using local filesystem (root: {self.local_storage_root})")

    def upload_file(self, file_content: bytes, destination_path: str) -> str:
        """
        Upload a file to storage (GCS or local filesystem).

        Args:
            file_content: File content as bytes
            destination_path: Path in storage (e.g., 'company_123/sales.csv')

        Returns:
            Storage path
        """
        if self.use_gcs:
            blob = self.bucket.blob(destination_path)
            blob.upload_from_string(file_content)
            return destination_path
        else:
            # Local filesystem
            local_path = os.path.join(self.local_storage_root, destination_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(file_content)
            return destination_path

    def upload_from_file(self, file_path: str, destination_path: str) -> str:
        """
        Upload a file from local path to storage.

        Args:
            file_path: Local file path
            destination_path: Path in storage

        Returns:
            Storage path
        """
        if self.use_gcs:
            blob = self.bucket.blob(destination_path)
            blob.upload_from_filename(file_path)
            return destination_path
        else:
            # Local filesystem - copy file
            local_path = os.path.join(self.local_storage_root, destination_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            import shutil
            shutil.copy2(file_path, local_path)
            return destination_path

    def download_file(self, source_path: str, destination_path: str) -> str:
        """
        Download a file from storage to local path.

        Args:
            source_path: Path in storage
            destination_path: Local file path to save to

        Returns:
            Local file path
        """
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        if self.use_gcs:
            blob = self.bucket.blob(source_path)
            blob.download_to_filename(destination_path)
        else:
            # Local filesystem - copy file
            local_path = os.path.join(self.local_storage_root, source_path)
            import shutil
            shutil.copy2(local_path, destination_path)

        return destination_path

    def download_as_bytes(self, source_path: str) -> bytes:
        """
        Download a file from storage as bytes.

        Args:
            source_path: Path in storage

        Returns:
            File content as bytes
        """
        if self.use_gcs:
            blob = self.bucket.blob(source_path)
            return blob.download_as_bytes()
        else:
            # Local filesystem
            local_path = os.path.join(self.local_storage_root, source_path)
            with open(local_path, 'rb') as f:
                return f.read()

    def delete_file(self, file_path: str) -> None:
        """
        Delete a file from storage.

        Args:
            file_path: Path in storage
        """
        if self.use_gcs:
            blob = self.bucket.blob(file_path)
            blob.delete()
        else:
            # Local filesystem
            local_path = os.path.join(self.local_storage_root, file_path)
            if os.path.exists(local_path):
                os.remove(local_path)

    def list_files(self, prefix: str = None) -> List[str]:
        """
        List files in storage.

        Args:
            prefix: Optional prefix to filter files (e.g., 'company_123/')

        Returns:
            List of file paths in storage
        """
        if self.use_gcs:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        else:
            # Local filesystem
            import glob
            search_path = os.path.join(self.local_storage_root, prefix or "", "**", "*")
            files = glob.glob(search_path, recursive=True)
            # Return relative paths
            return [os.path.relpath(f, self.local_storage_root) for f in files if os.path.isfile(f)]

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            file_path: Path in storage

        Returns:
            True if file exists, False otherwise
        """
        if self.use_gcs:
            blob = self.bucket.blob(file_path)
            return blob.exists()
        else:
            # Local filesystem
            local_path = os.path.join(self.local_storage_root, file_path)
            return os.path.exists(local_path)

    def download_as_string(self, source_path: str, encoding: str = 'utf-8') -> str:
        """
        Download a file from storage as string.

        Args:
            source_path: Path in storage
            encoding: Text encoding (default: utf-8)

        Returns:
            File content as string
        """
        if self.use_gcs:
            blob = self.bucket.blob(source_path)
            return blob.download_as_text(encoding=encoding)
        else:
            # Local filesystem
            local_path = os.path.join(self.local_storage_root, source_path)
            with open(local_path, 'r', encoding=encoding) as f:
                return f.read()

    @staticmethod
    def get_gcs_path(company_id: int, filename: str) -> str:
        """
        Generate a standardized GCS path for a file.

        Args:
            company_id: Company ID
            filename: Original filename

        Returns:
            GCS path like 'company_123/sales.csv'
        """
        # Sanitize filename to avoid path traversal
        safe_filename = Path(filename).name
        return f"company_{company_id}/{safe_filename}"

    @staticmethod
    def get_analysis_report_path(company_id: str, analysis_id: str) -> str:
        """
        Generate GCS path for analysis report.

        Args:
            company_id: Company ID
            analysis_id: Analysis ID

        Returns:
            GCS path like 'analyses/company_123/analysis_456/report.md'
        """
        return f"analyses/{company_id}/{analysis_id}/report.md"

    @staticmethod
    def get_analysis_dashboard_path(company_id: str, analysis_id: str) -> str:
        """
        Generate GCS path for analysis dashboard.

        Args:
            company_id: Company ID
            analysis_id: Analysis ID

        Returns:
            GCS path like 'analyses/company_123/analysis_456/dashboard.html'
        """
        return f"analyses/{company_id}/{analysis_id}/dashboard.html"

    @staticmethod
    def get_analysis_viz_data_path(company_id: str, analysis_id: str) -> str:
        """
        Generate GCS path for visualization data.

        Args:
            company_id: Company ID
            analysis_id: Analysis ID

        Returns:
            GCS path like 'analyses/company_123/analysis_456/viz_data.json'
        """
        return f"analyses/{company_id}/{analysis_id}/viz_data.json"

    def upload_analysis_outputs(self, company_id: str, analysis_id: str,
                                report_local_path: str = None,
                                dashboard_local_path: str = None,
                                viz_data_local_path: str = None) -> dict:
        """
        Upload analysis output files to storage.

        Args:
            company_id: Company ID
            analysis_id: Analysis ID
            report_local_path: Local path to report.md
            dashboard_local_path: Local path to dashboard.html
            viz_data_local_path: Local path to viz_data.json

        Returns:
            Dictionary with storage paths: {'report_path', 'dashboard_path', 'viz_data_path'}
        """
        result = {}
        storage_type = "GCS" if self.use_gcs else "local storage"

        if report_local_path and os.path.exists(report_local_path):
            storage_path = self.get_analysis_report_path(company_id, analysis_id)
            self.upload_from_file(report_local_path, storage_path)
            result['report_path'] = storage_path
            print(f"[STORAGE] Uploaded report to {storage_type}: {storage_path}")

        if dashboard_local_path and os.path.exists(dashboard_local_path):
            storage_path = self.get_analysis_dashboard_path(company_id, analysis_id)
            self.upload_from_file(dashboard_local_path, storage_path)
            result['dashboard_path'] = storage_path
            print(f"[STORAGE] Uploaded dashboard to {storage_type}: {storage_path}")

        if viz_data_local_path and os.path.exists(viz_data_local_path):
            storage_path = self.get_analysis_viz_data_path(company_id, analysis_id)
            self.upload_from_file(viz_data_local_path, storage_path)
            result['viz_data_path'] = storage_path
            print(f"[STORAGE] Uploaded viz_data to {storage_type}: {storage_path}")

        return result
