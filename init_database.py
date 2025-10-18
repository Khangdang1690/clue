"""Ensure database tables exist (Docker handles DB creation)."""

import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from src.database.connection import DatabaseManager


def ensure_tables_exist():
    """Ensure all required tables exist in the database."""
    print("\n" + "="*80)
    print("DATABASE TABLE VERIFICATION")
    print("="*80)

    try:
        # Wait for Docker database to be ready
        print("\n[1] Connecting to PostgreSQL...")
        retries = 5
        while retries > 0:
            try:
                DatabaseManager.initialize()
                print("    ✓ Connected successfully")
                break
            except Exception as e:
                retries -= 1
                if retries > 0:
                    print(f"    Waiting for database... ({retries} retries left)")
                    time.sleep(2)
                else:
                    raise e

        # Create tables if they don't exist (idempotent)
        print("\n[2] Ensuring tables exist...")
        DatabaseManager.create_all_tables()
        print("    ✓ All tables ready")

        print("\n" + "="*80)
        print("[SUCCESS] DATABASE READY")
        print("="*80)

        print("\nYou can now run:")
        print("  python test_etl_pipeline.py")

        return True

    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        print("\nMake sure Docker is running:")
        print("  docker-compose up -d")
        return False


if __name__ == "__main__":
    ensure_tables_exist()