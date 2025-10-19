"""Script to clear all data from the PostgreSQL database."""

from src.database.connection import DatabaseManager
from src.database.models import (
    Company, Dataset, TableRelationship, ColumnMetadata,
    KPIDefinition, AnalysisSession, InsightPattern
)

def clear_all_data():
    """Clear all data from all tables in the correct order."""

    print("="*60)
    print("CLEARING ALL DATABASE DATA")
    print("="*60)

    # Initialize database connection
    DatabaseManager.initialize()

    with DatabaseManager.get_session() as session:
        try:
            # Delete in order of dependencies (children first, parents last)

            # 1. Delete analysis-related tables
            deleted = session.query(AnalysisSession).delete()
            print(f"Deleted {deleted} analysis sessions")

            deleted = session.query(InsightPattern).delete()
            print(f"Deleted {deleted} insight patterns")

            # 2. Delete KPI definitions
            deleted = session.query(KPIDefinition).delete()
            print(f"Deleted {deleted} KPI definitions")

            # 3. Delete relationships (references datasets)
            deleted = session.query(TableRelationship).delete()
            print(f"Deleted {deleted} table relationships")

            # 4. Delete column metadata (references datasets)
            deleted = session.query(ColumnMetadata).delete()
            print(f"Deleted {deleted} column metadata entries")

            # 5. Delete datasets (references companies)
            deleted = session.query(Dataset).delete()
            print(f"Deleted {deleted} datasets")

            # 6. Finally delete companies
            deleted = session.query(Company).delete()
            print(f"Deleted {deleted} companies")

            # Commit all deletions
            session.commit()

            # Also drop all cleaned data tables
            print("\nDropping all cleaned_* tables...")
            from sqlalchemy import text

            # Get all table names starting with 'test_company' or 'cleaned_'
            result = session.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND (table_name LIKE 'cleaned_%' OR table_name LIKE 'test_company_%')
            """))

            tables = result.fetchall()
            for (table_name,) in tables:
                session.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
                print(f"Dropped table: {table_name}")

            session.commit()

            print("\n" + "="*60)
            print("SUCCESS: All data deleted from database")
            print("="*60)

        except Exception as e:
            session.rollback()
            print(f"\nERROR: Failed to clear database: {e}")
            import traceback
            traceback.print_exc()

            # Try alternative method - drop all cleaned tables
            try:
                print("\nTrying to drop all cleaned_* tables...")
                from sqlalchemy import text

                # Get all table names starting with 'cleaned_'
                result = session.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'cleaned_%'
                """))

                tables = result.fetchall()
                for (table_name,) in tables:
                    session.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
                    print(f"Dropped table: {table_name}")

                session.commit()
                print("\nDropped all cleaned_* tables")

            except Exception as e2:
                print(f"Failed to drop cleaned tables: {e2}")
                session.rollback()


if __name__ == "__main__":
    clear_all_data()