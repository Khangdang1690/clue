"""
Database migration script to update User and Company schema.

Changes:
1. Add company_id column to users table
2. Add industry column to companies table
3. Remove user_id column from companies table
"""

from src.database.connection import DatabaseManager
from sqlalchemy import text


def run_migration():
    """Run the schema migration."""

    print("Starting database migration...")
    print("="*80)

    # Initialize database connection
    DatabaseManager.initialize()

    with DatabaseManager.get_session() as session:
        try:
            # Step 1: Add company_id column to users table (nullable initially)
            print("\n[1/4] Adding company_id column to users table...")
            session.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS company_id VARCHAR;
            """))
            session.commit()
            print("[OK] Added company_id column to users")

            # Step 2: Add industry column to companies table
            print("\n[2/4] Adding industry column to companies table...")
            session.execute(text("""
                ALTER TABLE companies
                ADD COLUMN IF NOT EXISTS industry VARCHAR;
            """))
            session.commit()
            print("[OK] Added industry column to companies")

            # Step 3: Migrate existing data (if user_id exists in companies)
            print("\n[3/4] Migrating existing data...")
            result = session.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'companies' AND column_name = 'user_id';
            """))

            if result.fetchone():
                # user_id column exists, migrate data
                print("  Found user_id column in companies table, migrating data...")

                # Set company_id for users based on old user_id in companies
                session.execute(text("""
                    UPDATE users
                    SET company_id = companies.id
                    FROM companies
                    WHERE companies.user_id = users.id;
                """))
                session.commit()
                print("  [OK] Migrated user-company relationships")

                # Drop the old user_id column
                print("  Dropping old user_id column from companies...")
                session.execute(text("""
                    ALTER TABLE companies
                    DROP COLUMN IF EXISTS user_id;
                """))
                session.commit()
                print("  [OK] Dropped user_id column")
            else:
                print("  No user_id column found, skipping data migration")

            # Step 4: Add foreign key constraint
            print("\n[4/4] Adding foreign key constraint...")
            session.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints
                        WHERE constraint_name = 'users_company_id_fkey'
                    ) THEN
                        ALTER TABLE users
                        ADD CONSTRAINT users_company_id_fkey
                        FOREIGN KEY (company_id) REFERENCES companies(id);
                    END IF;
                END $$;
            """))
            session.commit()
            print("[OK] Added foreign key constraint")

            print("\n" + "="*80)
            print("SUCCESS: Migration completed successfully!")
            print("="*80)

        except Exception as e:
            print(f"\nERROR: Migration failed: {e}")
            session.rollback()
            raise


if __name__ == "__main__":
    run_migration()
