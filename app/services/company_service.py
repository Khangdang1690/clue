"""Business logic for company operations."""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from src.database.models import Company, User
from app.schemas.company import CompanyCreate, CompanyUpdate
import uuid


class CompanyService:
    """Service class for company-related operations."""

    @staticmethod
    def create_company(db: Session, company_data: CompanyCreate, user_id: str) -> Company:
        """
        Create a new company and associate it with the user.

        Args:
            db: Database session
            company_data: Company creation data
            user_id: Clerk user ID

        Returns:
            Created company

        Raises:
            ValueError: If company name already exists or user already has a company
        """
        # Check if user already has a company
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        if user.company_id:
            raise ValueError("User already belongs to a company")

        # Create company
        company = Company(
            id=str(uuid.uuid4()),
            name=company_data.name,
            industry=company_data.industry
        )

        try:
            db.add(company)
            db.flush()  # Get the company ID

            # Associate user with company
            user.company_id = company.id

            db.commit()
            db.refresh(company)

            return company

        except IntegrityError as e:
            db.rollback()
            if 'unique constraint' in str(e).lower() and 'name' in str(e).lower():
                raise ValueError(f"Company name '{company_data.name}' already exists")
            raise

    @staticmethod
    def get_company_by_id(db: Session, company_id: str) -> Optional[Company]:
        """Get company by ID."""
        return db.query(Company).filter_by(id=company_id).first()

    @staticmethod
    def get_user_company(db: Session, user_id: str) -> Optional[Company]:
        """Get the company that a user belongs to."""
        user = db.query(User).filter_by(id=user_id).first()
        if not user or not user.company_id:
            return None

        return db.query(Company).filter_by(id=user.company_id).first()

    @staticmethod
    def update_company(db: Session, company_id: str, company_data: CompanyUpdate) -> Company:
        """
        Update company information.

        Args:
            db: Database session
            company_id: Company UUID
            company_data: Company update data

        Returns:
            Updated company

        Raises:
            ValueError: If company not found or name conflict
        """
        company = db.query(Company).filter_by(id=company_id).first()
        if not company:
            raise ValueError(f"Company {company_id} not found")

        # Update fields
        if company_data.name is not None:
            company.name = company_data.name
        if company_data.industry is not None:
            company.industry = company_data.industry

        try:
            db.commit()
            db.refresh(company)
            return company
        except IntegrityError as e:
            db.rollback()
            if 'unique constraint' in str(e).lower() and 'name' in str(e).lower():
                raise ValueError(f"Company name '{company_data.name}' already exists")
            raise
