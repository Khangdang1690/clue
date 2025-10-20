"""User service for managing user operations."""

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from src.database.models import User
from app.schemas.user import UserCreate, UserUpdate


class UserService:
    """Service for user-related operations."""

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter_by(id=user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter_by(email=email).first()

    @staticmethod
    def create_user(db: Session, user_data: UserCreate, clerk_metadata: dict = None) -> User:
        """Create a new user."""
        user = User(
            id=user_data.id,
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            profile_image_url=user_data.profile_image_url,
            clerk_metadata=clerk_metadata or {},
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def update_user(db: Session, user_id: str, user_data: UserUpdate, clerk_metadata: dict = None) -> Optional[User]:
        """Update an existing user."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return None

        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        if clerk_metadata:
            user.clerk_metadata = clerk_metadata

        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def delete_user(db: Session, user_id: str) -> bool:
        """Delete a user."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return False

        db.delete(user)
        db.commit()
        return True

    @staticmethod
    def update_last_sign_in(db: Session, user_id: str) -> Optional[User]:
        """Update user's last sign in timestamp."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return None

        user.last_sign_in_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
        return user
