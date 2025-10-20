"""User schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    profile_image_url: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a user."""
    id: str  # Clerk user ID


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    profile_image_url: Optional[str] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: str
    company_id: Optional[str] = None  # Null if user hasn't completed onboarding
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_sign_in_at: Optional[datetime] = None
    clerk_metadata: Optional[dict] = None

    class Config:
        from_attributes = True
