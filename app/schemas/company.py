"""Pydantic schemas for company management."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class CompanyCreate(BaseModel):
    """Schema for creating a new company during onboarding."""
    name: str = Field(..., min_length=1, max_length=255, description="Company name")
    industry: str = Field(..., min_length=1, description="Industry/sector")


class CompanyUpdate(BaseModel):
    """Schema for updating company information."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    industry: Optional[str] = None


class CompanyResponse(BaseModel):
    """Schema for company response."""
    id: str
    name: str
    industry: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
