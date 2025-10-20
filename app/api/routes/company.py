"""Company management API routes."""

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional
from app.api.deps import get_db
from app.schemas.company import CompanyCreate, CompanyUpdate, CompanyResponse
from app.services.company_service import CompanyService

router = APIRouter()


def get_user_id_from_header(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract user ID from Clerk session token.

    For now, we'll expect the frontend to send the Clerk user ID in a header.
    In production, you should verify the Clerk session token.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # For MVP: Frontend sends "Bearer <clerk_user_id>"
    # In production: Verify Clerk JWT token
    if authorization.startswith("Bearer "):
        user_id = authorization.replace("Bearer ", "")
        return user_id

    raise HTTPException(status_code=401, detail="Invalid authorization header")


@router.post("/create", response_model=CompanyResponse, status_code=201)
async def create_company(
    company_data: CompanyCreate,
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Create a new company during onboarding.

    This endpoint is called after user signs up with Clerk.
    It creates a company and associates the user with it.
    """
    try:
        company = CompanyService.create_company(db, company_data, user_id)
        return company
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create company: {str(e)}")


@router.get("/me", response_model=CompanyResponse)
async def get_my_company(
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Get the company that the current user belongs to.

    Returns 404 if user hasn't created/joined a company yet.
    """
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(status_code=404, detail="User has not created a company yet")

    return company


@router.put("/update", response_model=CompanyResponse)
async def update_company(
    company_data: CompanyUpdate,
    user_id: str = Depends(get_user_id_from_header),
    db: Session = Depends(get_db)
):
    """
    Update company information.

    Only users who belong to the company can update it.
    """
    # Get user's company
    company = CompanyService.get_user_company(db, user_id)
    if not company:
        raise HTTPException(status_code=404, detail="User has not created a company yet")

    try:
        updated_company = CompanyService.update_company(db, company.id, company_data)
        return updated_company
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update company: {str(e)}")
