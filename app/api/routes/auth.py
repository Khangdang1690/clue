"""Authentication routes (Clerk webhooks and user management)."""

from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_settings
from app.core.config import Settings
from app.core.security import verify_clerk_webhook
from app.services.user_service import UserService
from app.schemas.user import UserResponse, UserCreate, UserUpdate

router = APIRouter()


@router.post("/webhooks/clerk")
async def clerk_webhook(
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_current_settings)
):
    """
    Handle Clerk webhook events.

    Events:
    - user.created: Create new user in database
    - user.updated: Update existing user
    - user.deleted: Delete user from database
    - session.created: Update last sign in
    """
    # Verify webhook
    payload = await verify_clerk_webhook(request, settings.webhook_secret)

    # Process the webhook event
    event_type = payload.get("type")
    data = payload.get("data", {})

    print(f"📬 Received webhook: {event_type}")

    try:
        if event_type == "user.created":
            # Create new user
            user_data = UserCreate(
                id=data.get("id"),
                email=data.get("email_addresses", [{}])[0].get("email_address"),
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                profile_image_url=data.get("profile_image_url"),
            )
            user = UserService.create_user(db, user_data, clerk_metadata=data)
            print(f"✅ User created: {user.email}")

        elif event_type == "user.updated":
            # Update existing user
            user_data = UserUpdate(
                email=data.get("email_addresses", [{}])[0].get("email_address"),
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                profile_image_url=data.get("profile_image_url"),
            )
            user = UserService.update_user(db, data.get("id"), user_data, clerk_metadata=data)
            if user:
                print(f"✅ User updated: {user.email}")
            else:
                print(f"⚠️  User not found: {data.get('id')}")

        elif event_type == "user.deleted":
            # Delete user
            deleted = UserService.delete_user(db, data.get("id"))
            if deleted:
                print(f"✅ User deleted: {data.get('id')}")
            else:
                print(f"⚠️  User not found: {data.get('id')}")

        elif event_type == "session.created":
            # Update last sign in
            user = UserService.update_last_sign_in(db, data.get("user_id"))
            if user:
                print(f"✅ Session created for user: {user.email}")
            else:
                print(f"⚠️  User not found: {data.get('user_id')}")

        else:
            print(f"ℹ️  Unhandled event type: {event_type}")

    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing webhook: {str(e)}"
        )

    return {"status": "success", "event": event_type}


@router.get("/me/{user_id}", response_model=UserResponse)
async def get_me(user_id: str, db: Session = Depends(get_db)):
    """Get current user from database by Clerk user ID."""
    user = UserService.get_user_by_id(db, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found in database. Make sure webhook has been triggered."
        )

    return user


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get user by ID."""
    user = UserService.get_user_by_id(db, user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user
