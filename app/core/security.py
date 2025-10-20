"""Security utilities."""

from fastapi import HTTPException, Request
from svix.webhooks import Webhook, WebhookVerificationError


async def verify_clerk_webhook(request: Request, webhook_secret: str) -> dict:
    """
    Verify Clerk webhook signature.

    Args:
        request: FastAPI request object
        webhook_secret: Clerk webhook secret

    Returns:
        Verified webhook payload

    Raises:
        HTTPException: If verification fails
    """
    if not webhook_secret:
        raise HTTPException(
            status_code=500,
            detail="WEBHOOK_SECRET not configured"
        )

    # Get headers
    svix_id = request.headers.get("svix-id")
    svix_timestamp = request.headers.get("svix-timestamp")
    svix_signature = request.headers.get("svix-signature")

    if not svix_id or not svix_timestamp or not svix_signature:
        raise HTTPException(
            status_code=400,
            detail="Missing svix headers"
        )

    # Get raw body
    body = await request.body()

    # Verify webhook signature
    wh = Webhook(webhook_secret)
    try:
        payload = wh.verify(body, {
            "svix-id": svix_id,
            "svix-timestamp": svix_timestamp,
            "svix-signature": svix_signature,
        })
        return payload
    except WebhookVerificationError as e:
        print(f"❌ Webhook verification failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Webhook verification failed"
        )
