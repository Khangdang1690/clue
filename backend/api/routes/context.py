"""Business context management endpoints."""

from fastapi import APIRouter, HTTPException
from api.models import BusinessContextRequest, BusinessContextResponse
from src.models.business_context import BusinessContext, Department
from src.utils.chroma_manager import ChromaDBManager
from api.websocket import ws_manager


router = APIRouter(prefix="/api", tags=["context"])
chroma_manager = ChromaDBManager()


@router.post("/business-context", response_model=BusinessContextResponse)
async def submit_business_context(context_request: BusinessContextRequest):
    """
    Submit business context information.

    This stores the company's business context including:
    - Company details (name, ICP, mission, goals)
    - Department information (objectives, pain points)
    - Success metrics

    Args:
        context_request: Business context data

    Returns:
        BusinessContextResponse with storage confirmation
    """
    try:
        await ws_manager.broadcast_log("Receiving business context...")

        # Convert request to internal model
        departments = [
            Department(
                name=dept.name,
                description=dept.description,
                objectives=dept.objectives,
                painpoints=dept.painpoints,
                perspectives=dept.perspectives
            )
            for dept in context_request.departments
        ]

        business_context = BusinessContext(
            company_name=context_request.company_name,
            icp=context_request.icp,
            mission=context_request.mission,
            current_goal=context_request.current_goal,
            departments=departments,
            success_metrics=context_request.success_metrics
        )

        # Store in ChromaDB
        context_id = "main_context"
        chroma_manager.store_business_context(
            business_context.to_dict(),
            context_id=context_id
        )

        await ws_manager.broadcast_log(f"✓ Business context stored for: {business_context.company_name}")
        await ws_manager.broadcast_log(f"  - {len(departments)} departments configured")
        await ws_manager.broadcast_log(f"  - {len(context_request.success_metrics)} success metrics defined")

        return BusinessContextResponse(
            status="success",
            message=f"Business context stored successfully for {business_context.company_name}",
            context_id=context_id
        )

    except Exception as e:
        await ws_manager.broadcast_error(f"Failed to store business context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store business context: {str(e)}")


@router.get("/business-context")
async def get_business_context():
    """
    Retrieve the stored business context.

    Returns:
        Business context dictionary or null if not set
    """
    try:
        context = chroma_manager.get_business_context()

        if not context:
            return {"status": "not_found", "context": None}

        return {"status": "success", "context": context}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve business context: {str(e)}")


@router.delete("/business-context")
async def clear_business_context():
    """
    Clear the stored business context.

    Returns:
        Success message
    """
    try:
        # Delete the business context collection
        try:
            chroma_manager.client.delete_collection("business_context")
            await ws_manager.broadcast_log("Business context cleared")
        except:
            pass  # Collection doesn't exist

        return {"status": "success", "message": "Business context cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear business context: {str(e)}")
