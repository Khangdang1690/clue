"""Business context data model."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Department(BaseModel):
    """Represents a department in the company."""
    name: str
    description: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    painpoints: List[str] = Field(default_factory=list)
    perspectives: List[str] = Field(default_factory=list)


class BusinessContext(BaseModel):
    """Complete business context information."""
    company_name: str
    icp: str = Field(description="Ideal Customer Profile")
    mission: str
    current_goal: str
    departments: List[Department] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for ChromaDB storage."""
        return self.model_dump()

    def to_context_string(self) -> str:
        """Convert to formatted string for LLM context."""
        context = f"""
Business Context:
- Company: {self.company_name}
- ICP: {self.icp}
- Mission: {self.mission}
- Current Goal: {self.current_goal}

Success Metrics:
{chr(10).join(f'- {metric}' for metric in self.success_metrics)}

Departments:
"""
        for dept in self.departments:
            context += f"\n{dept.name}:"
            if dept.description:
                context += f"\n  Description: {dept.description}"
            if dept.objectives:
                context += f"\n  Objectives: {', '.join(dept.objectives)}"
            if dept.painpoints:
                context += f"\n  Pain Points: {', '.join(dept.painpoints)}"
            if dept.perspectives:
                context += f"\n  Perspectives: {', '.join(dept.perspectives)}"

        return context
