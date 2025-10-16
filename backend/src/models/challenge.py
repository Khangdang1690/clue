"""Challenge data model with priority queue support."""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class PriorityLevel(str, Enum):
    """Priority levels for challenges."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Challenge(BaseModel):
    """Represents a business challenge identified from analysis."""
    id: str
    title: str
    description: str
    department: List[str]
    priority_score: float = Field(ge=0, le=100, description="0-100 priority score")
    priority_level: PriorityLevel
    related_painpoints: List[str] = Field(default_factory=list)
    related_objectives: List[str] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)
    data_sources_needed: List[str] = Field(default_factory=list)

    def __lt__(self, other: 'Challenge') -> bool:
        """Compare challenges by priority score (higher is better)."""
        return self.priority_score > other.priority_score

    def __le__(self, other: 'Challenge') -> bool:
        return self.priority_score >= other.priority_score

    def to_context_string(self) -> str:
        """Convert to formatted string for LLM context."""
        departments = ", ".join(self.department) if isinstance(self.department, list) else self.department
        return f"""
Challenge: {self.title}
Priority: {self.priority_level.value.upper()} (Score: {self.priority_score})
Department(s): {departments}
Description: {self.description}

Related Pain Points:
{chr(10).join(f'- {pp}' for pp in self.related_painpoints)}

Related Objectives:
{chr(10).join(f'- {obj}' for obj in self.related_objectives)}

Success Metrics:
{chr(10).join(f'- {metric}' for metric in self.success_metrics)}

Data Sources Needed:
{chr(10).join(f'- {ds}' for ds in self.data_sources_needed)}
"""
