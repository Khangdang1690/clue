"""Business context collection module."""

from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from src.models.business_context import BusinessContext, Department
from src.utils.llm_client import get_llm
from src.utils.chroma_manager import ChromaDBManager
import json


class ContextCollector:
    """Collects and stores business context information."""

    def __init__(self, chroma_manager: ChromaDBManager):
        """
        Initialize context collector.

        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.chroma_manager = chroma_manager
        self.llm = get_llm(temperature=0.3)

    def collect_basic_context(
        self,
        company_name: str,
        icp: str,
        mission: str,
        current_goal: str,
        success_metrics: List[str]
    ) -> BusinessContext:
        """
        Collect basic business context.

        Args:
            company_name: Name of the company
            icp: Ideal Customer Profile
            mission: Company mission statement
            current_goal: Current business goal
            success_metrics: List of success metrics

        Returns:
            BusinessContext object with basic information
        """
        context = BusinessContext(
            company_name=company_name,
            icp=icp,
            mission=mission,
            current_goal=current_goal,
            success_metrics=success_metrics,
            departments=[]
        )

        # Store in ChromaDB
        self.chroma_manager.store_business_context(
            context.to_dict(),
            context_id="main_context"
        )

        return context

    def add_department_data(
        self,
        department_name: str,
        description: str,
        painpoints: List[str],
        objectives: List[str],
        perspectives: List[str]
    ) -> Department:
        """
        Add department data to the business context.

        Args:
            department_name: Name of the department
            description: Department description
            painpoints: List of pain points
            objectives: List of objectives
            perspectives: List of perspectives

        Returns:
            Department object
        """
        department = Department(
            name=department_name,
            description=description,
            objectives=objectives,
            painpoints=painpoints,
            perspectives=perspectives
        )

        # Update context in ChromaDB
        context_dict = self.chroma_manager.get_business_context()
        if context_dict:
            context = BusinessContext(**context_dict)
            context.departments.append(department)
            self.chroma_manager.store_business_context(context.to_dict())

        return department

    def validate_context(self, context: BusinessContext) -> Dict[str, any]:
        """
        Use LLM to validate and enrich business context.

        Args:
            context: Business context to validate

        Returns:
            Dictionary with validation results and suggestions
        """
        print("🤖 I'm validating business context completeness and identifying gaps...")

        validation_prompt = ChatPromptTemplate.from_template(
            """You are a business analyst expert. Review the following business context and provide:
1. Completeness assessment (0-100%)
2. Missing critical information
3. Suggestions for improvement
4. Potential areas of concern

Business Context:
{context}

Provide your response in JSON format:
{{
    "completeness_score": <number>,
    "missing_information": [<list of missing items>],
    "suggestions": [<list of suggestions>],
    "concerns": [<list of concerns>]
}}
"""
        )

        chain = validation_prompt | self.llm
        response = chain.invoke({"context": context.to_context_string()})

        try:
            # Extract JSON from response
            content = response.content
            # Find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing validation response: {e}")

        return {
            "completeness_score": 50,
            "missing_information": [],
            "suggestions": [],
            "concerns": ["Unable to validate context"]
        }

    def get_current_context(self) -> BusinessContext:
        """
        Retrieve the current business context from ChromaDB.

        Returns:
            Current BusinessContext object
        """
        context_dict = self.chroma_manager.get_business_context()
        if context_dict:
            return BusinessContext(**context_dict)
        raise ValueError("No business context found. Please collect context first.")

    def summarize_context(self) -> str:
        """
        Generate a summary of the current business context using LLM.

        Returns:
            Human-readable summary
        """
        context = self.get_current_context()

        print("🤖 I'm generating executive summary of business context...")

        summary_prompt = ChatPromptTemplate.from_template(
            """As a business analyst, create a concise executive summary of this business context.
Focus on the key aspects that would be important for data analysis and decision-making.

Business Context:
{context}

Provide a clear, professional summary in 2-3 paragraphs.
"""
        )

        chain = summary_prompt | self.llm
        response = chain.invoke({"context": context.to_context_string()})

        return response.content
