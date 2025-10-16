"""Challenge identification and prioritization module."""

import heapq
import json
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from src.models.business_context import BusinessContext
from src.models.challenge import Challenge, PriorityLevel
from src.utils.llm_client import get_llm
from src.utils.chroma_manager import ChromaDBManager
import uuid


class ChallengePrioritizer:
    """Identifies and prioritizes business challenges."""

    def __init__(self, chroma_manager: ChromaDBManager):
        """
        Initialize challenge prioritizer.

        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.chroma_manager = chroma_manager
        self.llm = get_llm(temperature=0.4)
        self.challenge_heap: List[Challenge] = []

    def identify_challenges(self, context: BusinessContext) -> List[Challenge]:
        """
        Use LLM to identify challenges from business context.

        Args:
            context: Business context containing painpoints, objectives, and perspectives

        Returns:
            List of identified challenges
        """
        print("🤖 I'm analyzing business context to identify key challenges...")

        identification_prompt = ChatPromptTemplate.from_template(
            """You are a senior business analyst. Analyze the following business context and identify
key challenges that the company is facing. For each challenge, consider:
- Pain points from different departments
- Objectives that are not being met
- Different perspectives that reveal issues
- Success metrics that need improvement

Business Context:
{context}

For each challenge identified, provide:
1. A clear, specific title
2. Detailed description
3. Which department(s) it affects (can be multiple)
4. Related pain points
5. Related objectives
6. Relevant success metrics
7. What data sources might help analyze this (e.g., sales data, customer feedback, operational metrics)

Format your response as a JSON array of challenges:
[
    {{
        "title": "Challenge title",
        "description": "Detailed description",
        "department": ["Department1", "Department2"],
        "related_painpoints": ["painpoint1", "painpoint2"],
        "related_objectives": ["objective1", "objective2"],
        "success_metrics": ["metric1", "metric2"],
        "data_sources_needed": ["source1", "source2"]
    }}
]
"""
        )

        chain = identification_prompt | self.llm
        response = chain.invoke({"context": context.to_context_string()})

        challenges = []
        try:
            content = response.content
            # Extract JSON array
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                challenges_data = json.loads(json_str)

                for challenge_data in challenges_data:
                    # Calculate initial priority score
                    priority_score = self._calculate_priority_score(
                        challenge_data, context
                    )

                    # Ensure department is a list
                    dept = challenge_data.get("department", ["General"])
                    if isinstance(dept, str):
                        dept = [dept]

                    challenge = Challenge(
                        id=str(uuid.uuid4()),
                        title=challenge_data.get("title", "Untitled Challenge"),
                        description=challenge_data.get("description", ""),
                        department=dept,
                        priority_score=priority_score,
                        priority_level=self._score_to_level(priority_score),
                        related_painpoints=challenge_data.get("related_painpoints", []),
                        related_objectives=challenge_data.get("related_objectives", []),
                        success_metrics=challenge_data.get("success_metrics", []),
                        data_sources_needed=challenge_data.get("data_sources_needed", [])
                    )

                    challenges.append(challenge)

        except Exception as e:
            print(f"Error parsing challenges: {e}")
            print(f"Response content: {response.content}")

        return challenges

    def _calculate_priority_score(
        self,
        challenge_data: Dict,
        context: BusinessContext
    ) -> float:
        """
        Calculate priority score for a challenge using LLM.

        Args:
            challenge_data: Challenge information
            context: Business context

        Returns:
            Priority score (0-100)
        """
        print(f"🤖 I'm calculating priority score for challenge '{challenge_data.get('title', 'Untitled')}'...")

        scoring_prompt = ChatPromptTemplate.from_template(
            """You are a business priority analyst. Evaluate this challenge and assign a priority score from 0-100.

Consider:
1. Impact on business goals (0-30 points)
2. Number of departments affected (0-20 points)
3. Alignment with success metrics (0-25 points)
4. Urgency based on pain points (0-25 points)

Business Context:
Company Goal: {goal}
Success Metrics: {metrics}

Challenge:
{challenge}

Provide ONLY a number between 0 and 100 as your response.
"""
        )

        chain = scoring_prompt | self.llm
        response = chain.invoke({
            "goal": context.current_goal,
            "metrics": ", ".join(context.success_metrics),
            "challenge": json.dumps(challenge_data, indent=2)
        })

        try:
            # Extract number from response
            score_str = ''.join(filter(str.isdigit, response.content))
            if score_str:
                score = float(score_str)
                return min(100, max(0, score))
        except Exception as e:
            print(f"Error calculating priority score: {e}")

        # Default score based on number of related items
        return min(100, (
            len(challenge_data.get("related_painpoints", [])) * 10 +
            len(challenge_data.get("related_objectives", [])) * 8 +
            len(challenge_data.get("success_metrics", [])) * 12
        ))

    def _score_to_level(self, score: float) -> PriorityLevel:
        """Convert numeric score to priority level."""
        if score >= 80:
            return PriorityLevel.CRITICAL
        elif score >= 60:
            return PriorityLevel.HIGH
        elif score >= 40:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW

    def prioritize_challenges(self, challenges: List[Challenge]) -> List[Challenge]:
        """
        Create a priority queue of challenges.

        Args:
            challenges: List of challenges to prioritize

        Returns:
            Sorted list of challenges (highest priority first)
        """
        # Store challenges in ChromaDB
        for challenge in challenges:
            self.chroma_manager.store_challenge(
                challenge.model_dump(),
                challenge.id
            )

        # Create max heap (negate scores for max heap behavior)
        self.challenge_heap = challenges.copy()
        heapq.heapify(self.challenge_heap)

        # Return sorted list (highest priority first)
        return sorted(challenges, key=lambda c: c.priority_score, reverse=True)

    def get_next_challenge(self) -> Challenge:
        """
        Get the next highest priority challenge.

        Returns:
            Challenge object

        Raises:
            IndexError: If no challenges remain
        """
        if not self.challenge_heap:
            raise IndexError("No challenges remaining in the queue")

        return heapq.heappop(self.challenge_heap)

    def get_all_challenges_sorted(self) -> List[Challenge]:
        """
        Get all challenges sorted by priority.

        Returns:
            List of challenges sorted by priority (highest first)
        """
        return sorted(self.challenge_heap, key=lambda c: c.priority_score, reverse=True)

    def has_challenges(self) -> bool:
        """Check if there are remaining challenges."""
        return len(self.challenge_heap) > 0

    def generate_challenge_summary(self) -> str:
        """
        Generate a summary of all challenges using LLM.

        Returns:
            Formatted summary of challenges
        """
        challenges = self.get_all_challenges_sorted()

        if not challenges:
            return "No challenges identified."

        print("🤖 I'm generating executive summary of prioritized challenges...")

        summary_prompt = ChatPromptTemplate.from_template(
            """As a business analyst, create a concise summary of these prioritized challenges.
Highlight the most critical issues and their potential business impact.

Challenges:
{challenges}

Provide a professional summary with:
1. Overview of challenge landscape
2. Top 3 priorities and why they matter
3. Overall risk assessment
"""
        )

        challenges_text = "\n\n".join(
            f"Priority {i+1} ({c.priority_level.value.upper()}):\n{c.to_context_string()}"
            for i, c in enumerate(challenges[:5])  # Top 5 challenges
        )

        chain = summary_prompt | self.llm
        response = chain.invoke({"challenges": challenges_text})

        return response.content
