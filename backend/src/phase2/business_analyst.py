"""Business Analyst for transforming statistical findings into business insights."""

import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from src.models.challenge import Challenge
from src.models.business_context import BusinessContext
from src.phase2.business_query_engine import BusinessQueryEngine, QueryResult
from src.utils.llm_client import get_llm
import json


@dataclass
class BusinessInsight:
    """Represents a business insight derived from data."""
    finding: str  # The statistical finding or correlation
    business_question: str  # The business question it raises
    query_result: QueryResult  # The data answering the question
    recommendation: str  # Actionable recommendation
    priority: str  # 'high', 'medium', 'low'


class BusinessAnalyst:
    """Transforms statistical findings into business questions and insights."""

    def __init__(self):
        """Initialize the business analyst."""
        self.llm = get_llm(temperature=0.3)
        self.query_engine = BusinessQueryEngine()

    def analyze_for_business(
        self,
        challenge: Challenge,
        statistical_results: Dict,
        data: Dict[str, pd.DataFrame],
        business_context: BusinessContext
    ) -> Dict[str, Any]:
        """
        Transform statistical findings into business insights.

        Args:
            challenge: The business challenge being addressed
            statistical_results: Results from statistical analysis
            data: Available DataFrames
            business_context: Business context

        Returns:
            Dictionary containing:
            - business_questions: List of business questions generated
            - insights: List of BusinessInsight objects
            - key_metrics: Important KPIs discovered
            - visualizations_needed: List of business-oriented visualizations
        """
        result = {
            'business_questions': [],
            'insights': [],
            'key_metrics': {},
            'visualizations_needed': []
        }

        # Generate business questions from statistical findings
        questions = self._generate_business_questions(
            challenge, statistical_results, business_context
        )
        result['business_questions'] = questions

        # Answer each business question with data
        insights = []
        for question_info in questions:
            question = question_info['question']
            priority = question_info['priority']
            finding = question_info['based_on_finding']

            # Query data to answer the question
            query_result = self.query_engine.answer_business_question(
                question, data, challenge.description
            )

            # Generate business recommendation
            recommendation = self._generate_recommendation(
                question, query_result, business_context
            )

            insight = BusinessInsight(
                finding=finding,
                business_question=question,
                query_result=query_result,
                recommendation=recommendation,
                priority=priority
            )
            insights.append(insight)

            # Add visualization if data is available
            if not query_result.data.empty:
                result['visualizations_needed'].append({
                    'question': question,
                    'viz_type': query_result.visualization_recommendation,
                    'data': query_result.data,
                    'title': self._create_viz_title(question),
                    'priority': priority
                })

        result['insights'] = insights

        # Extract key metrics
        result['key_metrics'] = self._extract_key_metrics(
            insights, statistical_results, business_context
        )

        return result

    def _generate_business_questions(
        self,
        challenge: Challenge,
        statistical_results: Dict,
        business_context: BusinessContext
    ) -> List[Dict]:
        """
        Generate business questions based on statistical findings.

        Returns:
            List of dictionaries with questions and metadata
        """
        # Prepare findings summary
        findings_summary = self._summarize_statistical_findings(statistical_results)

        # Prepare dataset relationships summary
        relationships_summary = self._summarize_dataset_relationships(
            statistical_results.get('dataset_relationships', [])
        )

        generation_prompt = ChatPromptTemplate.from_template(
            """As a business analyst, transform these statistical findings into specific business questions that will provide actionable insights.

Challenge: {challenge_title}
Description: {challenge_description}
Department: {department}
Business Goal: {business_goal}

Statistical Findings:
{findings}

{relationships_section}

Generate 3-5 specific business questions that:
1. Are directly answerable with data queries (group by, aggregate, compare, etc.)
2. Provide actionable insights for the business
3. Help address the challenge
4. Are relevant to stakeholders
5. **IMPORTANT**: Consider questions that combine data from multiple datasets when relationships exist

Format as JSON:
[
    {{
        "question": "Specific business question",
        "priority": "high|medium|low",
        "based_on_finding": "Which statistical finding prompted this",
        "expected_insight": "What business insight this will provide",
        "query_approach": "How to query the data (group by X, compare Y, etc.)",
        "requires_join": true/false
    }}
]

Examples of SINGLE-DATASET questions:
- Finding: "Customer retention varies significantly"
  Question: "What is the retention rate by customer segment and which segments are at highest risk?"
  Requires Join: false

Examples of CROSS-DATASET questions (when relationships exist):
- Relationship: employees ↔ sales via employee_id
  Question: "How do sales performance vary by employee demographics (gender, tenure, department)?"
  Requires Join: true

- Relationship: products ↔ orders via product_id, orders ↔ customers via customer_id
  Question: "Which product categories have the highest customer retention rates?"
  Requires Join: true

- Relationship: employees ↔ departments via dept_id
  Question: "What's the average training completion rate by department and how does it correlate with department performance?"
  Requires Join: true
"""
        )

        try:
            chain = generation_prompt | self.llm
            response = chain.invoke({
                'challenge_title': challenge.title,
                'challenge_description': challenge.description,
                'department': ', '.join(challenge.department) if isinstance(challenge.department, list) else challenge.department,
                'business_goal': business_context.current_goal,
                'findings': findings_summary,
                'relationships_section': relationships_summary
            })

            # Parse JSON from response
            content = response.content
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                questions = json.loads(content[start_idx:end_idx])
                return questions

        except Exception as e:
            print(f"Error generating business questions: {e}")

        # Fallback questions based on common patterns
        return self._generate_fallback_questions(statistical_results, challenge)

    def _summarize_statistical_findings(self, statistical_results: Dict) -> str:
        """Summarize statistical findings for LLM context."""
        summary_parts = []

        # Key findings
        if statistical_results.get('key_findings'):
            summary_parts.append("Key Findings:")
            for finding in statistical_results['key_findings'][:5]:
                summary_parts.append(f"- {finding}")

        # Correlations
        if statistical_results.get('correlations'):
            summary_parts.append("\nSignificant Correlations:")
            for pair, value in list(statistical_results['correlations'].items())[:5]:
                summary_parts.append(f"- {pair}: {value:.3f}")

        # Statistical tests
        if statistical_results.get('statistical_tests'):
            summary_parts.append("\nStatistical Tests:")
            for test in statistical_results['statistical_tests'][:3]:
                if hasattr(test, 'interpretation'):
                    summary_parts.append(f"- {test.interpretation}")

        # Causality insights
        if statistical_results.get('causality_insights'):
            summary_parts.append("\nCausality Insights:")
            for insight in statistical_results['causality_insights'][:3]:
                summary_parts.append(f"- {insight}")

        return "\n".join(summary_parts) if summary_parts else "No significant findings"

    def _summarize_dataset_relationships(self, relationships: List[Dict]) -> str:
        """Summarize dataset relationships for LLM context."""
        if not relationships:
            return "Available Dataset Relationships:\nNone - all questions must use a single dataset."

        summary_parts = ["Available Dataset Relationships (for cross-dataset questions):"]

        for rel in relationships:
            dataset1 = rel.get('dataset1', '')
            dataset2 = rel.get('dataset2', '')
            join_key1 = rel.get('join_key1', '')
            join_key2 = rel.get('join_key2', '')
            rel_type = rel.get('relationship_type', 'many-to-many')
            confidence = rel.get('confidence', 'medium')

            # Create human-readable relationship description
            if rel_type == 'one-to-one':
                desc = f"- {dataset1} ↔ {dataset2} via {join_key1}={join_key2} (1:1 relationship, {confidence} confidence)"
            elif rel_type == 'one-to-many':
                desc = f"- {dataset1} → {dataset2} via {join_key1}={join_key2} (1:many, {confidence} confidence)"
            elif rel_type == 'many-to-one':
                desc = f"- {dataset1} ← {dataset2} via {join_key1}={join_key2} (many:1, {confidence} confidence)"
            else:
                desc = f"- {dataset1} ↔ {dataset2} via {join_key1}={join_key2} (many:many, {confidence} confidence)"

            summary_parts.append(desc)

        return "\n".join(summary_parts)

    def _generate_fallback_questions(
        self,
        statistical_results: Dict,
        challenge: Challenge
    ) -> List[Dict]:
        """Generate fallback business questions when LLM fails."""
        questions = []

        # Based on correlations
        if statistical_results.get('correlations'):
            for pair, corr_value in list(statistical_results['correlations'].items())[:2]:
                if abs(corr_value) > 0.5:
                    var1, var2 = pair.split('_vs_')
                    questions.append({
                        'question': f"How does {var1} vary across different levels of {var2}?",
                        'priority': 'high' if abs(corr_value) > 0.7 else 'medium',
                        'based_on_finding': f"Correlation of {corr_value:.3f} between {var1} and {var2}",
                        'expected_insight': f"Understanding the relationship between {var1} and {var2}",
                        'query_approach': f"Group by {var2} categories and analyze {var1}"
                    })

        # Based on key findings
        if statistical_results.get('key_findings'):
            for finding in statistical_results['key_findings'][:2]:
                if 'high variability' in finding.lower():
                    questions.append({
                        'question': "What factors contribute to the high variability in key metrics?",
                        'priority': 'high',
                        'based_on_finding': finding,
                        'expected_insight': "Identify sources of variability for better control",
                        'query_approach': "Segment data and compare variance across groups"
                    })
                elif 'outlier' in finding.lower():
                    questions.append({
                        'question': "What characterizes the outlier cases and should they be addressed differently?",
                        'priority': 'medium',
                        'based_on_finding': finding,
                        'expected_insight': "Understand exceptional cases for targeted strategies",
                        'query_approach': "Filter and analyze outlier characteristics"
                    })

        # Generic question for the challenge
        if not questions:
            questions.append({
                'question': f"What are the key performance indicators for {challenge.title}?",
                'priority': 'high',
                'based_on_finding': "General analysis needed",
                'expected_insight': "Baseline understanding of current state",
                'query_approach': "Calculate summary statistics for relevant metrics"
            })

        return questions

    def _generate_recommendation(
        self,
        question: str,
        query_result: QueryResult,
        business_context: BusinessContext
    ) -> str:
        """Generate actionable recommendation based on query results."""
        if query_result.data.empty:
            return "Insufficient data to generate recommendation"

        recommendation_prompt = ChatPromptTemplate.from_template(
            """Based on this business question and data analysis, provide a specific, actionable recommendation.

Business Question: {question}
Query Summary: {summary}
Business Goal: {business_goal}

Data Preview (first 5 rows):
{data_preview}

Provide ONE specific, actionable recommendation in 1-2 sentences.
Focus on what action to take based on this data insight.
"""
        )

        try:
            # Prepare data preview
            data_preview = query_result.data.head(5).to_string()

            chain = recommendation_prompt | self.llm
            response = chain.invoke({
                'question': question,
                'summary': query_result.summary,
                'business_goal': business_context.current_goal,
                'data_preview': data_preview
            })

            return response.content.strip()

        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return "Review the data insights to identify improvement opportunities"

    def _extract_key_metrics(
        self,
        insights: List[BusinessInsight],
        statistical_results: Dict,
        business_context: BusinessContext
    ) -> Dict[str, Any]:
        """Extract key business metrics from insights."""
        key_metrics = {}

        # Extract metrics from query results
        for insight in insights:
            if insight.priority == 'high' and not insight.query_result.data.empty:
                df = insight.query_result.data

                # Extract numeric summaries
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols[:3]:  # Top 3 numeric columns
                    if col not in key_metrics:
                        key_metrics[col] = {
                            'mean': float(df[col].mean()) if not df[col].isna().all() else 0,
                            'min': float(df[col].min()) if not df[col].isna().all() else 0,
                            'max': float(df[col].max()) if not df[col].isna().all() else 0,
                            'context': insight.business_question
                        }

        # Add success metrics if they match
        for metric in business_context.success_metrics:
            metric_name = metric.split('(')[0].strip().lower().replace(' ', '_')
            if metric_name not in key_metrics:
                # Try to find matching metric in statistical results
                for key in statistical_results.get('key_findings', []):
                    if metric_name in key.lower():
                        key_metrics[metric_name] = {
                            'status': 'tracked',
                            'finding': key
                        }

        return key_metrics

    def _create_viz_title(self, question: str) -> str:
        """Create a clear visualization title from a business question."""
        # Remove question mark and shorten if needed
        title = question.replace('?', '')

        # Shorten common phrases
        replacements = {
            'What is the': '',
            'How does': '',
            'How do': '',
            'What are the': '',
            'Which': ''
        }

        for old, new in replacements.items():
            if title.startswith(old):
                title = title.replace(old, new, 1).strip()
                title = title[0].upper() + title[1:] if title else title

        # Truncate if too long
        if len(title) > 60:
            title = title[:57] + '...'

        return title

    def create_executive_summary(
        self,
        insights: List[BusinessInsight],
        challenge: Challenge,
        business_context: BusinessContext
    ) -> str:
        """
        Create an executive summary of business insights.

        Args:
            insights: List of business insights
            challenge: The challenge being addressed
            business_context: Business context

        Returns:
            Executive summary as formatted text
        """
        summary_prompt = ChatPromptTemplate.from_template(
            """Create a concise executive summary of these business insights.

Challenge: {challenge}
Company: {company}

Top Insights:
{insights_text}

Write a 3-4 paragraph executive summary that:
1. Opens with the key finding
2. Explains the business implications
3. Provides clear next steps
4. Is written for C-level executives

Keep it under 200 words.
"""
        )

        # Prepare insights text
        insights_text = []
        for insight in insights[:5]:  # Top 5 insights
            insights_text.append(f"- {insight.business_question}")
            insights_text.append(f"  Finding: {insight.query_result.summary}")
            insights_text.append(f"  Recommendation: {insight.recommendation}")
            insights_text.append("")

        try:
            chain = summary_prompt | self.llm
            response = chain.invoke({
                'challenge': challenge.title,
                'company': business_context.company_name,
                'insights_text': '\n'.join(insights_text)
            })

            return response.content

        except Exception as e:
            print(f"Error creating executive summary: {e}")
            return "Executive summary generation failed. Please review individual insights."