"""Generate business-focused discovery reports."""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from src.models.discovery_models import DiscoveryResult
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_client import get_llm
from src.discovery.plotly_dashboard_generator import PlotlyDashboardGenerator


class DiscoveryReporter:
    """Generates business-focused markdown reports from discovery results."""

    def __init__(self, output_dir: str = "data/outputs/discovery"):
        """
        Initialize reporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm = get_llm(temperature=0.5)
        self.plotly_dashboard_generator = PlotlyDashboardGenerator()

    def generate_report(
        self,
        result: DiscoveryResult,
        dataset_context: Optional[Dict] = None
    ) -> str:
        """
        Generate business-focused markdown discovery report.

        Args:
            result: DiscoveryResult object
            dataset_context: Optional context from outer agent layer

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"discovery_report_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            # Header - Business-focused
            f.write(f"# Business Insights Report\n\n")
            f.write(f"## {result.dataset_name}\n\n")
            f.write(f"*Discovered on {result.timestamp.strftime('%B %d, %Y at %I:%M %p')}*\n\n")
            f.write("---\n\n")

            # Executive Summary - Business language
            exec_summary = self._generate_executive_summary(result, dataset_context)
            f.write("## Executive Summary\n\n")
            f.write(exec_summary)
            f.write("\n\n")

            # Dataset Context (if available from outer agent)
            if dataset_context and dataset_context.get('domain') != 'Unknown':
                f.write("## What This Data Represents\n\n")
                f.write(f"**Industry/Domain:** {dataset_context.get('domain', 'Unknown')}\n\n")
                f.write(f"**Data Type:** {dataset_context.get('dataset_type', 'Unknown')}\n\n")

                if dataset_context.get('key_entities'):
                    f.write(f"**Key Business Entities:** {', '.join(dataset_context['key_entities'])}\n\n")

                if dataset_context.get('time_period'):
                    f.write(f"**Time Coverage:** {dataset_context['time_period']}\n\n")

                # Business context explanations
                if dataset_context.get('business_context'):
                    f.write("### Key Business Terms\n\n")
                    for term, explanation in dataset_context['business_context'].items():
                        f.write(f"- **{term}:** {explanation}\n")
                    f.write("\n")

                f.write("---\n\n")

            # Key Business Insights (Most Important Section)
            f.write("## 🎯 Key Business Insights\n\n")
            if result.key_insights:
                business_insights = self._convert_to_business_insights(result.key_insights, dataset_context)
                for i, insight in enumerate(business_insights, 1):
                    f.write(f"### Insight {i}\n\n")
                    f.write(f"{insight}\n\n")
            else:
                f.write("*No significant insights were discovered in this dataset.*\n\n")

            f.write("---\n\n")

            # Detailed Findings - Business-focused Q&A
            f.write("## 📊 Detailed Findings\n\n")

            if result.answered_questions:
                # Group questions by business topic
                grouped_questions = self._group_questions_by_topic(result.answered_questions, dataset_context)

                for topic, questions in grouped_questions.items():
                    f.write(f"### {topic}\n\n")

                    for i, aq in enumerate(questions, 1):
                        # Business-friendly question format
                        f.write(f"**Q: {aq.question}**\n\n")

                        # Business-focused answer (rewrite technical answers)
                        business_answer = self._convert_to_business_answer(aq.answer, dataset_context)
                        f.write(f"{business_answer}\n\n")

                        # Embed visualizations if available
                        if hasattr(aq, 'supporting_visualizations') and aq.supporting_visualizations:
                            # Handle supporting_visualizations list (primary field)
                            for viz_path in aq.supporting_visualizations:
                                rel_path = self._get_relative_path(viz_path, str(report_path))
                                f.write(f"\n![Visualization]({rel_path})\n\n")
                        elif hasattr(aq, 'visualization_path') and aq.visualization_path:
                            # Handle single visualization path (fallback)
                            rel_path = self._get_relative_path(aq.visualization_path, str(report_path))
                            f.write(f"\n![Visualization]({rel_path})\n\n")

                        # Confidence indicator (business language)
                        if aq.confidence >= 0.8:
                            confidence_label = "[OK] High confidence"
                        elif aq.confidence >= 0.6:
                            confidence_label = "[WARN] Moderate confidence"
                        else:
                            confidence_label = "[?] Low confidence"

                        f.write(f"*{confidence_label} ({aq.confidence:.0%})*\n\n")
                        f.write("---\n\n")
            else:
                f.write("*No questions could be answered from this dataset.*\n\n")

            # Patterns & Relationships - Business language
            f.write("## 🔗 Key Patterns & Relationships\n\n")
            f.write(self._describe_relationships_business_style(result, dataset_context))
            f.write("\n")

            # What You Should Know - Data quality in business terms
            f.write("## [WARN] What You Should Know\n\n")

            # Data coverage
            missing_rate = result.data_profile.overall_missing_rate
            if missing_rate > 0.2:
                f.write(f"- **Data Completeness:** {(1-missing_rate):.0%} of data is available. ")
                f.write(f"Some insights may be limited due to {missing_rate:.0%} missing information.\n")
            else:
                f.write(f"- **Data Completeness:** Excellent ({(1-missing_rate):.0%} complete)\n")

            # Outliers
            if result.anomalies_detected:
                f.write(f"- **Unusual Values:** Detected {len(result.anomalies_detected)} fields with unusual patterns. ")
                f.write("These may represent special cases or data quality issues.\n")

            # Data quality issues in business terms
            if result.data_quality_issues:
                f.write("\n**Data Quality Notes:**\n\n")
                for issue in result.data_quality_issues[:5]:
                    # Convert technical issues to business language
                    business_issue = self._convert_technical_issue_to_business(issue)
                    f.write(f"- {business_issue}\n")

            f.write("\n---\n\n")

            # Recommended Next Steps (Business actionable)
            f.write("## 💡 Recommended Next Steps\n\n")
            recommendations = self._generate_recommendations(result, dataset_context)
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write("\n---\n\n")

            # Dataset Overview (Technical details moved to end)
            f.write("## 📋 Dataset Overview\n\n")
            f.write(f"- **Total Records:** {result.data_profile.num_rows:,}\n")
            f.write(f"- **Data Fields:** {result.data_profile.num_columns}\n")
            f.write(f"- **Data Size:** {result.data_profile.memory_usage_mb:.1f} MB\n")

            if dataset_context and dataset_context.get('time_period'):
                f.write(f"- **Time Period:** {dataset_context['time_period']}\n")

            f.write(f"\n**Field Types:**\n")
            f.write(f"- Numbers: {len(result.data_profile.numeric_columns)}\n")
            f.write(f"- Categories: {len(result.data_profile.categorical_columns)}\n")
            if result.data_profile.datetime_columns:
                f.write(f"- Dates/Times: {len(result.data_profile.datetime_columns)}\n")

            f.write("\n---\n\n")

            # Footer
            f.write("*This report was automatically generated using AI-powered data discovery.*\n")
            f.write("*For questions or concerns about these insights, please review the underlying data.*\n")

        print(f"\n[REPORT] Business Insights Report saved to: {report_path}")

        # Generate Plotly dashboard if viz_data_path is available
        if result.viz_data_path and Path(result.viz_data_path).exists():
            try:
                plotly_dashboard_path = self.plotly_dashboard_generator.generate_dashboard(
                    viz_data_json_path=result.viz_data_path
                )
                print(f"[REPORT] Plotly Dashboard saved to: {plotly_dashboard_path}")
            except Exception as e:
                print(f"[WARN] Failed to generate Plotly dashboard: {e}")

        return str(report_path)

    def _generate_executive_summary(
        self,
        result: DiscoveryResult,
        dataset_context: Optional[Dict]
    ) -> str:
        """Generate business-focused executive summary using LLM."""

        # Prepare context for LLM
        insights_text = "\n".join(result.key_insights[:5]) if result.key_insights else "No insights available"

        domain_info = ""
        if dataset_context:
            domain_info = f"""
Dataset Type: {dataset_context.get('dataset_type', 'Unknown')}
Domain: {dataset_context.get('domain', 'Unknown')}
Key Entities: {', '.join(dataset_context.get('key_entities', []))}
"""

        prompt = ChatPromptTemplate.from_template(
            """You are writing an executive summary for a business insights report.

Dataset: {dataset_name}
Records: {num_rows:,}
Fields: {num_columns}

{domain_info}

Top Insights Found:
{insights}

Questions Answered: {num_answered}

Write a 2-3 sentence executive summary in business language (not technical).
Focus on WHAT was found and WHY it matters for business decisions.
Use concrete language and avoid jargon like "correlation", "statistical", etc.

Example good summary:
"Analysis of 604,810 financial records from 2009-2024 reveals that revenue growth varies significantly across companies and fiscal periods. The data shows clear patterns in quarterly vs annual reporting that could inform investment strategies. Key findings include identification of consistently high-performing companies and seasonal revenue trends."

Example bad summary:
"Analyzed dataset with 604,810 rows and 10 columns. Found 16 relationships with correlation > 0.5. Performed statistical tests with p < 0.05."

Write the summary:"""
        )

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "dataset_name": result.dataset_name,
                "num_rows": result.data_profile.num_rows,
                "num_columns": result.data_profile.num_columns,
                "domain_info": domain_info,
                "insights": insights_text,
                "num_answered": len(result.answered_questions)
            })

            return response.content.strip()

        except Exception as e:
            # Fallback summary
            return (
                f"Analysis of {result.data_profile.num_rows:,} records across {result.data_profile.num_columns} fields "
                f"revealed {len(result.answered_questions)} key insights about patterns and relationships in the data."
            )

    def _convert_to_business_insights(
        self,
        technical_insights: list,
        dataset_context: Optional[Dict]
    ) -> list:
        """Convert technical insights to business language."""

        business_insights = []

        for insight in technical_insights:
            # Remove confidence prefix if exists
            clean_insight = insight
            if ']' in insight:
                clean_insight = insight.split(']', 1)[1].strip()

            # Use LLM to rewrite in business language
            business_version = self._rewrite_as_business_insight(clean_insight, dataset_context)
            business_insights.append(business_version)

        return business_insights

    def _rewrite_as_business_insight(
        self,
        technical_insight: str,
        dataset_context: Optional[Dict]
    ) -> str:
        """Use LLM to rewrite technical insight as business insight."""

        domain = dataset_context.get('domain', 'general business') if dataset_context else 'general business'

        prompt = ChatPromptTemplate.from_template(
            """Rewrite this technical data finding as a business insight.

Domain: {domain}
Technical Finding: {finding}

Rules:
1. Remove technical jargon (correlation, variance, distribution, etc.)
2. Explain WHAT it means and WHY it matters for business
3. Use concrete examples if possible
4. Keep it 1-2 sentences
5. Start with the business impact, not the technical detail

Example:
Technical: "Strong correlation (r=0.85) between customer_tier and purchase_amount"
Business: "Premium tier customers spend significantly more per purchase, suggesting targeted upselling strategies could increase revenue from standard tier customers."

Technical: "Revenue shows positive trend (p<0.01) over fiscal years 2009-2024"
Business: "The company has demonstrated consistent revenue growth over the past 15 years, indicating strong market position and sustainable business model."

Now rewrite this finding:"""
        )

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "domain": domain,
                "finding": technical_insight
            })

            return response.content.strip()

        except Exception:
            # Return original if LLM fails
            return technical_insight

    def _convert_to_business_answer(
        self,
        technical_answer: str,
        dataset_context: Optional[Dict]
    ) -> str:
        """Convert technical answer to business language."""

        # If answer is already business-friendly, return as-is
        if len(technical_answer) < 100 and not any(word in technical_answer.lower() for word in ['correlation', 'variance', 'standard deviation', 'p-value']):
            return technical_answer

        domain = dataset_context.get('domain', 'general') if dataset_context else 'general'

        prompt = ChatPromptTemplate.from_template(
            """Rewrite this technical answer in business language.

Domain: {domain}
Technical Answer: {answer}

Rules:
1. Keep the key numbers and facts
2. Remove statistical jargon
3. Explain what it means for business decisions
4. Keep it concise (2-3 sentences max)

Rewrite:"""
        )

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "domain": domain,
                "answer": technical_answer
            })

            return response.content.strip()

        except Exception:
            return technical_answer

    def _group_questions_by_topic(
        self,
        answered_questions: list,
        dataset_context: Optional[Dict]
    ) -> Dict[str, list]:
        """Group questions by business topic."""

        # Default grouping
        groups = {
            "Performance & Trends": [],
            "Comparisons & Segmentation": [],
            "Patterns & Relationships": [],
            "Other Findings": []
        }

        for aq in answered_questions:
            question_lower = aq.question.lower()

            # Classify by keywords
            if any(word in question_lower for word in ['trend', 'over time', 'growth', 'change', 'increase', 'decrease']):
                groups["Performance & Trends"].append(aq)
            elif any(word in question_lower for word in ['compare', 'difference', 'between', 'vs', 'across', 'by']):
                groups["Comparisons & Segmentation"].append(aq)
            elif any(word in question_lower for word in ['relationship', 'correlate', 'associate', 'pattern', 'predict']):
                groups["Patterns & Relationships"].append(aq)
            else:
                groups["Other Findings"].append(aq)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _describe_relationships_business_style(
        self,
        result: DiscoveryResult,
        dataset_context: Optional[Dict]
    ) -> str:
        """Describe relationships in business terms."""

        # Knowledge graph not used in autonomous approach
        return "*Patterns and relationships are described in the insights above.*\n"

    def _convert_technical_issue_to_business(self, technical_issue: str) -> str:
        """Convert technical data quality issue to business language."""

        # Simple replacements
        issue = technical_issue
        issue = issue.replace("missing values", "incomplete information")
        issue = issue.replace("outliers", "unusual values")
        issue = issue.replace("high cardinality", "many unique values")
        issue = issue.replace("skewed distribution", "uneven distribution")

        return issue

    def _generate_recommendations(
        self,
        result: DiscoveryResult,
        dataset_context: Optional[Dict]
    ) -> list:
        """Generate business-actionable recommendations."""

        recommendations = []

        # Based on dataset context
        if dataset_context:
            if dataset_context.get('typical_use_cases'):
                use_case = dataset_context['typical_use_cases'][0]
                recommendations.append(f"Use these insights for {use_case.lower()}")

        # Based on findings
        if len(result.answered_questions) > 5:
            recommendations.append("Review the detailed findings to identify opportunities for business optimization")

        # Based on data quality
        if result.data_profile.overall_missing_rate > 0.3:
            recommendations.append("Improve data collection processes to reduce missing information and increase insight accuracy")

        # Default recommendations
        if not recommendations:
            recommendations.append("Share these findings with stakeholders to inform strategic decisions")
            recommendations.append("Consider collecting additional data to validate and expand on these insights")

        return recommendations[:5]  # Max 5 recommendations

    def _get_relative_path(self, viz_path: str, report_path: str) -> str:
        """
        Get relative path from report to visualization file.

        Args:
            viz_path: Absolute path to visualization file
            report_path: Absolute path to report file

        Returns:
            Relative path from report to visualization
        """
        import os
        try:
            # Convert to Path objects
            viz_path_obj = Path(viz_path).resolve()
            report_path_obj = Path(report_path).resolve()

            # Get relative path
            rel_path = os.path.relpath(viz_path_obj, report_path_obj.parent)

            # Convert to forward slashes for markdown compatibility
            rel_path = rel_path.replace('\\', '/')

            return rel_path
        except Exception as e:
            # Fallback to absolute path if relative path fails
            return viz_path

    def generate_html_dashboard(
        self,
        result: DiscoveryResult,
        dataset_context: Optional[Dict] = None
    ) -> str:
        """
        Generate interactive HTML dashboard from discovery results.

        Args:
            result: DiscoveryResult object
            dataset_context: Optional context from outer agent layer

        Returns:
            Path to generated HTML dashboard
        """
        return self.dashboard_generator.generate_dashboard(result, dataset_context)
