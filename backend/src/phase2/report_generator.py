"""Report generation module for analytical and business insight reports."""

from typing import List, Dict
from datetime import datetime
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from src.models.business_context import BusinessContext
from src.models.challenge import Challenge
from src.models.analysis_result import AnalysisResult
from src.models.report import AnalyticalReport, BusinessInsightReport, ReportSection
from src.utils.llm_client import get_llm


class ReportGenerator:
    """Generates professional analytical and business insight reports."""

    def __init__(self, output_directory: str = "data/outputs/reports"):
        """
        Initialize report generator.

        Args:
            output_directory: Directory to save reports
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.llm = get_llm(temperature=0.5)

    def generate_analytical_report(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult]
    ) -> AnalyticalReport:
        """
        Generate a comprehensive analytical report following data analyst standards.

        Args:
            business_context: Business context information
            analysis_results: List of analysis results

        Returns:
            AnalyticalReport object
        """
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            business_context, analysis_results
        )

        # Generate methodology section
        methodology = self._generate_methodology(analysis_results)

        # Extract data sources
        data_sources = set()
        for result in analysis_results:
            data_sources.update(result.data_sources_used)

        # Generate main sections
        sections = self._generate_analytical_sections(analysis_results)

        # Extract key findings
        all_findings = []
        for result in analysis_results:
            all_findings.extend(result.key_findings)

        # Generate synthesized key findings
        key_findings = self._synthesize_key_findings(all_findings, business_context)

        # Generate limitations
        limitations = self._generate_limitations(analysis_results)

        report = AnalyticalReport(
            title=f"Analytical Report: {business_context.company_name}",
            company_name=business_context.company_name,
            executive_summary=executive_summary,
            methodology=methodology,
            data_sources=list(data_sources),
            analysis_results=analysis_results,
            sections=sections,
            key_findings=key_findings,
            limitations=limitations
        )

        # Save report
        self._save_report(report, "analytical_report")

        return report

    def generate_business_insight_report(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult],
        challenges_addressed: List[Challenge]
    ) -> BusinessInsightReport:
        """
        Generate a business-focused insight report.

        Args:
            business_context: Business context information
            analysis_results: List of analysis results
            challenges_addressed: List of challenges that were addressed

        Returns:
            BusinessInsightReport object
        """
        # Generate executive summary
        executive_summary = self._generate_business_executive_summary(
            business_context, challenges_addressed, analysis_results
        )

        # Generate strategic insights
        strategic_insights = self._generate_strategic_insights(
            business_context, analysis_results
        )

        # List challenges addressed
        challenges_list = [
            f"{c.title} (Priority: {c.priority_level.value})"
            for c in challenges_addressed
        ]

        # Identify opportunities
        opportunities = self._identify_opportunities(
            business_context, analysis_results
        )

        # Generate action items
        action_items = self._generate_action_items(
            business_context, analysis_results, challenges_addressed
        )

        # Generate expected impact
        expected_impact = self._generate_expected_impact(
            business_context, action_items
        )

        report = BusinessInsightReport(
            title=f"Business Insights: {business_context.company_name}",
            company_name=business_context.company_name,
            executive_summary=executive_summary,
            strategic_insights=strategic_insights,
            challenges_addressed=challenges_list,
            opportunities_identified=opportunities,
            action_items=action_items,
            expected_impact=expected_impact,
            analysis_results=analysis_results
        )

        # Save report
        self._save_report(report, "business_insight_report")

        return report

    def _generate_executive_summary(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult]
    ) -> str:
        """Generate executive summary for analytical report."""
        # Gather specific metrics from analysis
        challenge_titles = [r.challenge_title for r in analysis_results]
        total_tests = sum(len(r.statistical_tests) for r in analysis_results)
        significant_tests = sum(1 for r in analysis_results for t in r.statistical_tests if t.is_significant)
        total_data_sources = len(set(source for r in analysis_results for source in r.data_sources_used))

        print("🤖 I'm generating analytical report executive summary...")

        summary_prompt = ChatPromptTemplate.from_template(
            """As a senior data analyst, write a concise executive summary for this analytical report.

Business Context:
Company: {company}
Goal: {goal}
Key Success Metrics: {metrics}

Analysis Scope:
- Challenges Analyzed: {challenges}
- Data Sources: {num_sources} distinct sources
- Statistical Tests: {total_tests} performed ({significant_tests} statistically significant)

Key Findings:
{findings}

Write a professional executive summary (2-3 paragraphs) that:
1. States the purpose tied to the specific business goal stated above
2. Quantifies the scope of analysis (specific challenges, data sources, tests)
3. Highlights the most significant findings with SPECIFIC NUMBERS and METRICS
4. Indicates business relevance and urgency

IMPORTANT:
- Use actual numbers from findings (percentages, ranges, counts)
- Reference the Key Success Metrics targets when relevant
- Be specific about which challenges were analyzed
- Avoid generic statements - use concrete data points
"""
        )

        # Collect top findings with numbers
        all_findings = []
        for result in analysis_results[:5]:
            all_findings.extend(result.key_findings[:4])

        chain = summary_prompt | self.llm
        response = chain.invoke({
            "company": business_context.company_name,
            "goal": business_context.current_goal,
            "metrics": ", ".join(business_context.success_metrics[:5]),
            "challenges": "; ".join(challenge_titles),
            "num_sources": total_data_sources,
            "total_tests": total_tests,
            "significant_tests": significant_tests,
            "findings": "\n".join(f"- {f}" for f in all_findings[:15])
        })

        return response.content

    def _generate_methodology(self, analysis_results: List[AnalysisResult]) -> str:
        """Generate methodology section based on actual analysis performed."""
        # Build methodology from actual tests and data
        methodology = "## Methodology\n\n"
        methodology += "This analysis employed a systematic ETL (Extract, Transform, Load) approach combined with statistical analysis and data visualization.\n\n"

        # Data sources and ETL
        all_sources = set()
        for result in analysis_results:
            all_sources.update(result.data_sources_used)

        methodology += "### Data Sources\n\n"
        for source in sorted(all_sources):
            methodology += f"- {source}\n"

        # Statistical methods actually used
        test_types = set()
        test_details = {}

        for result in analysis_results:
            for test in result.statistical_tests:
                test_types.add(test.test_name)
                if test.test_name not in test_details:
                    test_details[test.test_name] = {
                        'count': 0,
                        'significant': 0,
                        'variables': set()
                    }
                test_details[test.test_name]['count'] += 1
                if test.is_significant:
                    test_details[test.test_name]['significant'] += 1
                if 'variables' in test.parameters:
                    for var in test.parameters['variables']:
                        test_details[test.test_name]['variables'].add(var)

        if test_types:
            methodology += "\n### Statistical Methods\n\n"
            methodology += "The following statistical tests were performed:\n\n"
            for test_name in sorted(test_types):
                details = test_details[test_name]
                methodology += f"- **{test_name}**: {details['count']} tests performed"
                if details['significant'] > 0:
                    methodology += f" ({details['significant']} yielded significant results at α=0.05)"
                methodology += "\n"

        # Visualization approaches
        viz_types = set()
        for result in analysis_results:
            for viz in result.visualizations:
                viz_types.add(viz.viz_type)

        if viz_types:
            methodology += "\n### Visualization Approaches\n\n"
            methodology += "Visualizations were generated to support data interpretation:\n\n"
            for viz_type in sorted(viz_types):
                methodology += f"- {viz_type.capitalize()} charts\n"

        return methodology

    def _generate_analytical_sections(
        self,
        analysis_results: List[AnalysisResult]
    ) -> List[ReportSection]:
        """Generate main analytical sections."""
        sections = []

        # Create a section for each analysis
        for idx, result in enumerate(analysis_results, 1):
            # Main section for this challenge
            section_content = f"""
## Analysis Overview

**Challenge:** {result.challenge_title}

### ETL Process

**Extraction:** {result.extraction_summary}

**Transformation:** {result.transformation_summary}

**Loading:** {result.load_summary}

### Statistical Analysis

"""
            # Add statistical test results (filter weak results)
            filtered_tests = []
            skipped_count = 0

            for test in result.statistical_tests:
                # Filter out weak regression results
                if test.test_name == "Linear Regression":
                    r_squared = test.parameters.get('r_squared', 0)
                    if r_squared < 0.3 and not test.is_significant:
                        skipped_count += 1
                        continue  # Skip weak and non-significant regressions

                # Filter out non-significant tests with high p-values
                if not test.is_significant and test.p_value > 0.1:
                    skipped_count += 1
                    continue  # Skip clearly non-significant tests

                filtered_tests.append(test)

            # Log filtering results
            if skipped_count > 0:
                print(f"  ℹ️  Filtered out {skipped_count} weak/non-significant tests for '{result.challenge_title}'")

            # Add test count header
            section_content += f"{len(filtered_tests)} statistical tests were performed"
            if skipped_count > 0:
                section_content += f" ({skipped_count} weak results filtered)"
            section_content += ":\n\n"

            for test in filtered_tests:
                section_content += f"""
**{test.test_name}**
- Test Statistic: {test.test_statistic:.4f}
- P-value: {test.p_value:.4f}
- Significance: {'Yes' if test.is_significant else 'No'} (α = {test.significance_level})
- Interpretation: {test.interpretation}

"""

            # Add key findings
            section_content += """
### Key Findings

"""
            for finding in result.key_findings:
                section_content += f"- {finding}\n"

            # Add correlations if any
            if result.correlations:
                section_content += """
### Correlation Analysis

Significant correlations identified:

"""
                for corr_pair, corr_value in list(result.correlations.items())[:10]:
                    section_content += f"- {corr_pair}: {corr_value:.3f}\n"

            # Add causality insights
            if result.causality_insights:
                section_content += """
### Causality Insights

"""
                for insight in result.causality_insights:
                    section_content += f"- {insight}\n"

            # Add visualizations
            if result.visualizations:
                section_content += f"""
### Visualizations

{len(result.visualizations)} visualizations were generated to support this analysis.

"""
                for viz in result.visualizations:
                    section_content += f"- {viz.title}: {viz.description}\n"

            sections.append(ReportSection(
                title=f"Analysis {idx}: {result.challenge_title}",
                content=section_content
            ))

        return sections

    def _extract_raw_statistics_from_findings(self, all_findings: List[str]) -> str:
        """
        Extract numeric statistics mentioned in findings to prevent hallucination.
        Parses findings text to identify metrics and their values.
        """
        import re

        stats_lines = []
        seen_metrics = set()

        # Pattern to match metric mentions with values
        # E.g., "revenue (range: 13396.30 to 91539.95)" or "conversion rate: 4-5%"
        patterns = [
            r'(\w+(?:_\w+)*)\s+contains potential outliers \(range:\s*([\d.]+)\s+to\s*([\d.]+)\)',
            r'(\w+(?:_\w+)*)\s+shows high variability \(std:\s*([\d.]+),\s*mean:\s*([\d.]+)\)',
            r'(\w+(?:_\w+)*)\s+shows time-based patterns',
            r'(\w+(?:_\w+)*):\s*([\d.]+)%?\s*to\s*([\d.]+)%?',
        ]

        for finding in all_findings:
            for pattern in patterns:
                matches = re.finditer(pattern, finding)
                for match in matches:
                    metric = match.group(1)
                    if metric not in seen_metrics:
                        seen_metrics.add(metric)
                        # Extract the full finding about this metric
                        if 'range:' in finding:
                            stats_lines.append(f"- {metric}: {match.group(0)}")
                        elif 'variability' in finding:
                            stats_lines.append(f"- {metric}: {match.group(0)}")

        if stats_lines:
            return "Raw Data Statistics (use these EXACT numbers):\n" + "\n".join(stats_lines[:20])
        return ""

    def _synthesize_key_findings(
        self,
        all_findings: List[str],
        business_context: BusinessContext
    ) -> List[str]:
        """Synthesize key findings using LLM with business context."""
        # Extract department objectives for context
        all_objectives = []
        for dept in business_context.departments[:4]:  # Top 4 departments
            all_objectives.extend(dept.objectives[:2])  # Top 2 objectives each

        # Extract raw statistics from findings to prevent number hallucination
        raw_stats = self._extract_raw_statistics_from_findings(all_findings)

        context_details = f"""
Business Goals:
- Primary Goal: {business_context.current_goal}

Success Metrics (these are the targets):
{chr(10).join(f'- {metric}' for metric in business_context.success_metrics[:8])}

Department Objectives:
{chr(10).join(f'- {obj}' for obj in all_objectives[:10])}

{raw_stats if raw_stats else ""}
"""

        print("🤖 I'm synthesizing key findings from analysis results...")

        synthesis_prompt = ChatPromptTemplate.from_template(
            """As a senior data analyst, synthesize these findings into 5-7 key takeaways for an executive report.

{context}

All findings from analysis:
{findings}

Provide 5-7 synthesized key findings that:
1. Are specific and quantitative where possible (include numbers, percentages, ranges from findings)
2. Highlight critical issues and opportunities by comparing findings to the Success Metrics targets listed above
3. Connect directly to business objectives and success metrics
4. Identify root causes and systemic patterns, not just symptoms
5. Are prioritized by business impact

CRITICAL REQUIREMENTS:
- If "Raw Data Statistics" are provided above, you MUST use those EXACT numbers - DO NOT change or round them
- Use specific values from the findings (e.g., "39.6% first contact resolution" not "low resolution rate")
- Compare to the Success Metrics targets (e.g., if target is 95% and finding shows 88%, note the 7% gap)
- Reference relevant Department Objectives when explaining business impact
- Identify patterns across time periods or segments
- DO NOT invent, guess, or hallucinate numbers - only use numbers explicitly mentioned in findings or Raw Data Statistics

Format as a numbered list with bold titles followed by brief explanations.
Example: "1. **Title**: Explanation with specific metrics and business impact."
"""
        )

        chain = synthesis_prompt | self.llm
        response = chain.invoke({
            "context": context_details,
            "findings": "\n".join(f"- {f}" for f in all_findings[:40])
        })

        # Parse numbered list
        key_findings = []
        lines = response.content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('**')):
                cleaned = line.lstrip('0123456789.-• ')
                if cleaned and len(cleaned) > 10:  # Filter out very short lines
                    key_findings.append(cleaned)

        return key_findings[:7]

    def _generate_limitations(self, analysis_results: List[AnalysisResult]) -> List[str]:
        """Generate list of analysis limitations."""
        print("🤖 I'm identifying analysis limitations...")

        limitations_prompt = ChatPromptTemplate.from_template(
            """As a data analyst, identify key limitations of this analysis.

Number of analyses: {num_analyses}
Data sources: {data_sources}

List 3-5 important limitations that should be noted in the report.
Consider: data quality, sample size, generalizability, temporal factors, etc.

Format as a list.
"""
        )

        all_data_sources = set()
        for result in analysis_results:
            all_data_sources.update(result.data_sources_used)

        chain = limitations_prompt | self.llm
        response = chain.invoke({
            "num_analyses": len(analysis_results),
            "data_sources": ", ".join(all_data_sources)
        })

        # Parse list
        limitations = []
        lines = response.content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                limitations.append(line.lstrip('-•0123456789. '))

        return limitations[:5]

    def _generate_business_executive_summary(
        self,
        business_context: BusinessContext,
        challenges: List[Challenge],
        analysis_results: List[AnalysisResult]
    ) -> str:
        """Generate executive summary for business insight report."""
        print("🤖 I'm generating business insight report executive summary...")

        summary_prompt = ChatPromptTemplate.from_template(
            """As a business consultant, write an executive summary for business leaders.
Focus on business impact and strategic implications.

Company: {company}
Mission: {mission}
Current Goal: {goal}

Challenges addressed: {challenges}

Key insights from analysis:
{insights}

Write a compelling executive summary (3-4 paragraphs) that:
1. Sets the business context
2. Highlights strategic insights
3. Emphasizes business value
4. Creates urgency for action
"""
        )

        challenges_text = "\n".join(f"- {c.title}" for c in challenges[:5])
        insights_text = []

        for result in analysis_results[:5]:
            insights_text.extend(result.key_findings[:2])

        chain = summary_prompt | self.llm
        response = chain.invoke({
            "company": business_context.company_name,
            "mission": business_context.mission,
            "goal": business_context.current_goal,
            "challenges": challenges_text,
            "insights": "\n".join(f"- {i}" for i in insights_text[:10])
        })

        return response.content

    def _generate_strategic_insights(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult]
    ) -> List[str]:
        """Generate strategic business insights."""
        print("🤖 I'm extracting strategic business insights...")

        insights_prompt = ChatPromptTemplate.from_template(
            """As a strategic business advisor, extract strategic insights from this analysis.

Business Context:
Company: {company}
ICP: {icp}
Mission: {mission}
Current Goal: {goal}

Analysis findings:
{findings}

Causality insights:
{causality}

Generate 5-7 strategic insights that:
1. Connect data findings to business strategy
2. Identify competitive advantages or risks
3. Suggest strategic directions
4. Are actionable at the executive level

Format as a list of strategic insights.
"""
        )

        all_findings = []
        all_causality = []

        for result in analysis_results:
            all_findings.extend(result.key_findings[:3])
            all_causality.extend(result.causality_insights[:2])

        chain = insights_prompt | self.llm
        response = chain.invoke({
            "company": business_context.company_name,
            "icp": business_context.icp,
            "mission": business_context.mission,
            "goal": business_context.current_goal,
            "findings": "\n".join(f"- {f}" for f in all_findings[:15]),
            "causality": "\n".join(f"- {c}" for c in all_causality[:10])
        })

        # Parse list
        insights = []
        lines = response.content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                insights.append(line.lstrip('-•0123456789. '))

        return insights[:7]

    def _identify_opportunities(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult]
    ) -> List[str]:
        """Identify business opportunities from analysis."""
        print("🤖 I'm identifying business opportunities...")

        opportunities_prompt = ChatPromptTemplate.from_template(
            """As a business development expert, identify opportunities based on this analysis.

Business Goal: {goal}
Success Metrics: {metrics}

Analysis insights:
{insights}

Recommendations:
{recommendations}

Identify 4-6 specific business opportunities that could be pursued.
Each opportunity MUST include:
1. A descriptive title
2. A 2-3 sentence explanation of HOW to pursue it and WHAT the expected business impact is

Format as:
- **Opportunity Title:** Clear description explaining what this opportunity is, how to pursue it, and expected business impact.

Example:
- **Expand Enterprise Tier Offering:** Based on the 28% higher revenue per customer in the West region, create a premium enterprise tier targeting large organizations with dedicated support and custom integrations. Expected impact: 15-20% increase in ARR from high-value customers within 6 months.

DO NOT provide just titles without descriptions. Each opportunity must be a complete sentence with actionable details.
"""
        )

        all_insights = []
        all_recommendations = []

        for result in analysis_results:
            all_insights.extend(result.key_findings[:2])
            all_recommendations.extend(result.recommendations[:2])

        chain = opportunities_prompt | self.llm
        response = chain.invoke({
            "goal": business_context.current_goal,
            "metrics": ", ".join(business_context.success_metrics),
            "insights": "\n".join(f"- {i}" for i in all_insights[:15]),
            "recommendations": "\n".join(f"- {r}" for r in all_recommendations[:10])
        })

        # Parse list with multi-line support
        opportunities = []
        lines = response.content.split('\n')
        current_opportunity = ""

        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or (line and line[0].isdigit()):
                # If we have a previous opportunity, save it
                if current_opportunity and len(current_opportunity) > 30:
                    opportunities.append(current_opportunity)
                # Start new opportunity
                current_opportunity = line.lstrip('-•0123456789. ')
            elif current_opportunity and line and not line.startswith('#'):
                # Continue current opportunity (multi-line description)
                current_opportunity += " " + line

        # Add last opportunity
        if current_opportunity and len(current_opportunity) > 30:
            opportunities.append(current_opportunity)

        print(f"🤖 Identified {len(opportunities)} business opportunities")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"   {i}. {opp[:80]}...")

        return opportunities[:6]

    def _generate_action_items(
        self,
        business_context: BusinessContext,
        analysis_results: List[AnalysisResult],
        challenges: List[Challenge]
    ) -> List[str]:
        """Generate prioritized action items."""
        print("🤖 I'm generating prioritized action items...")

        action_prompt = ChatPromptTemplate.from_template(
            """As a business strategist, create a prioritized action plan.

Business Context:
{context}

Challenges addressed:
{challenges}

Key recommendations from analysis:
{recommendations}

Create 5-7 prioritized action items that:
1. Address the most critical challenges
2. Are specific and measurable
3. Have clear business impact
4. Can be implemented in the near term

Format: "Priority level - Action item with specific details"
"""
        )

        challenges_text = "\n".join(f"- {c.title} ({c.priority_level.value})" for c in challenges[:5])

        all_recommendations = []
        for result in analysis_results:
            all_recommendations.extend(result.recommendations)

        chain = action_prompt | self.llm
        response = chain.invoke({
            "context": business_context.to_context_string()[:1000],
            "challenges": challenges_text,
            "recommendations": "\n".join(f"- {r}" for r in all_recommendations[:15])
        })

        # Parse list with improved filtering
        actions = []
        lines = response.content.split('\n')
        for line in lines:
            line = line.strip()

            # Skip lines that are just "Priority:" with no content
            if line in ['Priority:', '**Priority:**', 'Priority', '**Priority**']:
                continue

            # Skip very short lines (likely headers or incomplete content)
            if len(line) < 15:
                continue

            # Match lines that look like action items
            if (line.startswith('-') or line.startswith('•') or
                (line and line[0].isdigit()) or '**' in line):
                cleaned = line.lstrip('-•0123456789. ')

                # Only add if it has substantial content and isn't just a header
                if len(cleaned) > 20 and not cleaned.endswith(':') and not cleaned.startswith('**') and cleaned.endswith('**'):
                    actions.append(cleaned)
                elif len(cleaned) > 30:  # Longer lines are more likely to be complete actions
                    actions.append(cleaned)

        print(f"🤖 Generated {len(actions)} action items")
        if len(actions) < 3:
            print(f"⚠️  WARNING: Only {len(actions)} action items generated (expected 5-7).")
            print(f"   Raw LLM response preview:")
            print(f"   {response.content[:300]}...")
            print(f"   Insufficient data to generate proper action items - returning what was found.")

        return actions[:7]

    def _generate_expected_impact(
        self,
        business_context: BusinessContext,
        action_items: List[str]
    ) -> str:
        """Generate expected impact statement."""
        print("🤖 I'm analyzing expected business impact...")

        impact_prompt = ChatPromptTemplate.from_template(
            """As a business analyst, describe the expected impact of implementing these actions.

Business Goal: {goal}
Success Metrics: {metrics}

Recommended actions:
{actions}

Write 2-3 paragraphs describing:
1. Expected quantitative and qualitative impacts
2. Timeline for seeing results
3. Risk mitigation benefits
4. Competitive advantages gained
"""
        )

        chain = impact_prompt | self.llm
        response = chain.invoke({
            "goal": business_context.current_goal,
            "metrics": ", ".join(business_context.success_metrics),
            "actions": "\n".join(f"- {a}" for a in action_items)
        })

        return response.content

    def _save_report(self, report, report_type: str):
        """Save report to file."""
        # Save as markdown
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{timestamp}.md"
        filepath = self.output_directory / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())

        print(f"Report saved to: {filepath}")
