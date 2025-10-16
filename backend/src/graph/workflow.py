"""LangGraph workflow for ETL to Insights agent with independent phase execution."""

from typing import Dict, List, Optional
from langgraph.graph import StateGraph, END
from src.graph.state import WorkflowState
from src.models.business_context import BusinessContext
from src.models.challenge import Challenge
from src.models.analysis_result import AnalysisResult
from src.phase1.context_collector import ContextCollector
from src.phase1.challenge_prioritizer import ChallengePrioritizer
from src.phase2.etl_pipeline import ETLPipeline
from src.phase2.statistical_analyzer import StatisticalAnalyzer
from src.phase2.business_analyst import BusinessAnalyst
from src.phase2.visualization_engine import VisualizationEngine
from src.phase2.report_generator import ReportGenerator
from src.phase2.dashboard_generator import DashboardGenerator
from src.utils.chroma_manager import ChromaDBManager
from datetime import datetime


class ETLInsightsWorkflow:
    """
    LangGraph workflow orchestrating the ETL to Insights process.

    Supports independent execution:
    - Phase 1: Setup (run once to identify and prioritize challenges)
    - Phase 2: Analysis (run multiple times, processes one challenge per execution)
    """

    def __init__(self):
        """Initialize the workflow."""
        self.chroma_manager = ChromaDBManager()
        self.context_collector = ContextCollector(self.chroma_manager)
        self.challenge_prioritizer = ChallengePrioritizer(self.chroma_manager)
        self.etl_pipeline = ETLPipeline()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.business_analyst = BusinessAnalyst()
        self.visualization_engine = VisualizationEngine()
        self.report_generator = ReportGenerator()
        self.dashboard_generator = DashboardGenerator()

    def run_phase1(self, business_context: BusinessContext) -> Dict:
        """
        Run Phase 1: Problem Identification.

        This identifies and prioritizes all challenges based on business context.
        Run this ONCE to set up the challenge queue.

        Args:
            business_context: Business context information

        Returns:
            Dictionary with:
            - challenges: List of prioritized challenges
            - status: "completed" or "error"
            - error_message: Error details if failed
        """
        print("\n" + "="*60)
        print("=== PHASE 1: PROBLEM IDENTIFICATION ===")
        print("="*60)

        result = {
            "challenges": [],
            "status": "pending",
            "error_message": ""
        }

        try:
            # Store business context
            print("\n--- Storing Business Context ---")
            self.chroma_manager.store_business_context(
                business_context.to_dict(),
                context_id="main_context"
            )
            print(f"✓ Context stored for: {business_context.company_name}")

            # Identify challenges
            print("\n--- Identifying Challenges ---")
            challenges = self.challenge_prioritizer.identify_challenges(business_context)
            print(f"✓ Identified {len(challenges)} challenges")

            # Prioritize challenges
            print("\n--- Prioritizing Challenges ---")
            prioritized = self.challenge_prioritizer.prioritize_challenges(challenges)

            print(f"✓ Challenges prioritized (highest priority first)")
            print(f"\nTop 5 priorities:")
            for i, challenge in enumerate(prioritized[:5], 1):
                print(f"  {i}. {challenge.title}")
                print(f"     Priority: {challenge.priority_level.value.upper()} ({challenge.priority_score:.1f}/100)")
                print(f"     Department: {challenge.department}")

            result["challenges"] = prioritized
            result["status"] = "completed"

            print(f"\n✓ Phase 1 completed - {len(prioritized)} challenges ready for analysis")
            print(f"  Challenges stored in ChromaDB")
            print(f"  Run Phase 2 to analyze the highest priority challenge")

        except Exception as e:
            result["status"] = "error"
            result["error_message"] = str(e)
            print(f"\n✗ Phase 1 failed: {e}")

        print("="*60)
        return result

    def run_phase2_single(self, max_reports: Optional[int] = None) -> Dict:
        """
        Run Phase 2: Analysis for ONE challenge.

        This processes the highest priority remaining challenge:
        1. Pops the highest priority challenge from ChromaDB
        2. Runs ETL on relevant data
        3. Performs statistical analysis
        4. Generates visualizations
        5. Saves analysis result
        6. Optionally generates report if threshold met

        Can be run multiple times - each run processes one challenge.

        Args:
            max_reports: Generate full report after this many analyses (None = no auto-reports)

        Returns:
            Dictionary with:
            - challenge_processed: Challenge that was analyzed
            - analysis_result: AnalysisResult object
            - challenges_remaining: Number of challenges left
            - status: "completed", "no_challenges", or "error"
            - error_message: Error details if failed
        """
        print("\n" + "="*60)
        print("=== PHASE 2: ANALYSIS (SINGLE CHALLENGE) ===")
        print("="*60)

        result = {
            "challenge_processed": None,
            "analysis_result": None,
            "challenges_remaining": 0,
            "status": "pending",
            "error_message": ""
        }

        try:
            # Get business context
            business_context_dict = self.chroma_manager.get_business_context()
            if not business_context_dict:
                result["status"] = "error"
                result["error_message"] = "No business context found. Run Phase 1 first."
                print(f"\n✗ {result['error_message']}")
                return result

            business_context = BusinessContext(**business_context_dict)

            # Get next challenge
            print("\n--- Selecting Next Challenge ---")
            if not self.challenge_prioritizer.has_challenges():
                # Try to reload from ChromaDB
                all_challenges = self.chroma_manager.get_all_challenges()
                if not all_challenges:
                    result["status"] = "no_challenges"
                    result["error_message"] = "No challenges remaining. Run Phase 1 to identify new challenges."
                    print(f"\n✓ {result['error_message']}")
                    return result

                # Rebuild challenge heap
                challenges = [Challenge(**c) for c in all_challenges]
                self.challenge_prioritizer.prioritize_challenges(challenges)

            challenge = self.challenge_prioritizer.get_next_challenge()
            result["challenge_processed"] = challenge
            result["challenges_remaining"] = len(self.challenge_prioritizer.challenge_heap)

            print(f"✓ Selected: {challenge.title}")
            print(f"  Priority: {challenge.priority_level.value.upper()} ({challenge.priority_score:.1f}/100)")
            print(f"  Department: {challenge.department}")
            print(f"  Remaining challenges: {result['challenges_remaining']}")

            # Run ETL
            print("\n--- ETL Pipeline ---")
            loaded_data, extract_summary, transform_summary, load_summary = \
                self.etl_pipeline.run_etl(department=challenge.department)
            print(f"✓ ETL completed - {len(loaded_data)} datasets loaded")

            # Analyze data (Statistical Analysis - not visualized directly)
            print("\n--- Statistical Analysis ---")
            if not loaded_data:
                print("⚠ No data available - skipping analysis")
                analysis_dict = {
                    "statistical_tests": [],
                    "key_findings": ["No data available for analysis"],
                    "correlations": {},
                    "causality_insights": [],
                }
                business_insights = {
                    'business_questions': [],
                    'insights': [],
                    'key_metrics': {},
                    'visualizations_needed': []
                }
            else:
                # Perform statistical analysis (finds patterns, correlations, etc.)
                analysis_dict = self.statistical_analyzer.analyze_challenge(
                    challenge, loaded_data, business_context
                )
                print(f"✓ Statistical analysis completed")
                print(f"  Statistical tests: {len(analysis_dict['statistical_tests'])}")
                print(f"  Key findings: {len(analysis_dict['key_findings'])}")
                print(f"  Correlations: {len(analysis_dict['correlations'])}")

                # Transform statistical findings into business insights
                print("\n--- Business Analysis ---")
                business_insights = self.business_analyst.analyze_for_business(
                    challenge, analysis_dict, loaded_data, business_context
                )
                print(f"✓ Business analysis completed")
                print(f"  Business questions generated: {len(business_insights['business_questions'])}")
                print(f"  Insights created: {len(business_insights['insights'])}")
                print(f"  Visualizations planned: {len(business_insights['visualizations_needed'])}")

            # Generate business-oriented visualizations
            print("\n--- Generating Business Visualizations ---")
            if loaded_data and business_insights.get('visualizations_needed'):
                visualizations = self.visualization_engine.generate_business_visualizations(
                    challenge, business_insights
                )
                print(f"✓ Generated {len(visualizations)} business-oriented visualizations")
            else:
                visualizations = []
                print("⚠ No visualizations generated (no data or business insights)")

            # Create analysis result
            analysis_result = AnalysisResult(
                challenge_id=challenge.id,
                challenge_title=challenge.title,
                data_sources_used=list(loaded_data.keys()) if loaded_data else [],
                extraction_summary=extract_summary,
                transformation_summary=transform_summary,
                load_summary=load_summary,
                statistical_tests=analysis_dict.get("statistical_tests", []),
                key_findings=analysis_dict.get("key_findings", []),
                correlations=analysis_dict.get("correlations", {}),
                visualizations=visualizations,
                causality_insights=analysis_dict.get("causality_insights", []),
                recommendations=self._generate_recommendations(analysis_dict, challenge)
            )

            # Save to ChromaDB as processed
            self._save_analysis_to_chromadb(challenge.id, analysis_result)

            result["analysis_result"] = analysis_result
            result["status"] = "completed"

            print(f"\n✓ Phase 2 completed for: {challenge.title}")
            print(f"  Analysis saved")
            print(f"  Challenges remaining: {result['challenges_remaining']}")

            # Generate reports and dashboard immediately after analysis
            print("\n--- Generating Reports and Dashboard ---")
            try:
                # Get all analysis results so far
                all_analyses = self._load_all_analyses_from_chromadb()

                if all_analyses:
                    # Generate analytical report
                    print("Generating analytical report...")
                    analytical_report = self.report_generator.generate_analytical_report(
                        business_context, all_analyses
                    )

                    # Generate business report
                    print("Generating business insight report...")
                    all_challenges_dict = self.chroma_manager.get_all_challenges()
                    challenges = [Challenge(**c) for c in all_challenges_dict]
                    business_report = self.report_generator.generate_business_insight_report(
                        business_context, all_analyses, challenges
                    )

                    # Generate interactive dashboard
                    print("Generating interactive dashboard...")
                    dashboard_path = self.dashboard_generator.generate_dashboard(
                        business_context, all_analyses
                    )

                    print(f"✓ Reports and dashboard generated")
                    print(f"  Reports: data/outputs/reports/")
                    print(f"  Dashboard: {dashboard_path}")

                    result["analytical_report_path"] = "data/outputs/reports/"
                    result["business_report_path"] = "data/outputs/reports/"
                    result["dashboard_path"] = dashboard_path
                else:
                    print("⚠ No analyses to report on")
            except Exception as report_error:
                print(f"⚠ Report generation failed: {report_error}")
                # Don't fail the whole phase if reports fail
                result["report_error"] = str(report_error)

        except Exception as e:
            result["status"] = "error"
            result["error_message"] = str(e)
            print(f"\n✗ Phase 2 failed: {e}")

        print("="*60)
        return result

    def generate_reports(self, analysis_results: Optional[List[AnalysisResult]] = None) -> Dict:
        """
        Generate comprehensive reports from all analyses.

        Can be called after analyzing one or more challenges.

        Args:
            analysis_results: List of AnalysisResult objects (or None to load from ChromaDB)

        Returns:
            Dictionary with:
            - analytical_report_path: Path to analytical report
            - business_report_path: Path to business insight report
            - dashboard_path: Path to interactive dashboard
            - status: "completed" or "error"
        """
        print("\n" + "="*60)
        print("=== GENERATING COMPREHENSIVE REPORTS ===")
        print("="*60)

        result = {
            "analytical_report_path": None,
            "business_report_path": None,
            "dashboard_path": None,
            "status": "pending",
            "error_message": ""
        }

        try:
            # Get business context
            business_context_dict = self.chroma_manager.get_business_context()
            if not business_context_dict:
                raise ValueError("No business context found")

            business_context = BusinessContext(**business_context_dict)

            # Get analysis results if not provided
            if analysis_results is None:
                print("Loading analysis results from ChromaDB...")
                analysis_results = self._load_all_analyses_from_chromadb()

            if not analysis_results:
                raise ValueError("No analysis results available for reporting")

            # Get all challenges
            all_challenges_dict = self.chroma_manager.get_all_challenges()
            challenges = [Challenge(**c) for c in all_challenges_dict]

            print(f"Generating reports for {len(analysis_results)} analyses...")

            # Generate analytical report
            print("\n--- Analytical Report ---")
            analytical_report = self.report_generator.generate_analytical_report(
                business_context, analysis_results
            )
            result["analytical_report_path"] = "data/outputs/reports/"
            print("✓ Analytical report generated")

            # Generate business insight report
            print("\n--- Business Insight Report ---")
            business_report = self.report_generator.generate_business_insight_report(
                business_context, analysis_results, challenges
            )
            result["business_report_path"] = "data/outputs/reports/"
            print("✓ Business insight report generated")

            # Generate interactive dashboard
            print("\n--- Interactive Dashboard ---")
            dashboard_path = self.dashboard_generator.generate_dashboard(
                business_context, analysis_results
            )
            result["dashboard_path"] = dashboard_path
            print("✓ Interactive dashboard generated")

            result["status"] = "completed"
            print(f"\n✓ Reports and dashboard generated successfully")
            print(f"  Reports location: data/outputs/reports/")
            print(f"  Dashboard location: {dashboard_path}")

        except Exception as e:
            result["status"] = "error"
            result["error_message"] = str(e)
            print(f"\n✗ Report generation failed: {e}")

        print("="*60)
        return result

    def _generate_recommendations(self, analysis: Dict, challenge: Challenge) -> List[str]:
        """Generate recommendations based on analysis (no hardcoding, real logic)."""
        recommendations = []

        # Based on findings
        findings = analysis.get("key_findings", [])
        if findings:
            for finding in findings[:2]:
                recommendations.append(f"Address: {finding}")

        # Based on correlations
        correlations = analysis.get("correlations", {})
        if correlations:
            for corr_pair, corr_value in list(correlations.items())[:2]:
                if abs(corr_value) > 0.7:
                    recommendations.append(
                        f"Strong correlation detected in {corr_pair} ({corr_value:.2f}) - investigate causal relationship"
                    )

        # Based on causality
        causality = analysis.get("causality_insights", [])
        if causality:
            for insight in causality[:2]:
                recommendations.append(f"Explore: {insight}")

        # Based on statistical tests
        tests = analysis.get("statistical_tests", [])
        significant_tests = [t for t in tests if t.is_significant]
        if significant_tests:
            recommendations.append(
                f"Significant patterns found in {len(significant_tests)} statistical tests - review for action items"
            )

        return recommendations[:5]

    def _save_analysis_to_chromadb(self, challenge_id: str, analysis_result: AnalysisResult):
        """Save analysis result to ChromaDB."""
        import json

        collection_name = "analysis_results"
        try:
            collection = self.chroma_manager.client.get_collection(collection_name)
        except:
            collection = self.chroma_manager.client.create_collection(
                collection_name,
                metadata={"description": "Stores analysis results"}
            )

        collection.upsert(
            ids=[challenge_id],
            documents=[json.dumps(analysis_result.model_dump(), default=str)],
            metadatas=[{
                "challenge_id": challenge_id,
                "challenge_title": analysis_result.challenge_title,
                "timestamp": str(analysis_result.timestamp)
            }]
        )

    def _load_all_analyses_from_chromadb(self) -> List[AnalysisResult]:
        """Load all analysis results from ChromaDB."""
        import json

        try:
            collection = self.chroma_manager.client.get_collection("analysis_results")
            results = collection.get()

            analyses = []
            if results and results['documents']:
                for doc in results['documents']:
                    data = json.loads(doc)
                    analyses.append(AnalysisResult(**data))

            return analyses
        except Exception as e:
            print(f"Warning: Could not load analyses from ChromaDB: {e}")
            return []

    def get_challenge_status(self) -> Dict:
        """
        Get status of challenge queue.

        Returns:
            Dictionary with challenge queue status
        """
        try:
            all_challenges = self.chroma_manager.get_all_challenges()

            # Load into prioritizer to check heap
            if all_challenges and not self.challenge_prioritizer.has_challenges():
                challenges = [Challenge(**c) for c in all_challenges]
                self.challenge_prioritizer.prioritize_challenges(challenges)

            remaining = len(self.challenge_prioritizer.challenge_heap)
            total = len(all_challenges) if all_challenges else 0
            processed = total - remaining

            return {
                "total_challenges": total,
                "processed": processed,
                "remaining": remaining,
                "next_challenge": self.challenge_prioritizer.challenge_heap[0] if remaining > 0 else None
            }
        except Exception as e:
            return {
                "total_challenges": 0,
                "processed": 0,
                "remaining": 0,
                "next_challenge": None,
                "error": str(e)
            }

    # Legacy method for backward compatibility
    def run(self, initial_state: WorkflowState) -> WorkflowState:
        """
        Run the complete workflow (legacy method - processes ALL challenges).

        For new code, use run_phase1() and run_phase2_single() instead.
        """
        print("\n" + "="*60)
        print("=== ETL TO INSIGHTS AI AGENT (FULL RUN) ===")
        print("="*60)
        print("Note: This processes ALL challenges. Use phase1/phase2 methods for incremental processing.")

        business_context = initial_state["business_context"]

        # Run Phase 1
        phase1_result = self.run_phase1(business_context)

        if phase1_result["status"] != "completed":
            initial_state["status"] = "error"
            initial_state["error_message"] = phase1_result["error_message"]
            return initial_state

        challenges = phase1_result["challenges"]
        initial_state["challenges"] = challenges

        # Run Phase 2 for all challenges
        analysis_results = []
        for i, challenge in enumerate(challenges, 1):
            print(f"\n--- Processing Challenge {i}/{len(challenges)} ---")
            phase2_result = self.run_phase2_single()

            if phase2_result["status"] == "completed":
                analysis_results.append(phase2_result["analysis_result"])
            elif phase2_result["status"] == "error":
                print(f"⚠ Skipping challenge due to error: {phase2_result['error_message']}")

        initial_state["analysis_results"] = analysis_results

        # Generate reports
        report_result = self.generate_reports(analysis_results)

        if report_result["status"] == "completed":
            initial_state["status"] = "completed"
        else:
            initial_state["status"] = "error"
            initial_state["error_message"] = report_result["error_message"]

        print("\n" + "="*60)
        print("=== WORKFLOW COMPLETED ===")
        print("="*60)

        return initial_state
