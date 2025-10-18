"""LangGraph workflow for autonomous code-based discovery."""

from typing import Dict
from langgraph.graph import StateGraph, END
import pandas as pd
from src.models.discovery_models import DiscoveryState, DiscoveryResult
from src.discovery.data_profiler import DataProfiler
from src.discovery.dataset_context_generator import DatasetContextGenerator
from src.discovery.autonomous_explorer import AutonomousExplorer


class DiscoveryWorkflow:
    """
    LangGraph workflow for autonomous code-based data discovery.

    Autonomous Approach:
    1. Generate context (outer agent understands domain)
    2. Profile data (basic statistics)
    3. Autonomous exploration (LLM inspects, cleans, and explores data by writing Python code)
    4. Generate business report

    The LLM agent handles all data cleaning adaptively based on what it discovers.
    No fixed cleaning pipeline - the agent decides what cleaning is needed.
    """

    def __init__(
        self,
        max_iterations: int = 15,
        max_insights: int = 10,
        confidence_threshold: float = 0.6,
        generate_context: bool = True
    ):
        """
        Initialize discovery workflow.

        Args:
            max_iterations: Maximum agent iterations for exploration
            max_insights: Maximum number of insights (not a fixed requirement)
            confidence_threshold: Minimum confidence for insights (deprecated)
            generate_context: Whether to generate dataset context (outer agent layer)
        """
        # Dataset context generator (Outer agent layer)
        self.generate_context = generate_context
        self.context_generator = DatasetContextGenerator()

        # Data profiler (lightweight - just gets basic stats)
        self.profiler = DataProfiler(sample_size=100000)

        # Autonomous explorer - handles data inspection, cleaning, and exploration
        self.explorer = AutonomousExplorer(
            max_iterations=max_iterations,
            max_insights=max_insights
        )

        # Build LangGraph
        self.graph = self._build_graph()

        # Store last context for reporter
        self.last_dataset_context = None
        self.last_exploration_result = None

    def _build_graph(self) -> StateGraph:
        """Build the autonomous exploration workflow (no fixed data cleaning)."""
        workflow = StateGraph(DiscoveryState)

        # Add nodes
        workflow.add_node("generate_context", self._generate_context_node)
        workflow.add_node("profile_data", self._profile_data_node)
        workflow.add_node("autonomous_explore", self._autonomous_explore_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)

        # Define edges (linear flow without fixed cleaning)
        workflow.set_entry_point("generate_context")
        workflow.add_edge("generate_context", "profile_data")
        workflow.add_edge("profile_data", "autonomous_explore")
        workflow.add_edge("autonomous_explore", "synthesize_results")
        workflow.add_edge("synthesize_results", END)

        return workflow.compile()

    def run_discovery(self, df: pd.DataFrame, dataset_name: str) -> DiscoveryResult:
        """
        Run the full autonomous discovery workflow.

        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset

        Returns:
            DiscoveryResult with insights
        """
        print("\n" + "="*80)
        print("[DISCOVERY] AUTONOMOUS DATA DISCOVERY")
        print("="*80)
        print("Using code-based exploration (not fixed algorithms)")
        print("="*80)

        # Initialize state
        initial_state: DiscoveryState = {
            "raw_data": df,
            "dataset_name": dataset_name,
            "data_profile": None,
            "relationship_edges": [],
            "mi_matrix": None,
            "causal_graph_data": None,
            "knowledge_graph": None,
            "insight_questions": [],
            "current_question": None,
            "question_queue": [],
            "current_question_plan": None,
            "answered_questions": [],
            "failed_questions": [],
            "analysis_cache": {},
            "backtrack_count": 0,
            "key_insights": [],
            "discovery_result": None,
            "current_phase": "initialization",
            "status": "pending",
            "error_message": "",
            "max_questions": 10,
            "max_backtrack_attempts": 3,
            "confidence_threshold": 0.6,
            # New fields
            "dataset_context": None,
            "context_summary": None,
            "cleaning_report": None,
            "exploration_result": None
        }

        # Run workflow
        try:
            final_state = self.graph.invoke(initial_state)

            if final_state["status"] == "completed":
                return final_state["discovery_result"]
            else:
                raise Exception(final_state.get("error_message", "Discovery failed"))

        except Exception as e:
            print(f"\n[ERROR] Discovery workflow failed: {e}")
            raise

    # Node implementations

    def _generate_context_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node 1: Generate dataset context (Outer agent layer)."""
        state["current_phase"] = "context_generation"
        state["status"] = "in_progress"

        try:
            if self.generate_context:
                print("\n[CONTEXT] Generating dataset context...")
                dataset_context = self.context_generator.generate_context(
                    state["raw_data"],
                    state["dataset_name"]
                )
                state["dataset_context"] = dataset_context

                # Store for report generation
                self.last_dataset_context = dataset_context

                print(f"[OK] Context generated")
                print(f"  Domain: {dataset_context.get('domain', 'Unknown')}")
                print(f"  Type: {dataset_context.get('dataset_type', 'Unknown')}")

                # Generate context summary for downstream use
                context_summary = self.context_generator.generate_context_summary_for_discovery(
                    dataset_context
                )
                state["context_summary"] = context_summary

            else:
                print("\n[SKIP] Context generation skipped")
                state["dataset_context"] = None
                state["context_summary"] = None
                self.last_dataset_context = None

        except Exception as e:
            print(f"\n[WARN] Context generation failed: {e}")
            print("   Proceeding without context...")
            state["dataset_context"] = {"error": str(e)}
            state["context_summary"] = None
            self.last_dataset_context = None

        return state

    def _profile_data_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node 2: Profile the dataset (lightweight - just basic stats)."""
        state["current_phase"] = "profiling"

        try:
            print("\n[PROFILING] Profiling data...")
            profile = self.profiler.profile_dataset(state["raw_data"])
            state["data_profile"] = profile
            print(f"[OK] Data profiled")
            print(f"  Rows: {len(state['raw_data']):,}")
            print(f"  Columns: {len(state['raw_data'].columns)}")
            print(f"  Numeric: {len(profile.numeric_columns)}")
            print(f"  Categorical: {len(profile.categorical_columns)}")
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Profiling failed: {e}"
            print(f"\n[ERROR] Profiling failed: {e}")

        return state

    def _autonomous_explore_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node 3: Autonomous exploration (LLM inspects, cleans, and explores data)."""
        state["current_phase"] = "exploration"

        try:
            # Get dataset context (handle case where it's None or has error)
            dataset_context = state.get("dataset_context")
            if dataset_context and isinstance(dataset_context, dict) and "error" in dataset_context:
                dataset_context = None

            # Run autonomous exploration
            exploration_result = self.explorer.explore(
                state["raw_data"],
                state["dataset_name"],
                dataset_context
            )

            state["exploration_result"] = exploration_result
            self.last_exploration_result = exploration_result

            print(f"\n[OK] Exploration complete")
            print(f"  Insights: {len(exploration_result.insights)}")
            print(f"  Executions: {exploration_result.total_executions}")
            print(f"[DEBUG] State keys after exploration: {list(state.keys())}")
            print(f"[DEBUG] exploration_result in state: {'exploration_result' in state}")

        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Exploration failed: {e}"
            print(f"\n[ERROR] Exploration failed: {e}")
            import traceback
            traceback.print_exc()

        return state

    def _synthesize_results_node(self, state: DiscoveryState) -> DiscoveryState:
        """Node 5: Synthesize final results."""
        state["current_phase"] = "synthesis"

        print("\n[SYNTHESIS] Synthesizing results...")
        print(f"[DEBUG] State keys in synthesis: {list(state.keys())}")
        print(f"[DEBUG] exploration_result in state: {'exploration_result' in state}")

        try:
            exploration_result = state.get("exploration_result")
            print(f"[DEBUG] exploration_result type: {type(exploration_result)}")
            print(f"[DEBUG] exploration_result is None: {exploration_result is None}")
            if exploration_result:
                print(f"[DEBUG] exploration_result has insights: {hasattr(exploration_result, 'insights')}")
                if hasattr(exploration_result, 'insights'):
                    print(f"[DEBUG] Number of insights: {len(exploration_result.insights)}")

            if not exploration_result:
                state["status"] = "error"
                state["error_message"] = "No exploration results to synthesize"
                return state

            # Convert exploration insights to answered questions format
            from src.models.discovery_models import AnsweredQuestion

            answered_questions = []
            for insight in exploration_result.insights:
                aq = AnsweredQuestion(
                    question=insight.question,
                    answer=insight.finding,
                    confidence=insight.confidence,
                    supporting_data=insight.supporting_data,
                    visualization_path=insight.visualization_paths[0] if insight.visualization_paths else None,
                    supporting_visualizations=insight.visualization_paths  # Pass all visualizations
                )
                answered_questions.append(aq)

            # Extract key insights
            key_insights = [
                f"[{insight.confidence:.0%}] {insight.finding}"
                for insight in exploration_result.insights[:10]
            ]

            # Create discovery result
            result = DiscoveryResult(
                dataset_name=state["dataset_name"],
                data_profile=state["data_profile"],
                answered_questions=answered_questions,
                unanswered_questions=[],
                key_insights=key_insights,
                anomalies_detected=[],
                recommended_analyses=[],
                data_quality_issues=[],
                viz_data_path=exploration_result.viz_data_path if exploration_result else None
            )

            state["discovery_result"] = result
            state["status"] = "completed"

            print(f"[SUCCESS] Discovery completed!")
            print(f"   Insights: {len(answered_questions)}")
            print(f"   Key findings: {len(key_insights)}")

        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Synthesis failed: {e}"
            print(f"[ERROR] Synthesis failed: {e}")

        return state


if __name__ == "__main__":
    print("DiscoveryWorkflow module - ready for use")
    print("Import: from src.graph.discovery_workflow import DiscoveryWorkflow")
