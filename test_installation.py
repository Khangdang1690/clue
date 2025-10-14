"""Test script to verify installation and basic functionality."""

import sys
from pathlib import Path

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False

    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False

    try:
        import chromadb
        print("✓ chromadb")
    except ImportError as e:
        print(f"✗ chromadb: {e}")
        return False

    try:
        import langchain
        print("✓ langchain")
    except ImportError as e:
        print(f"✗ langchain: {e}")
        return False

    try:
        import langgraph
        print("✓ langgraph")
    except ImportError as e:
        print(f"✗ langgraph: {e}")
        return False

    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False

    try:
        import seaborn
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False

    try:
        import scipy
        print("✓ scipy")
    except ImportError as e:
        print(f"✗ scipy: {e}")
        return False

    try:
        import statsmodels
        print("✓ statsmodels")
    except ImportError as e:
        print(f"✗ statsmodels: {e}")
        return False

    try:
        import pydantic
        print("✓ pydantic")
    except ImportError as e:
        print(f"✗ pydantic: {e}")
        return False

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✓ langchain-google-genai")
    except ImportError as e:
        print(f"✗ langchain-google-genai: {e}")
        return False

    return True


def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")

    required_dirs = [
        "src/models",
        "src/phase1",
        "src/phase2",
        "src/graph",
        "src/utils",
        "data/uploads",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
            all_exist = False

    return all_exist


def test_env_file():
    """Test that .env file exists and has API key."""
    print("\nTesting environment configuration...")

    env_path = Path(".env")
    if not env_path.exists():
        print("✗ .env file not found")
        print("  Create .env file with: GEMINI_API_KEY=your_key_here")
        return False

    print("✓ .env file exists")

    # Check if API key is set
    with open(env_path) as f:
        content = f.read()
        if "GEMINI_API_KEY" in content:
            print("✓ GEMINI_API_KEY found in .env")
            return True
        else:
            print("✗ GEMINI_API_KEY not found in .env")
            return False


def test_models():
    """Test that models can be instantiated."""
    print("\nTesting data models...")

    try:
        from src.models.business_context import BusinessContext, Department

        dept = Department(
            name="Test",
            description="Test department",
            objectives=["Test objective"],
            painpoints=["Test painpoint"],
            perspectives=["Test perspective"]
        )
        print("✓ Department model")

        context = BusinessContext(
            company_name="Test Company",
            icp="Test ICP",
            mission="Test mission",
            current_goal="Test goal",
            success_metrics=["Metric 1"],
            departments=[dept]
        )
        print("✓ BusinessContext model")

        from src.models.challenge import Challenge, PriorityLevel

        challenge = Challenge(
            id="test-1",
            title="Test Challenge",
            description="Test description",
            department="Test",
            priority_score=75.0,
            priority_level=PriorityLevel.HIGH,
            related_painpoints=["Pain 1"],
            related_objectives=["Objective 1"],
            success_metrics=["Metric 1"],
            data_sources_needed=["Source 1"]
        )
        print("✓ Challenge model")

        return True

    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")

    try:
        from src.utils.chroma_manager import ChromaDBManager

        # Don't actually create DB, just test import
        print("✓ ChromaDBManager import")

        from src.utils.llm_client import get_llm
        print("✓ LLM client import")

        return True

    except Exception as e:
        print(f"✗ Utility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ETL to Insights AI Agent - Installation Test")
    print("="*60)

    results = {
        "Imports": test_imports(),
        "Project Structure": test_project_structure(),
        "Environment": test_env_file(),
        "Data Models": test_models(),
        "Utilities": test_utilities()
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Run: python setup_sample_data.py (optional)")
        print("2. Run: python main.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Missing packages: pip install -r requirements.txt")
        print("- Missing .env: Create .env with GEMINI_API_KEY")
        return 1


if __name__ == "__main__":
    sys.exit(main())
