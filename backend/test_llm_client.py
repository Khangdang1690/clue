"""Test the updated LLM client configuration."""

from src.utils.llm_client import get_llm


def test_llm_connection():
    """Test that the LLM client can connect and generate a response."""
    print("Testing LLM connection with Gemini 2.5 Flash...")
    print("=" * 60)

    try:
        # Initialize LLM with default settings
        llm = get_llm(temperature=0.7)
        print(f"[OK] LLM initialized successfully")
        print(f"    Model: gemini-2.5-flash")
        print(f"    Temperature: 0.7")
        print()

        # Test simple invocation
        print("Sending test message...")
        response = llm.invoke("Say 'Hello from Gemini 2.5 Flash!' in one sentence.")

        print(f"[OK] Response received:")
        print(f"    {response.content}")
        print()

        # Test with messages
        print("Testing conversation with messages...")
        messages = [
            ("system", "You are a helpful AI assistant."),
            ("human", "What is your model name?")
        ]
        response = llm.invoke(messages)
        print(f"[OK] Response received:")
        print(f"    {response.content}")
        print()

        # Check usage metadata if available
        if hasattr(response, 'usage_metadata'):
            print("[INFO] Token usage:")
            print(f"    Input tokens: {response.usage_metadata.get('input_tokens', 'N/A')}")
            print(f"    Output tokens: {response.usage_metadata.get('output_tokens', 'N/A')}")

        print()
        print("=" * 60)
        print("[SUCCESS] All tests passed!")
        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        print()
        print("Common issues:")
        print("1. Check that GEMINI_API_KEY is set in .env file")
        print("2. Verify API key is valid at https://ai.google.dev/")
        print("3. Ensure langchain-google-genai package is installed:")
        print("   pip install -U langchain-google-genai")
        return False


if __name__ == "__main__":
    test_llm_connection()
