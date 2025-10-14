"""LLM client configuration for Google Gemini.

Updated for 2025 best practices with Gemini 2.5 Flash.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def get_llm(
    temperature: float = 0.7,
    model: str = "gemini-2.5-flash",
    max_tokens: int | None = None,
    max_retries: int = 2,
    timeout: int | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Get configured LLM instance using Google Gemini.

    Args:
        temperature: Temperature for generation (0.0 to 1.0).
                    0 = deterministic, 1.0 = most creative
        model: Model name to use. Options:
               - "gemini-2.5-flash" (recommended, stable)
               - "gemini-2.5-flash-preview-09-2025" (latest preview)
               - "gemini-2.5-pro" (more capable, slower)
        max_tokens: Optional maximum tokens to generate
        max_retries: Number of retries on API failures (default: 2)
        timeout: Optional timeout in seconds for API calls

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Raises:
        ValueError: If GEMINI_API_KEY is not found in environment

    Example:
        >>> llm = get_llm(temperature=0.7)
        >>> response = llm.invoke("What is LangGraph?")
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )

    # Use recommended parameter names for 2025
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
        google_api_key=api_key,
        # Additional recommended settings
        convert_system_message_to_human=True,  # Better compatibility
    )
