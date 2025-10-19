"""LLM client configuration for Google Gemini.

Updated for 2025 best practices with Gemini 2.5 Flash.
Includes retry logic for rate limiting.
"""

import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import wraps

# Load environment variables
load_dotenv()


def retry_with_exponential_backoff(max_retries=5, initial_delay=30):
    """Decorator to retry with exponential backoff for rate limiting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a rate limit error
                    if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                        if attempt < max_retries - 1:
                            # Extract wait time from error message if available
                            wait_time = initial_delay * (2 ** attempt)

                            # Try to parse retry delay from error message
                            if "retry in" in error_msg.lower():
                                try:
                                    import re
                                    match = re.search(r'retry.*?(\d+\.?\d*)\s*s', error_msg, re.IGNORECASE)
                                    if match:
                                        wait_time = float(match.group(1)) + 1  # Add 1 second buffer
                                except:
                                    pass

                            print(f"[RATE LIMIT] Attempt {attempt + 1}/{max_retries} failed. Waiting {wait_time:.1f}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"[ERROR] Max retries ({max_retries}) exceeded for rate limiting")
                            raise
                    else:
                        # Non-rate-limit error, don't retry
                        raise
            return None
        return wrapper
    return decorator


def get_llm(
    temperature: float = 0.7,
    model: str = "gemini-2.5-flash",
    max_tokens: int | None = None,
    max_retries: int = 5,
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
