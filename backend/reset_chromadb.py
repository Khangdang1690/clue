#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reset ChromaDB - Clears all data from the vector database.

This script removes all ChromaDB data, allowing you to start fresh.
Use this when you want to clear old business contexts, challenges, and analysis results.

Usage:
    python reset_chromadb.py
"""

import shutil
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def reset_chromadb():
    """Remove all ChromaDB data."""
    chroma_paths = [
        Path(__file__).parent / "chroma_db",
        Path(__file__).parent / "data" / "chroma_db",
    ]

    for chroma_path in chroma_paths:
        if chroma_path.exists():
            print(f"[DELETE] Removing: {chroma_path}")
            shutil.rmtree(chroma_path)
            print(f"[SUCCESS] Deleted: {chroma_path}")
        else:
            print(f"[INFO] Not found (skipping): {chroma_path}")

    print("\n[SUCCESS] ChromaDB reset complete!")
    print("Next server start will create a fresh database.")

if __name__ == "__main__":
    print("=" * 60)
    print("ChromaDB Reset Tool")
    print("=" * 60)
    print("\n[WARNING] This will delete all stored data:")
    print("  - Business contexts")
    print("  - Challenges")
    print("  - Analysis results")
    print("  - All ChromaDB collections")

    response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()

    if response in ["yes", "y"]:
        reset_chromadb()
    else:
        print("\n[CANCELLED] Reset cancelled.")
