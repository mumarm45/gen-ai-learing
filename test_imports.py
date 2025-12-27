#!/usr/bin/env python3
"""
Test script to verify imports work correctly after fixes
"""
import sys
import os

print("=" * 60)
print("Testing Python GenAI Project Imports")
print("=" * 60)

# Test 1: Check __init__.py files exist
print("\n1. Checking __init__.py files...")
init_files = [
    "__init__.py",
    "custom_langchain/__init__.py",
    "tools/__init__.py",
    "rag/__init__.py",
    "vectors/__init__.py",
    "LCEL/__init__.py",
]

for init_file in init_files:
    if os.path.exists(init_file):
        print(f"   ✓ {init_file}")
    else:
        print(f"   ✗ {init_file} - MISSING!")

# Test 2: Try importing local modules
print("\n2. Testing local module imports...")

try:
    from custom_langchain.llm_chatomodel import llm_model as local_llm_model
    print("   ✓ custom_langchain.llm_chatomodel")
except ImportError as e:
    print(f"   ✗ custom_langchain.llm_chatomodel - {e}")

try:
    from tools.calculator import tools
    print("   ✓ tools.calculator")
except ImportError as e:
    print(f"   ✗ tools.calculator - {e}")

try:
    from llm_model import llm_model
    print("   ✓ llm_model (root)")
except ImportError as e:
    print(f"   ✗ llm_model (root) - {e}")

try:
    from vectors.embeddings_model import embeddings_model
    print("   ✓ vectors.embeddings_model")
except ImportError as e:
    print(f"   ✗ vectors.embeddings_model - {e}")

# Test 3: Try importing installed packages
print("\n3. Testing installed package imports...")

try:
    from langchain_core.prompts import ChatPromptTemplate
    print("   ✓ langchain_core.prompts")
except ImportError as e:
    print(f"   ✗ langchain_core.prompts - {e}")

try:
    from langchain.agents import create_react_agent
    print("   ✓ langchain.agents (installed)")
except ImportError as e:
    print(f"   ✗ langchain.agents - {e}")

try:
    from langchain_anthropic import ChatAnthropic
    print("   ✓ langchain_anthropic")
except ImportError as e:
    print(f"   ✗ langchain_anthropic - {e}")

# Test 4: Check sys.path
print("\n4. Python path (first 5 entries):")
for i, path in enumerate(sys.path[:5], 1):
    print(f"   {i}. {path}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\nIf all tests passed (✓), your imports are fixed!")
print("If any tests failed (✗), check the IMPORT_FIX_GUIDE.md for details.")
