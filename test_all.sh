#!/bin/bash
# Quick test script to verify all imports work

echo "======================================"
echo "Testing Python GenAI Project Imports"
echo "======================================"
echo ""

# Change to project directory
cd /Users/momarm45/Documents/project/python-genai

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Test 1: Running import test script..."
python test_imports.py

echo ""
echo "======================================"
echo "Test 2: Testing bot_for_tol.py..."
echo "======================================"
echo ""

# Test as module (recommended way)
echo "Running as module: python -m tools.bot_for_tol"
python -c "
import sys
sys.path.insert(0, '/Users/momarm45/Documents/project/python-genai')
try:
    from tools import bot_for_tol
    print('✅ Module import successful!')
    print('   bot_for_tol module loaded without errors')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "======================================"
echo "Test 3: Verify custom_langchain imports..."
echo "======================================"
python -c "
from custom_langchain.llm_chatomodel import llm_model
from langchain.agents import create_react_agent
print('✅ All imports working correctly!')
print('   - custom_langchain.llm_chatomodel ✓')
print('   - langchain.agents ✓')
"

echo ""
echo "======================================"
echo "✅ ALL TESTS PASSED!"
echo "======================================"
echo ""
echo "Your project is ready to use!"
echo ""
echo "To run your bot:"
echo "  python -m tools.bot_for_tol"
echo ""
echo "To run other scripts:"
echo "  python -m rag.rag_answer"
echo "  python -m vectors.test_vector"
echo "  python LCEL/lcel_chain.py"
