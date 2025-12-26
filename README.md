# gen-ai-learing

Learning GenAI with Python (Anthropic + optional LangChain).

## Requirements

- **Python**: 3.11 or 3.12 recommended
- **API key**: Anthropic key provided via `ANTHROPIC_API_KEY`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with:

```bash
ANTHROPIC_API_KEY=your_key_here
```

## Run

Run the main demo:

```bash
python main.py
```

By default, `main.py` builds a small LangChain pipeline that formats a prompt and calls Anthropic.

## Notes

- If `ANTHROPIC_API_KEY` is missing, the code raises: `ValueError("Please set the ANTHROPIC_API_KEY in your .env file")`.
- LangChain is not compatible with Python 3.14+ in this project (see `llm_model_langchain()` guard in `main.py`). If you are on 3.14+, use Python 3.11/3.12 or use the direct Anthropic SDK function `llm_model()`.

- **max_tokens**: 400 # Try 256 or 512 for more detailed answers
- **min_new_tokens**: 10 # Increase to 25-50 if you want more substantial answers
- **system**: "You are a helpful assistant."
- **temperature**: 0.7 # Lower (0.1-0.3): More focused, consistent, factual responses
                      # Higher (0.7-1.0): More creative, diverse, unpredictable outputs
- **top_p**: 0.9 # Nucleus sampling - considers only highest probability tokens
                       # Lower values (0.1-0.3): More conservative, focused text
                       # Higher values (0.7-0.9): More diverse vocabulary and ideas
- **top_k**: 50 # Limits token selection to top k most likely tokens
                       # 1 = greedy decoding (always picks most likely token)
                       # Try 40-50 for more varied outputs

## Project files

- `main.py`: examples and helper functions for calling Anthropic directly
- `lang_chain.py`: examples and helper functions for calling Anthropic via LangChain
- `llm_model.py`: functions for calling Anthropic
- `few_shot.py`: examples and helper functions for few shot learning
- `cot_prompt.py`: examples and helper functions for chain of thought prompts
- `self_consistency.py`: examples and helper functions for self consistency
- `zero_shot.py`: examples and helper functions for zero shot learning
- `one_shot.py`: examples and helper functions for one shot learning
- `requirements.txt`: Python dependencies
