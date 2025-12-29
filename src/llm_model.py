from anthropic import Anthropic
from dotenv import load_dotenv
import os


def llm_model(prompt, params=None):
    load_dotenv()

    params = params or {}
    api_key = params.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY in your .env file")

    client = Anthropic(api_key=api_key)

    model = params.get("model", "claude-3-haiku-20240307")
    max_tokens = params.get("max_tokens", 400)
    min_new_tokens = params.get("min_new_tokens", 10)
    temperature = params.get("temperature", 0.7)
    top_p = params.get("top_p")
    top_k = params.get("top_k")
    system = params.get("system")
    stop_sequences = params.get("stop_sequences")

    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p
    if top_k is not None:
        request["top_k"] = top_k
    if system is not None:
        request["system"] = system
    if stop_sequences is not None:
        request["stop_sequences"] = stop_sequences

    response = client.messages.create(**request)
    return response.content[0].text
