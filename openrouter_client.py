import os
import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

OPENROUTER_VISION_MODEL = "meta-llama/llama-3.2-11b-vision-instruct"

_HEADERS_COMMON = {
    "HTTP-Referer": "https://medical-image-analysis.app",
    "X-Title": "Medical Image Analysis",
}


def _build_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **_HEADERS_COMMON,
    }


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """
    Send a minimal test request to OpenRouter to validate the key.

    Returns:
        (True, "")           -- key is valid
        (False, error_msg)   -- key is invalid or network error
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty."
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=_build_headers(api_key),
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=15,
        )
        if resp.status_code in (200, 400):
            return True, ""
        if resp.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)."
        if resp.status_code == 429:
            return True, ""
        return False, f"Unexpected status {resp.status_code}: {resp.text[:200]}"
    except requests.RequestException as exc:
        return False, f"Network error during key validation: {exc}"


def call_openrouter(
    api_key: str,
    messages: list,
    *,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 800,
    temperature: float = 0.2,
    stream: bool = True,
) -> str:
    """
    Send a chat-completion request to OpenRouter.

    Streaming is used by default so that gpt-oss-120b returns reasoning
    tokens correctly (mirrors the JS SDK pattern).
    """
    if not api_key:
        raise ValueError("OpenRouter API key is required.")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=_build_headers(api_key),
        json=payload,
        timeout=120,
        stream=stream,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: {response.text[:500]}"
        )

    if stream:
        return _collect_stream(response)

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content


def _collect_stream(response: requests.Response) -> str:
    """Consume a streaming SSE response and return assembled text."""
    import json as _json

    collected = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = _json.loads(payload)
        except _json.JSONDecodeError:
            continue
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            collected.append(content)
    return "".join(collected)


def call_vision(
    api_key: str,
    image_b64: str,
    prompt: str,
    *,
    media_type: str = "image/jpeg",
    max_tokens: int = 1000,
    temperature: float = 0.2,
) -> str:
    """Send an image + text prompt to the vision model (non-streaming)."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return call_openrouter(
        api_key,
        messages,
        model=OPENROUTER_VISION_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
