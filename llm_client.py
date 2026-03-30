import os
from typing import Any

import requests


def _bases_to_try(api_base: str) -> list[str]:
    api_base = api_base.rstrip("/")
    bases = [api_base]
    if "localhost" in api_base:
        bases.append(api_base.replace("localhost", "127.0.0.1"))
    elif "127.0.0.1" in api_base:
        bases.append(api_base.replace("127.0.0.1", "localhost"))
    out: list[str] = []
    for b in bases:
        if b not in out:
            out.append(b)
    return out


def detect_model_id(api_base: str, api_key: str = "", timeout_s: int = 30) -> str:
    proxies = {"http": None, "https": None}
    headers = {"Accept": "application/json"}
    headers_auth = dict(headers)
    if api_key:
        headers_auth["Authorization"] = f"Bearer {api_key}"

    last_err: Exception | None = None
    for base in _bases_to_try(api_base):
        try:
            r = requests.get(f"{base}/models", headers=headers_auth, timeout=timeout_s, proxies=proxies)
            if r.status_code >= 400:
                r = requests.get(f"{base}/models", headers=headers, timeout=timeout_s, proxies=proxies)
            r.raise_for_status()
            payload = r.json() if r.text else {}
            ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")]
            if not ids:
                continue
            ids_sorted = sorted(ids, key=lambda x: (("instruct" not in x.lower()), x.lower()))
            return ids_sorted[0]
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not detect model id from /v1/models. Last error: {last_err}")


def chat_completion(prompt: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    api_base = os.environ.get("LLM_API_BASE", "http://127.0.0.1:1234/v1").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "")
    timeout_s = int(os.environ.get("LLM_TIMEOUT_SECONDS", "240"))
    model = os.environ.get("LLM_MODEL") or detect_model_id(api_base, api_key=api_key)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise educational assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err: Exception | None = None
    for base in _bases_to_try(api_base):
        try:
            r = requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("Empty LLM response.")
            return content
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"LLM request failed. Last error: {last_err}")

