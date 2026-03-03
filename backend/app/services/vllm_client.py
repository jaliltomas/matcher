import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)


class VllmChatClient:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str,
        timeout_seconds: int,
        max_retries: int,
        disable_thinking: bool,
        max_parallel: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking
        self.max_parallel = max_parallel

    def _headers(self) -> dict[str, str]:
        token = self.api_key or "EMPTY"
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
        system_prompt: str = "Devuelve solo JSON valido sin texto adicional.",
    ) -> str:
        payload: dict = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("vLLM response without choices")
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    text_chunks = [part.get("text", "") for part in content if isinstance(part, dict)]
                    content = "".join(text_chunks)
                return str(content)
            except Exception as exc:
                last_error = exc
                sleep_seconds = min(2.5, 0.5 * attempt)
                time.sleep(sleep_seconds)
        raise RuntimeError(f"vLLM completion failed: {last_error}")

    def complete_many(
        self,
        prompts: list[str],
        max_tokens: int,
        workers: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        system_prompt: str = "Devuelve solo JSON valido sin texto adicional.",
    ) -> list[str]:
        if not prompts:
            return []
        effective_workers = max(1, min(workers or self.max_parallel, self.max_parallel, len(prompts)))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            return list(
                executor.map(
                    lambda p: self.complete(
                        p,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        system_prompt=system_prompt,
                    ),
                    prompts,
                )
            )
