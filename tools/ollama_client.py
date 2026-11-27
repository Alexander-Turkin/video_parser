#!/usr/bin/env python3
"""Local Ollama client shared by punctuation, speaker roles, and summaries."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = 300.0
DEFAULT_ATTEMPTS = 2
CONTEXT_TOKENS = 16384
OUTPUT_TOKENS = 3072
MAX_PROMPT_BYTES = 256 * 1024


class OllamaError(Exception):
    """Local server did not produce a usable answer."""


class OllamaUnavailable(OllamaError):
    """The local service or selected model is unavailable."""


class OllamaFailed(OllamaError):
    """The local model did not produce a usable response."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _request(path: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            result = json.load(response)
            if not isinstance(result, dict):
                raise OllamaFailed("Ollama вернула JSON не в форме объекта")
            return result
    except urllib.error.HTTPError as exc:
        raise OllamaFailed("Ollama вернула HTTP %d" % exc.code) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise OllamaUnavailable("Ollama недоступна: %s" % exc) from None
    except (ValueError, UnicodeError) as exc:
        raise OllamaFailed("Ollama вернула негодный JSON: %s" % exc) from None


def probe(model: str = DEFAULT_MODEL) -> None:
    """Check the loopback server and the exact downloaded model name."""
    test_bin = os.environ.get("OLLAMA_TEST_BIN")
    if test_bin is not None:
        if not os.path.isfile(test_bin) or not os.access(test_bin, os.X_OK):
            raise OllamaUnavailable("файл не найден: тестовый исполнитель %s" % test_bin)
        return
    data = _request("/api/tags", None, 5)
    models = data.get("models")
    if not isinstance(models, list):
        raise OllamaFailed("Ollama вернула список моделей в неверном формате")
    names = {item.get("name") for item in models if isinstance(item, dict)}
    if model not in names:
        raise OllamaUnavailable("модель %s не установлена; выполните ollama pull %s" % (model, model))


def available(model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    try:
        probe(model)
    except OllamaError as exc:
        return False, str(exc)
    return True, "модель %s доступна" % model


def _log(log_dir: str | None, tag: str, prompt: str) -> Path | None:
    if log_dir is None:
        return None
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    number = 1
    while (folder / f"{tag}-{number:03d}.prompt.txt").exists():
        number += 1
    prefix = folder / f"{tag}-{number:03d}"
    Path(str(prefix) + ".prompt.txt").write_text(prompt, encoding="utf-8")
    return Path(str(prefix) + ".response.txt")


def _ask_test_binary(binary: str, prompt: str, timeout: float, attempts: int) -> str:
    """Self-test bridge for existing shell fixtures; never used by normal runs."""
    size = len(prompt.encode("utf-8"))
    if size > MAX_PROMPT_BYTES:
        raise OllamaFailed("промпт слишком длинный: %d байт" % size)
    probe()
    problem = ""
    for _ in range(attempts):
        try:
            result = subprocess.run([binary, "-p", prompt], capture_output=True,
                                    text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            problem = "таймаут %g с" % timeout
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            if "invalid api key" in detail.lower():
                raise OllamaUnavailable("тестовая модель не авторизована: %s" % detail)
            problem = "модель вернула код %d: %s" % (result.returncode, detail)
            continue
        if result.stdout.strip():
            return result.stdout.strip()
        raise OllamaFailed("модель вернула пустой ответ")
    raise OllamaFailed("%s (всего %d попыток)" % (problem, attempts))


def ask(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
    log_dir: str | None = None,
    tag: str = "саммари",
) -> str:
    """Ask a locally installed model; reject capped or empty responses."""
    response_path = _log(log_dir, tag, prompt)
    test_bin = os.environ.get("OLLAMA_TEST_BIN")
    if test_bin is not None:
        try:
            answer = _ask_test_binary(test_bin, prompt, timeout, attempts)
        except OllamaError as exc:
            if response_path is not None:
                response_path.write_text("НЕ ПОЛУЧЕНО\n" + str(exc), encoding="utf-8")
            raise
        if response_path is not None:
            response_path.write_text(answer, encoding="utf-8")
        return answer
    problem = ""
    error_type: type[OllamaError] = OllamaFailed
    for attempt in range(attempts):
        try:
            data = _request("/api/chat", {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "num_ctx": CONTEXT_TOKENS,
                    "num_predict": OUTPUT_TOKENS,
                    "temperature": 0,
                },
            }, timeout)
            message = data.get("message")
            if not isinstance(message, dict):
                raise OllamaFailed("Ollama вернула ответ без поля message")
            answer = message.get("content", "")
            prompt_count, output_count = data.get("prompt_eval_count"), data.get("eval_count")
            if not isinstance(prompt_count, int) or not isinstance(output_count, int):
                raise OllamaFailed("Ollama не сообщила число обработанных токенов")
            used = prompt_count + output_count
            if used >= CONTEXT_TOKENS * 0.9:
                raise OllamaFailed("контекст почти заполнен (%s токенов)" % used)
            if data.get("done_reason") == "length":
                raise OllamaFailed("ответ обрезан лимитом токенов")
            if not isinstance(answer, str) or not answer.strip():
                raise OllamaFailed("Ollama вернула пустой ответ")
            if response_path is not None:
                response_path.write_text(answer, encoding="utf-8")
            return answer.strip()
        except OllamaError as exc:
            problem = str(exc)
            error_type = type(exc)
            if attempt + 1 < attempts and ("недоступна" in problem or "контекст" in problem):
                time.sleep(1)
                continue
            break
    if response_path is not None:
        response_path.write_text("НЕ ПОЛУЧЕНО\n" + problem, encoding="utf-8")
    raise error_type(problem)


def main(argv: list[str]) -> int:
    if argv in (["-h"], ["--help"]):
        print("Использование: python tools/ollama_client.py --probe [--model <имя>]")
        return 0
    if not argv or argv[0] != "--probe":
        print("ollama_client.py: ожидается --probe", file=sys.stderr)
        return 2
    model = DEFAULT_MODEL
    if len(argv) == 3 and argv[1] == "--model" and argv[2]:
        model = argv[2]
    elif len(argv) != 1:
        print("ollama_client.py: неверные аргументы", file=sys.stderr)
        return 2
    try:
        probe(model)
    except OllamaError as exc:
        print("ollama_client.py: %s" % exc, file=sys.stderr)
        return 1
    print("Ollama: модель %s доступна" % model)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
