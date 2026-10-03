#!/usr/bin/env python3
"""
claude_cli.py — единственное место, где проект запускает Claude CLI.

Зачем он нужен. Пунктуацию расшифровки и саммари ставит claude, и вызывают его
два инструмента: восстановление пунктуации (много вызовов по окнам текста) и
саммари (один большой вызов). Если каждый из них будет сам собирать команду,
ловить таймаут и разбирать коды возврата, то правила разъедутся, а однажды
кто-нибудь передаст интерактивный флаг и прогон встанет, ожидая ввода. Поэтому
вызов собран здесь один раз: находим бинарник, запускаем ровно
`claude -p <промпт>`, ждём с таймаутом, при сбое повторяем, а любую неудачу
превращаем в исключение с русским сообщением.

Решение владельца: вызывать именно `claude -p <промпт>` с моделью по умолчанию.
Модель здесь не задаётся намеренно — ни флагом, ни переменной окружения.

Почему модуль ничего не печатает в stdout и не завершает процесс. Требование к
пайплайну: расшифровка обязана сохраниться, даже если claude недоступен или упал
на саммари, — вместо саммари ставится пометка «не получено», а код возврата
прогона ненулевой. Решение об этом принимает вызывающий инструмент, поэтому
модуль только поднимает исключение. В stderr он всё же пишет — предупреждением о
повторной попытке (warn), как и соседние tools/*.py.

Публичный API (фиксированный, на него опираются tools/punctuate.py и
tools/summarize.py):

    available() -> tuple[bool, str]
        Есть ли чем вызывать claude. (True, путь к бинарнику) либо
        (False, причина по-русски). Живого вызова не делает, поэтому
        авторизацию не проверяет: это видно только по ответу самого claude.

    ask(prompt: str, *, timeout: float = 300, attempts: int = 2,
        log_dir: str | None = None, tag: str = "call") -> str
        Один неинтерактивный вызов: возвращает текст ответа без крайних
        пробелов. timeout — секунды на одну попытку, attempts — сколько попыток
        всего, log_dir — куда складывать промпт и ответ, tag — префикс имён
        этих файлов. Ошибки в самих аргументах (timeout <= 0, attempts < 1,
        негодный tag) — это ошибка программиста, она поднимает ValueError, а не
        ClaudeFailed.

    ClaudeUnavailable
        Вызывать нечем: бинарника нет, он не исполняемый, не удалось запустить,
        либо claude сообщил, что не авторизован. Повторять бессмысленно.

    ClaudeFailed
        Вызвали, но ответа нет: таймаут, ненулевой код, пустой ответ, слишком
        длинный промпт.

    Обе — потомки ClaudeError, и обе несут человекочитаемый str(e) на русском.

Как собирается команда. Ровно `claude -p <промпт>`: промпт идёт одним
аргументом, модель не задаётся, интерактивные и потоковые флаги не передаются.
Путь к бинарнику берётся из переменной окружения CLAUDE_BIN (удобно и для
нестандартной установки, и для подстановки заглушки в тестах), иначе ищется в
PATH. stdin закрыт (subprocess.DEVNULL): неинтерактивный вызов не должен иметь
ни малейшего шанса чего-то дождаться с терминала.

Почему есть предел длины промпта. Промпт передаётся аргументом командной
строки, а её размер ограничен ARG_MAX (на macOS это 1 МиБ на всё: аргументы
плюс окружение). Промпт сверх предела не отправляется совсем: иначе получили бы
«Argument list too long» от ядра — сообщение, по которому неочевидно, что делать.
Предел — MAX_PROMPT_BYTES, считается от ARG_MAX с запасом (ARG_MAX_SHARE).
Нарезка текста на окна — задача вызывающего инструмента, здесь только отказ.

Повторы. Повтор делается только там, где он может помочь: таймаут и ненулевой
код возврата. Отсутствие бинарника и неавторизованность повтора не получают —
второй вызов даст то же самое. Пустой ответ тоже не повторяется: claude ответил,
просто сказать ему нечего, и это разбирает вызывающий.

Логи вызовов. При переданном log_dir каждый вызов ask() кладёт два файла:
`<tag>-<номер>.prompt.txt` и `<tag>-<номер>.response.txt`. Номер — первый
свободный, существующие файлы не перетираются, каталог создаётся при
необходимости. Файлы пишутся на каждый вызов, включая неудачный: в response
тогда лежит причина неудачи — иначе по логам нельзя понять, какое окно текста
осталось без пунктуации.

Зависимости: только стандартная библиотека. Ни сетевых клиентов, ни ключей API:
авторизацию держит сам Claude CLI, модуль её не видит и не хранит.

Использование:
  python tools/claude_cli.py --probe
  python tools/claude_cli.py --ask промпт.txt
  python tools/claude_cli.py --selftest

Пути с пробелами, скобками и кириллицей поддерживаются — передавайте в кавычках.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

PROG = "claude_cli.py"

# Имя бинарника и переменная окружения с путём к нему.
BINARY_NAME = "claude"
ENV_BIN = "CLAUDE_BIN"

# Значения по умолчанию публичного API. Саммари — один большой вызов по всей
# записи, ему пяти минут хватает с запасом; окна пунктуации отвечают быстрее.
DEFAULT_TIMEOUT = 300.0
DEFAULT_ATTEMPTS = 2

# Пауза между попытками: повторять мгновенно бессмысленно, если claude споткнулся
# о сеть или о лимит запросов.
RETRY_PAUSE_SEC = 3.0

# Предел длины промпта. ARG_MAX — это лимит на всю командную строку вместе с
# окружением, поэтому берём от него долю: остальное уходит на путь к бинарнику,
# флаг -p и переменные окружения прогона.
ARG_MAX_FALLBACK = 256 * 1024
ARG_MAX_SHARE = 0.25
MIN_PROMPT_BYTES = 32 * 1024


def _system_arg_max() -> int:
    """ARG_MAX системы. Значения нет или оно бессмысленное — берём запасное."""
    try:
        value = os.sysconf("SC_ARG_MAX")
    except (AttributeError, ValueError, OSError):
        return ARG_MAX_FALLBACK
    if not isinstance(value, int) or value <= 0:
        return ARG_MAX_FALLBACK
    return value


# Именованный предел: длина промпта в байтах UTF-8, при которой вызов ещё
# заведомо безопасен. Считается один раз при импорте — чтобы сообщение об отказе
# было одинаковым во всём прогоне.
MAX_PROMPT_BYTES = max(MIN_PROMPT_BYTES, int(_system_arg_max() * ARG_MAX_SHARE))

# По этим признакам в выводе упавшего claude видно, что дело не в сбое, а в
# отсутствии авторизации: повторять нечего, нужен человек с `claude login`.
AUTH_MARKERS = (
    "invalid api key",
    "please run /login",
    "run /login",
    "not logged in",
    "log in to claude",
    "please log in",
    "unauthorized",
    "authentication_error",
    "authentication error",
    "authentication failed",
    "oauth token",
    "credit balance is too low",
    "не авторизован",
)

# Сколько строк вывода claude показывать в сообщении об ошибке.
ERROR_TAIL_LINES = 5

# Номер в именах файлов логов: трёх цифр хватает на окна полной записи, и файлы
# сортируются по порядку вызовов.
LOG_NUMBER_DIGITS = 3

# Tag попадает в имя файла, поэтому разделители путей в нём недопустимы.
TAG_RE = re.compile(r"^[^\s/\\]{1,64}$")


# --------------------------------------------------------------- ошибки ---


class ClaudeError(Exception):
    """Общий предок ошибок вызова claude: str(e) уже готов для человека."""


class ClaudeUnavailable(ClaudeError):
    """Вызывать нечем: бинарника нет, он не запускается или не авторизован."""


class ClaudeFailed(ClaudeError):
    """Вызвали, но ответа нет: таймаут, ненулевой код, пустой или длинный ввод."""


class ToolError(Exception):
    """Ошибка самого CLI (не вызова claude): сообщение на русском, код выхода."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


class UsageError(ToolError):
    """Ошибка в аргументах командной строки: вход непригоден, код 2."""

    def __init__(self, message: str):
        super().__init__(message, code=2)


def die(message: str, code: int = 1) -> int:
    """Печатает понятную ошибку и возвращает код выхода (трейсбека не будет)."""
    print(f"{PROG}: {message}", file=sys.stderr)
    return code


def warn(message: str) -> None:
    print(f"{PROG}: {message}", file=sys.stderr)


# ------------------------------------------------------ поиск бинарника ---


def available() -> tuple[bool, str]:
    """Есть ли чем вызывать claude: (True, путь) или (False, причина).

    Живого вызова не делает — значит и про авторизацию ничего не знает: её
    видно только по ответу самого claude (см. AUTH_MARKERS в ask()).
    """
    override = os.environ.get(ENV_BIN, "").strip()
    if override:
        found = shutil.which(override)
        if found:
            return True, found
        target = Path(override)
        if not target.exists():
            return False, f"{ENV_BIN}={override}: файл не найден"
        if target.is_dir():
            return False, f"{ENV_BIN}={override}: это каталог, а не исполняемый файл"
        return False, (
            f"{ENV_BIN}={override}: файл есть, но он не исполняемый "
            f"(исправить: chmod +x {override})"
        )

    found = shutil.which(BINARY_NAME)
    if found:
        return True, found
    return False, (
        f"{BINARY_NAME} не найден в PATH: установите Claude CLI или укажите путь "
        f"к бинарнику в переменной окружения {ENV_BIN}"
    )


def resolve_binary() -> str:
    """Путь к бинарнику или ClaudeUnavailable с объяснением, чего не хватает."""
    ok, detail = available()
    if not ok:
        raise ClaudeUnavailable(detail)
    return detail


def build_command(binary: str, prompt: str) -> list[str]:
    """Команда вызова — ровно `claude -p <промпт>`, и ничего больше.

    Отдельная функция, чтобы форма команды проверялась селф-тестом: модель здесь
    не задаётся (используется модель по умолчанию, решение владельца), а
    интерактивные и потоковые флаги не передаются никогда.
    """
    return [binary, "-p", prompt]


# ------------------------------------------------------------ логи вызовов ---


@dataclass
class CallLog:
    """Пара файлов «промпт — ответ» для одного вызова ask()."""

    prompt_path: Path
    response_path: Path

    def write_response(self, text: str) -> None:
        _write_new(self.response_path, text)


def _write_new(path: Path, text: str) -> None:
    """Пишет файл, не затирая существующий (режим «x»)."""
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")


def _next_number(directory: Path, tag: str) -> int:
    """Первый номер, под которым ни промпта, ни ответа ещё нет."""
    number = 1
    while True:
        if not (
            _log_path(directory, tag, number, "prompt").exists()
            or _log_path(directory, tag, number, "response").exists()
        ):
            return number
        number += 1


def _log_path(directory: Path, tag: str, number: int, kind: str) -> Path:
    return directory / f"{tag}-{number:0{LOG_NUMBER_DIGITS}d}.{kind}.txt"


def open_log(log_dir: str, tag: str, prompt: str) -> CallLog | None:
    """Заводит пару файлов для вызова и сразу пишет промпт.

    Неудача записи логов не должна рушить сам вызов: пунктуация важнее, чем
    возможность её потом разобрать. Поэтому здесь только предупреждение в stderr.
    """
    directory = Path(log_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        number = _next_number(directory, tag)
        record = CallLog(
            prompt_path=_log_path(directory, tag, number, "prompt"),
            response_path=_log_path(directory, tag, number, "response"),
        )
        _write_new(record.prompt_path, prompt)
    except OSError as exc:
        warn(f"не удалось записать лог вызова в {log_dir}: {exc}")
        return None
    return record


def _log_response(record: CallLog | None, text: str) -> None:
    if record is None:
        return
    try:
        record.write_response(text)
    except OSError as exc:
        warn(f"не удалось записать ответ в {record.response_path}: {exc}")


# ----------------------------------------------------------------- вызов ---


def _tail(text: str, lines: int = ERROR_TAIL_LINES) -> str:
    """Последние строки вывода: в сообщении об ошибке полезен только хвост."""
    rows = [row.strip() for row in (text or "").splitlines() if row.strip()]
    return " | ".join(rows[-lines:])


def _looks_like_auth(*streams: str) -> bool:
    """Похоже ли, что claude отказал из-за авторизации, а не сломался."""
    haystack = " ".join(streams).lower()
    return any(marker in haystack for marker in AUTH_MARKERS)


def _terminate(proc: subprocess.Popen) -> None:
    """Снимает зависший вызов вместе с его потомками.

    Процесс запускается в своей сессии (start_new_session=True), поэтому гасить
    можно всю группу: claude может оставить после себя дочерние процессы, и
    убийство одного родителя оставило бы их висеть до конца прогона.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def _run_once(binary: str, prompt: str, timeout: float) -> tuple[int | None, str, str]:
    """Один запуск. Возвращает (код, stdout, stderr); код None — таймаут."""
    try:
        proc = subprocess.Popen(
            build_command(binary, prompt),
            stdin=subprocess.DEVNULL,  # неинтерактивный вызов: ввода нет вовсе
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise ClaudeUnavailable(f"не удалось запустить {binary}: {exc}") from exc

    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", err or ""
    except subprocess.TimeoutExpired:
        _terminate(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover — после SIGKILL
            out, err = "", ""
        return None, out or "", err or ""


def _check_arguments(timeout: float, attempts: int, tag: str) -> None:
    """Проверка аргументов самого вызова: это ошибка кода, а не claude."""
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise ValueError(f"timeout: ожидалось число секунд, получено {timeout!r}")
    if timeout <= 0:
        raise ValueError(f"timeout должен быть больше нуля, получено {timeout!r}")
    if not isinstance(attempts, int) or isinstance(attempts, bool):
        raise ValueError(f"attempts: ожидалось целое число, получено {attempts!r}")
    if attempts < 1:
        raise ValueError(f"attempts должно быть не меньше 1, получено {attempts!r}")
    if not isinstance(tag, str) or not TAG_RE.match(tag):
        raise ValueError(
            f"tag попадает в имя файла лога: ожидалась короткая строка без "
            f"пробелов и разделителей пути, получено {tag!r}"
        )


def ask(
    prompt: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
    log_dir: str | None = None,
    tag: str = "call",
) -> str:
    """Задаёт claude один вопрос и возвращает текст ответа.

    Поднимает ClaudeUnavailable (вызывать нечем или не авторизован) и
    ClaudeFailed (таймаут, ненулевой код, пустой ответ, слишком длинный промпт).
    Ничего не печатает в stdout и процесс не завершает: что делать с неудачей —
    решает вызывающий инструмент.
    """
    if not isinstance(prompt, str):
        raise ValueError(f"prompt: ожидалась строка, получено {type(prompt).__name__}")
    _check_arguments(timeout, attempts, tag)

    record = open_log(log_dir, tag, prompt) if log_dir else None
    try:
        answer = _ask(prompt, timeout=timeout, attempts=attempts)
    except ClaudeError as exc:
        _log_response(record, f"НЕ ПОЛУЧЕНО\n{exc}")
        raise
    _log_response(record, answer)
    return answer


def _ask(prompt: str, *, timeout: float, attempts: int) -> str:
    """Вся логика вызова без логирования — чтобы лог писался в любом случае."""
    size = len(prompt.encode("utf-8"))
    if size > MAX_PROMPT_BYTES:
        raise ClaudeFailed(
            f"промпт слишком длинный: {len(prompt)} символов ({size} байт в UTF-8), "
            f"предел {MAX_PROMPT_BYTES} байт — это ARG_MAX с запасом, промпт "
            "передаётся аргументом командной строки; порежьте текст на окна"
        )
    if not prompt.strip():
        raise ClaudeFailed("пустой промпт: вызывать claude не с чем")

    # Отсутствие бинарника повтора не получает: ClaudeUnavailable летит сразу.
    binary = resolve_binary()

    problem = ""
    for attempt in range(1, attempts + 1):
        code, out, err = _run_once(binary, prompt, timeout)
        detail = _tail(err) or _tail(out)

        if code is None:
            problem = f"таймаут {timeout:g} с"
        elif code != 0:
            if _looks_like_auth(out, err):
                raise ClaudeUnavailable(
                    f"{BINARY_NAME} не авторизован (код {code})"
                    + (f": {detail}" if detail else "")
                    + " — выполните `claude login` и повторите прогон"
                )
            problem = f"{BINARY_NAME} вернул код {code}" + (f": {detail}" if detail else "")
        else:
            answer = out.strip()
            if answer:
                return answer
            # Код 0 и пустой stdout: иногда так выглядит именно неавторизованность.
            if _looks_like_auth(out, err):
                raise ClaudeUnavailable(
                    f"{BINARY_NAME} не авторизован"
                    + (f": {detail}" if detail else "")
                    + " — выполните `claude login` и повторите прогон"
                )
            # Повторять нечего: claude ответил, просто ответ пустой.
            raise ClaudeFailed(
                f"{BINARY_NAME} вернул пустой ответ (код 0)"
                + (f", stderr: {detail}" if detail else "")
            )

        if attempt < attempts:
            warn(
                f"попытка {attempt} из {attempts} не удалась ({problem}); "
                f"повтор через {RETRY_PAUSE_SEC:g} с"
            )
            time.sleep(RETRY_PAUSE_SEC)

    plural = "попытка" if attempts == 1 else "попыток"
    raise ClaudeFailed(f"{problem} (всего {attempts} {plural})")


# ------------------------------------------------------------------ CLI ---


@dataclass
class Options:
    probe: bool = False
    ask_path: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    attempts: int = DEFAULT_ATTEMPTS
    log_dir: str | None = None
    tag: str = "call"
    selftest: bool = False
    help: bool = False


USAGE = f"""Использование: python tools/claude_cli.py --probe | --ask <файл> | --selftest

Обёртка неинтерактивного вызова Claude CLI. Команда собирается ровно как
`{BINARY_NAME} -p <промпт>`: промпт одним аргументом, модель не задаётся
(используется модель по умолчанию), интерактивные и потоковые флаги не
передаются, stdin закрыт. Путь к бинарнику берётся из переменной окружения
{ENV_BIN}, иначе ищется в PATH.

Этот файл одновременно библиотека: tools/punctuate.py и tools/summarize.py
импортируют из него available(), ask(), ClaudeUnavailable и ClaudeFailed.

Опции:
  --probe              проверить, есть ли чем вызывать claude, и напечатать
                       найденный бинарник. Живого вызова не делает, поэтому
                       про авторизацию не говорит: её видно только по ответу
                       самого claude
  --ask <путь>         прочитать промпт из файла («-» — из stdin), задать его
                       claude и напечатать ответ в stdout (ручная проверка
                       живым claude)
  --timeout <сек>      сколько ждать одну попытку (по умолчанию
                       {DEFAULT_TIMEOUT:g})
  --attempts <N>       сколько попыток делать (по умолчанию {DEFAULT_ATTEMPTS};
                       повтор бывает только на таймауте и ненулевом коде)
  --log-dir <путь>     куда складывать <tag>-<номер>.prompt.txt и
                       <tag>-<номер>.response.txt (каталог создаётся,
                       существующие файлы не перетираются)
  --tag <имя>          префикс имён файлов лога (по умолчанию «call»)
  --selftest           прогнать встроенные проверки на заглушках и выйти
                       (установленный claude не нужен)
  -h, --help           показать эту справку

Предел длины промпта — {MAX_PROMPT_BYTES} байт UTF-8: промпт идёт аргументом
командной строки, и её размер ограничен ARG_MAX. Промпт сверх предела не
отправляется, вызывающий инструмент обязан порезать текст на окна.

Коды возврата: 0 — получилось (--probe: claude доступен), 1 — не получилось
(--probe: claude недоступен; --ask: таймаут, ненулевой код, пустой ответ,
нет бинарника, не авторизован), 2 — ошибка в аргументах или файл промпта не
читается.

Примеры:
  python tools/claude_cli.py --probe
  python tools/claude_cli.py --ask промпт.txt --timeout 120 --log-dir data/logs
  CLAUDE_BIN=/opt/claude/bin/claude python tools/claude_cli.py --probe
  python tools/claude_cli.py --selftest
"""


def read_prompt(source: str) -> str:
    """Промпт из файла или из stdin по «-»."""
    if source == "-":
        return sys.stdin.read()
    path = Path(source)
    if not path.exists():
        raise ToolError(f"--ask: файл не найден: {path}", code=2)
    if path.is_dir():
        raise ToolError(f"--ask: это каталог, а не файл: {path}", code=2)
    try:
        # utf-8-sig: промпт мог приехать из редактора, BOM в начале не нужен.
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ToolError(f"--ask: файл не читается как UTF-8: {path}", code=2) from None


def run_probe() -> int:
    ok, detail = available()
    if ok:
        print(f"{BINARY_NAME} доступен: {detail}")
        print(
            "авторизация здесь не проверяется: она видна только по ответу самого "
            f"{BINARY_NAME} (сделайте пробный --ask)"
        )
        return 0
    print(f"{BINARY_NAME} недоступен: {detail}")
    return 1


def run_ask(opts: Options) -> int:
    assert opts.ask_path is not None  # проверено в main()
    prompt = read_prompt(opts.ask_path)
    try:
        answer = ask(
            prompt,
            timeout=opts.timeout,
            attempts=opts.attempts,
            log_dir=opts.log_dir,
            tag=opts.tag,
        )
    except ClaudeUnavailable as exc:
        return die(f"вызывать нечем: {exc}")
    except ClaudeFailed as exc:
        return die(f"ответ не получен: {exc}")
    print(answer)
    return 0


def _need_value(argv: list[str], index: int, name: str) -> str:
    if index + 1 >= len(argv):
        raise UsageError(f"{name} требует значение")
    return argv[index + 1]


def parse_args(argv: list[str]) -> Options:
    """Разбирает аргументы вручную — ради сообщений на русском, как в соседних tools/."""
    opts = Options()
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("-h", "--help"):
            opts.help = True
            return opts
        if arg == "--selftest":
            opts.selftest = True
            index += 1
            continue
        if arg == "--probe":
            opts.probe = True
            index += 1
            continue
        if arg == "--ask":
            opts.ask_path = _need_value(argv, index, arg)
            index += 2
            continue
        if arg == "--timeout":
            raw = _need_value(argv, index, arg)
            try:
                opts.timeout = float(raw)
            except ValueError:
                raise UsageError(f"--timeout ожидает число секунд, получено: {raw}") from None
            if opts.timeout <= 0:
                raise UsageError(f"--timeout должен быть больше нуля, получено: {raw}")
            index += 2
            continue
        if arg == "--attempts":
            raw = _need_value(argv, index, arg)
            try:
                opts.attempts = int(raw)
            except ValueError:
                raise UsageError(f"--attempts ожидает целое число, получено: {raw}") from None
            if opts.attempts < 1:
                raise UsageError(f"--attempts должно быть не меньше 1, получено: {raw}")
            index += 2
            continue
        if arg == "--log-dir":
            opts.log_dir = _need_value(argv, index, arg)
            index += 2
            continue
        if arg == "--tag":
            opts.tag = _need_value(argv, index, arg)
            if not TAG_RE.match(opts.tag):
                raise UsageError(
                    f"--tag попадает в имя файла: нужна короткая строка без пробелов "
                    f"и разделителей пути, получено: {opts.tag}"
                )
            index += 2
            continue
        raise UsageError(f"неизвестный аргумент: {arg} (см. --help)")
    return opts


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        opts = parse_args(args)
        if opts.help:
            print(USAGE, end="")
            return 0
        if opts.selftest:
            return selftest()
        if opts.probe and opts.ask_path:
            raise UsageError("--probe и --ask вместе не работают: выберите одно")
        if opts.probe:
            return run_probe()
        if opts.ask_path:
            return run_ask(opts)
        raise UsageError("нечего делать: укажите --probe, --ask или --selftest (см. --help)")
    except ToolError as exc:
        return die(str(exc), exc.code)
    except KeyboardInterrupt:
        return die("прервано пользователем")


# ------------------------------------------------------------ селф-тест ---
#
# Установленный claude здесь не нужен и намеренно не используется: все проверки
# идут на подставных sh-скриптах в временном каталоге. Каждая заглушка лежит в
# своём подкаталоге и пишет рядом с собой, сколько раз её позвали и с какими
# аргументами, — так проверяется и форма команды, и число попыток.

STUB_ANSWER = "Ответ подставного claude."

# Преамбула заглушки: записать номер вызова и полученные аргументы.
# Каталог берётся подстановкой оболочки, а не dirname: часть проверок подменяет
# PATH на один каталог с заглушкой, и внешних команд там нет вовсе.
STUB_PREAMBLE = """#!/bin/sh
dir=${0%/*}
printf 'x\\n' >> "$dir/calls.txt"
printf '%s\\n' "$#" > "$dir/argc.txt"
: > "$dir/argv.txt"
for a in "$@"; do printf '%s\\n' "$a" >> "$dir/argv.txt"; done
"""


def _stub(root: Path, name: str, body: str) -> Path:
    """Создаёт подкаталог с исполняемой заглушкой `claude` внутри."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / BINARY_NAME
    path.write_text(STUB_PREAMBLE + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _calls(stub: Path) -> int:
    counter = stub.parent / "calls.txt"
    if not counter.exists():
        return 0
    return len([row for row in counter.read_text(encoding="utf-8").splitlines() if row])


def _argv(stub: Path) -> list[str]:
    recorded = stub.parent / "argv.txt"
    if not recorded.exists():
        return []
    return recorded.read_text(encoding="utf-8").splitlines()


@contextmanager
def _env(**values: str | None):
    """Временно правит os.environ: None — убрать переменную."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _patched(name: str, value):
    """Временно подменяет константу модуля (пауза повторов, предел длины)."""
    saved = globals()[name]
    globals()[name] = value
    try:
        yield
    finally:
        globals()[name] = saved


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Прогон main() с перехватом вывода: возвращает (код возврата, stdout)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
        code = main(argv)
    return code, buffer.getvalue()


def _quiet_ask(prompt: str, **kwargs) -> tuple[str | None, Exception | None, str]:
    """ask() с перехватом вывода: (ответ, исключение, напечатанное в stdout)."""
    buffer = io.StringIO()
    answer: str | None = None
    error: Exception | None = None
    with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
        try:
            answer = ask(prompt, **kwargs)
        except (ClaudeError, ValueError) as exc:
            error = exc
    return answer, error, buffer.getvalue()


def _check_success():
    """Удачный вызов: команда ровно `-p <промпт>`, ответ обрезан по краям."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(
            Path(tmp),
            "ok",
            f'cat > "$dir/stdin.txt"\nprintf \'\\n%s\\n\\n\' "{STUB_ANSWER}"\n',
        )
        with _env(CLAUDE_BIN=str(stub)):
            ok, detail = available()
            assert ok and detail == str(stub), (ok, detail)
            answer, error, out = _quiet_ask("Расставь пунктуацию: привет мир")
        assert error is None, error
        # Крайние переводы строк и пробелы сняты.
        assert answer == STUB_ANSWER, repr(answer)
        # Модуль ничего не печатает в stdout сам.
        assert out == "", repr(out)
        assert _calls(stub) == 1, _calls(stub)
        # Команда: ровно два аргумента, флаг -p и промпт одной строкой.
        assert (stub.parent / "argc.txt").read_text(encoding="utf-8").strip() == "2"
        assert _argv(stub) == ["-p", "Расставь пунктуацию: привет мир"], _argv(stub)
        # stdin закрыт: заглушка прочитала из него пустоту.
        assert (stub.parent / "stdin.txt").read_text(encoding="utf-8") == ""


def _check_command_shape():
    """В команде нет ни модели, ни интерактивных, ни потоковых флагов."""
    cmd = build_command("/usr/local/bin/claude", "промпт")
    assert cmd == ["/usr/local/bin/claude", "-p", "промпт"], cmd
    assert len(cmd) == 3, cmd
    for forbidden in (
        "--model",
        "-m",
        "--output-format",
        "--stream",
        "--stream-json",
        "--verbose",
        "--interactive",
        "-i",
        "--continue",
        "-c",
    ):
        assert forbidden not in cmd, forbidden


def _check_nonzero_code():
    """Ненулевой код: ClaudeFailed, повтор был, в сообщении код и хвост stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(
            Path(tmp),
            "fail",
            "printf 'claude: внутренняя поломка\\n' >&2\nexit 3\n",
        )
        with _env(CLAUDE_BIN=str(stub)), _patched("RETRY_PAUSE_SEC", 0.0):
            answer, error, _ = _quiet_ask("вопрос", attempts=2)
        assert answer is None, answer
        assert isinstance(error, ClaudeFailed), error
        text = str(error)
        assert "код 3" in text, text
        assert "внутренняя поломка" in text, text
        assert "всего 2 попыток" in text, text
        # Повтор делается именно на ненулевом коде.
        assert _calls(stub) == 2, _calls(stub)

        # attempts=1 — ровно один вызов.
        (stub.parent / "calls.txt").unlink()
        with _env(CLAUDE_BIN=str(stub)), _patched("RETRY_PAUSE_SEC", 0.0):
            _, error, _ = _quiet_ask("вопрос", attempts=1)
        assert isinstance(error, ClaudeFailed), error
        assert _calls(stub) == 1, _calls(stub)


def _check_empty_output():
    """Пустой stdout при коде 0: ClaudeFailed и без повтора — claude ответил."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(Path(tmp), "empty", "exit 0\n")
        with _env(CLAUDE_BIN=str(stub)), _patched("RETRY_PAUSE_SEC", 0.0):
            answer, error, _ = _quiet_ask("вопрос", attempts=3)
        assert answer is None, answer
        assert isinstance(error, ClaudeFailed), error
        assert "пустой ответ" in str(error), str(error)
        assert _calls(stub) == 1, _calls(stub)

        # Пробелы и перевод строки — тоже пустой ответ.
        blank = _stub(Path(tmp), "blank", "printf '   \\n\\n'\n")
        with _env(CLAUDE_BIN=str(blank)):
            _, error, _ = _quiet_ask("вопрос", attempts=1)
        assert isinstance(error, ClaudeFailed), error
        assert "пустой ответ" in str(error), str(error)


def _check_timeout():
    """Таймаут: ClaudeFailed с указанием секунд, повтор был, зависший снят."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(Path(tmp), "slow", "sleep 30\n")
        started = time.monotonic()
        with _env(CLAUDE_BIN=str(stub)), _patched("RETRY_PAUSE_SEC", 0.0):
            answer, error, _ = _quiet_ask("вопрос", timeout=0.4, attempts=2)
        spent = time.monotonic() - started
        assert answer is None, answer
        assert isinstance(error, ClaudeFailed), error
        assert "таймаут 0.4 с" in str(error), str(error)
        assert "всего 2 попыток" in str(error), str(error)
        assert _calls(stub) == 2, _calls(stub)
        # Зависший процесс снят, а не дождан: два таймаута по 0,4 с, не 60 с.
        assert spent < 15, spent


def _check_missing_binary():
    """Нет бинарника: ClaudeUnavailable сразу, без повторов и без запуска."""
    with tempfile.TemporaryDirectory() as tmp:
        empty_path = Path(tmp) / "пустой-path"
        empty_path.mkdir()

        # CLAUDE_BIN указывает в пустоту.
        missing = Path(tmp) / "нет-такого" / BINARY_NAME
        with _env(CLAUDE_BIN=str(missing)):
            ok, detail = available()
            assert not ok, detail
            assert "файл не найден" in detail, detail
            answer, error, _ = _quiet_ask("вопрос", attempts=3)
        assert answer is None, answer
        assert isinstance(error, ClaudeUnavailable), error
        assert str(missing) in str(error), str(error)

        # CLAUDE_BIN не задан, а в PATH claude нет.
        with _env(CLAUDE_BIN=None, PATH=str(empty_path)):
            ok, detail = available()
            assert not ok, detail
            assert "не найден в PATH" in detail and ENV_BIN in detail, detail
            _, error, _ = _quiet_ask("вопрос", attempts=3)
        assert isinstance(error, ClaudeUnavailable), error

        # Файл есть, но не исполняемый.
        not_exec = Path(tmp) / "не-исполняемый"
        not_exec.write_text("#!/bin/sh\n", encoding="utf-8")
        not_exec.chmod(0o644)
        with _env(CLAUDE_BIN=str(not_exec)):
            ok, detail = available()
            assert not ok, detail
            assert "не исполняемый" in detail, detail
            _, error, _ = _quiet_ask("вопрос")
        assert isinstance(error, ClaudeUnavailable), error

        # CLAUDE_BIN указывает на каталог.
        with _env(CLAUDE_BIN=str(empty_path)):
            ok, detail = available()
            assert not ok, detail
            assert "каталог" in detail, detail


def _check_binary_sources():
    """CLAUDE_BIN сильнее PATH, пустая переменная равна незаданной."""
    with tempfile.TemporaryDirectory() as tmp:
        from_path = _stub(Path(tmp), "из-path", 'printf \'%s\\n\' "ИЗ PATH"\n')
        from_env = _stub(Path(tmp), "из-env", 'printf \'%s\\n\' "ИЗ CLAUDE_BIN"\n')

        # Бинарник найден в PATH по имени claude.
        with _env(CLAUDE_BIN=None, PATH=str(from_path.parent)):
            ok, detail = available()
            assert ok and detail == str(from_path), (ok, detail)
            answer, error, _ = _quiet_ask("вопрос")
        assert error is None and answer == "ИЗ PATH", (error, answer)

        # CLAUDE_BIN перебивает PATH.
        with _env(CLAUDE_BIN=str(from_env), PATH=str(from_path.parent)):
            answer, error, _ = _quiet_ask("вопрос")
        assert error is None and answer == "ИЗ CLAUDE_BIN", (error, answer)

        # Пустая и пробельная переменная — как будто её нет: берётся PATH.
        for blank in ("", "   "):
            with _env(CLAUDE_BIN=blank, PATH=str(from_path.parent)):
                ok, detail = available()
                assert ok and detail == str(from_path), (blank, ok, detail)


def _check_unauthorized():
    """Признаки неавторизованности: ClaudeUnavailable и ни одного повтора."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(
            Path(tmp),
            "auth",
            "printf 'Invalid API key · Please run /login\\n' >&2\nexit 1\n",
        )
        with _env(CLAUDE_BIN=str(stub)), _patched("RETRY_PAUSE_SEC", 0.0):
            answer, error, _ = _quiet_ask("вопрос", attempts=3)
        assert answer is None, answer
        assert isinstance(error, ClaudeUnavailable), error
        assert "не авторизован" in str(error), str(error)
        assert "claude login" in str(error), str(error)
        # Повторять нечего: вызов был один.
        assert _calls(stub) == 1, _calls(stub)

        # Код 0, пустой stdout и просьба залогиниться в stderr — тот же случай.
        quiet = _stub(Path(tmp), "auth-0", "printf 'Please log in first\\n' >&2\nexit 0\n")
        with _env(CLAUDE_BIN=str(quiet)):
            _, error, _ = _quiet_ask("вопрос", attempts=2)
        assert isinstance(error, ClaudeUnavailable), error
        assert _calls(quiet) == 1, _calls(quiet)

        # Обычная поломка на признаки авторизации не похожа.
        assert not _looks_like_auth("", "claude: segmentation fault")
        assert _looks_like_auth("", "Authentication failed")


def _check_prompt_limit():
    """Длинный промпт не отправляется: ClaudeFailed, бинарник не вызывался."""
    # Предел осмысленный: не меньше нижней границы и не больше ARG_MAX.
    assert MAX_PROMPT_BYTES >= MIN_PROMPT_BYTES, MAX_PROMPT_BYTES
    assert MAX_PROMPT_BYTES <= max(_system_arg_max(), ARG_MAX_FALLBACK), MAX_PROMPT_BYTES

    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(Path(tmp), "ok", f'printf \'%s\\n\' "{STUB_ANSWER}"\n')

        # Настоящий предел: на байт больше — отказ, причём до запуска claude.
        with _env(CLAUDE_BIN=str(stub)):
            _, error, _ = _quiet_ask("я" * (MAX_PROMPT_BYTES // 2 + 1))
        assert isinstance(error, ClaudeFailed), error
        assert _calls(stub) == 0, _calls(stub)

        # Сообщение называет и длину промпта, и предел.
        with _env(CLAUDE_BIN=str(stub)), _patched("MAX_PROMPT_BYTES", 100):
            _, error, _ = _quiet_ask("а" * 80)  # 160 байт UTF-8
        assert isinstance(error, ClaudeFailed), error
        text = str(error)
        assert "80 символов" in text, text
        assert "160 байт" in text, text
        assert "предел 100 байт" in text, text
        assert "ARG_MAX" in text, text
        assert _calls(stub) == 0, _calls(stub)

        # Ровно предел ещё проходит.
        with _env(CLAUDE_BIN=str(stub)), _patched("MAX_PROMPT_BYTES", 100):
            answer, error, _ = _quiet_ask("a" * 100)
        assert error is None and answer == STUB_ANSWER, (error, answer)
        assert _calls(stub) == 1, _calls(stub)

        # Пустой промпт тоже не отправляется.
        with _env(CLAUDE_BIN=str(stub)):
            _, error, _ = _quiet_ask("   \n  ")
        assert isinstance(error, ClaudeFailed), error
        assert "пустой промпт" in str(error), str(error)
        assert _calls(stub) == 1, _calls(stub)


def _check_logs():
    """log_dir: два файла на вызов, нумерация по порядку, ничего не перетёрто."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(Path(tmp), "ok", f'printf \'%s\\n\' "{STUB_ANSWER}"\n')
        # Каталога ещё нет: он должен создаться вместе с родителями.
        log_dir = Path(tmp) / "логи" / "вызовы"

        with _env(CLAUDE_BIN=str(stub)):
            for number in (1, 2):
                answer, error, _ = _quiet_ask(
                    f"промпт {number}", log_dir=str(log_dir), tag="окно"
                )
                assert error is None and answer == STUB_ANSWER, (number, error)

        names = sorted(item.name for item in log_dir.iterdir())
        assert names == [
            "окно-001.prompt.txt",
            "окно-001.response.txt",
            "окно-002.prompt.txt",
            "окно-002.response.txt",
        ], names
        assert (log_dir / "окно-001.prompt.txt").read_text(encoding="utf-8") == "промпт 1\n"
        assert (log_dir / "окно-002.prompt.txt").read_text(encoding="utf-8") == "промпт 2\n"
        assert (
            log_dir / "окно-002.response.txt"
        ).read_text(encoding="utf-8") == STUB_ANSWER + "\n"

        # Чужие файлы с теми же именами не перетираются: номер берётся свободный.
        marker = "не трогать\n"
        (log_dir / "окно-003.prompt.txt").write_text(marker, encoding="utf-8")
        with _env(CLAUDE_BIN=str(stub)):
            _quiet_ask("промпт 4", log_dir=str(log_dir), tag="окно")
        assert (log_dir / "окно-003.prompt.txt").read_text(encoding="utf-8") == marker
        assert (log_dir / "окно-004.prompt.txt").read_text(encoding="utf-8") == "промпт 4\n"

        # Другой tag — своя нумерация с единицы.
        with _env(CLAUDE_BIN=str(stub)):
            _quiet_ask("саммари", log_dir=str(log_dir), tag="саммари")
        assert (log_dir / "саммари-001.prompt.txt").read_text(encoding="utf-8") == "саммари\n"

        # Неудачный вызов тоже оставляет два файла, в ответе — причина.
        broken = _stub(Path(tmp), "fail", "printf 'всё плохо\\n' >&2\nexit 7\n")
        with _env(CLAUDE_BIN=str(broken)), _patched("RETRY_PAUSE_SEC", 0.0):
            _, error, _ = _quiet_ask(
                "промпт неудачи", attempts=1, log_dir=str(log_dir), tag="окно"
            )
        assert isinstance(error, ClaudeFailed), error
        assert (log_dir / "окно-005.prompt.txt").read_text(encoding="utf-8") == "промпт неудачи\n"
        failure = (log_dir / "окно-005.response.txt").read_text(encoding="utf-8")
        assert failure.startswith("НЕ ПОЛУЧЕНО"), failure
        assert "код 7" in failure, failure

        # Без log_dir никаких файлов рядом не появляется.
        before = sorted(item.name for item in Path(tmp).iterdir())
        with _env(CLAUDE_BIN=str(stub)):
            _quiet_ask("без логов")
        assert sorted(item.name for item in Path(tmp).iterdir()) == before


def _check_argument_validation():
    """Негодные timeout/attempts/tag — это ошибка кода, значит ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(Path(tmp), "ok", f'printf \'%s\\n\' "{STUB_ANSWER}"\n')
        with _env(CLAUDE_BIN=str(stub)):
            for kwargs in (
                {"timeout": 0},
                {"timeout": -5},
                {"timeout": "много"},
                {"attempts": 0},
                {"attempts": -1},
                {"attempts": 1.5},
                {"tag": ""},
                {"tag": "окна/вызов"},
                {"tag": "окно 1"},
                {"tag": "../побег"},
            ):
                _, error, _ = _quiet_ask("вопрос", **kwargs)
                assert isinstance(error, ValueError), (kwargs, error)
                assert not isinstance(error, ClaudeError), (kwargs, error)
            # Негодный аргумент отсекается до запуска claude.
            assert _calls(stub) == 0, _calls(stub)
            # Не строка в промпте — тоже ValueError.
            _, error, _ = _quiet_ask(None)
            assert isinstance(error, ValueError), error


def _check_probe_cli():
    """--probe: код 0 с найденным бинарником, код 1 без него, текст по-русски."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(Path(tmp), "ok", f'printf \'%s\\n\' "{STUB_ANSWER}"\n')
        with _env(CLAUDE_BIN=str(stub)):
            code, out = _run_cli(["--probe"])
        assert code == 0, (code, out)
        assert f"{BINARY_NAME} доступен: {stub}" in out, out
        assert "авторизация здесь не проверяется" in out, out
        # --probe живого вызова не делает.
        assert _calls(stub) == 0, _calls(stub)

        with _env(CLAUDE_BIN=str(Path(tmp) / "нет-такого")):
            code, out = _run_cli(["--probe"])
        assert code == 1, (code, out)
        assert f"{BINARY_NAME} недоступен:" in out, out
        assert "файл не найден" in out, out


def _check_ask_cli():
    """--ask печатает ответ в stdout; неудача даёт ненулевой код и пустой stdout."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(Path(tmp), "ok", f'printf \'%s\\n\' "{STUB_ANSWER}"\n')
        prompt_file = Path(tmp) / "промпт.txt"
        prompt_file.write_text("Расставь пунктуацию: привет мир\n", encoding="utf-8")

        with _env(CLAUDE_BIN=str(stub)):
            code, out = _run_cli(["--ask", str(prompt_file)])
        assert code == 0, (code, out)
        assert out.strip() == STUB_ANSWER, repr(out)
        # Промпт уходит как есть, вместе с хвостовым переводом строки файла:
        # сравниваем запись заглушки целиком, построчно её не разложить.
        recorded = (stub.parent / "argv.txt").read_text(encoding="utf-8")
        assert recorded == "-p\nРасставь пунктуацию: привет мир\n\n", repr(recorded)

        # Промпт с BOM читается без BOM в команде.
        bom_file = Path(tmp) / "bom.txt"
        bom_file.write_text("﻿привет", encoding="utf-8")
        with _env(CLAUDE_BIN=str(stub)):
            code, out = _run_cli(["--ask", str(bom_file)])
        assert code == 0, (code, out)
        assert _argv(stub) == ["-p", "привет"], _argv(stub)

        # --log-dir и --tag из CLI доходят до ask().
        log_dir = Path(tmp) / "логи-cli"
        with _env(CLAUDE_BIN=str(stub)):
            code, out = _run_cli(
                ["--ask", str(prompt_file), "--log-dir", str(log_dir), "--tag", "ручной"]
            )
        assert code == 0, (code, out)
        assert (log_dir / "ручной-001.response.txt").exists(), sorted(log_dir.iterdir())

        # Упавший claude: код 1 и ни строчки в stdout.
        broken = _stub(Path(tmp), "fail", "exit 4\n")
        with _env(CLAUDE_BIN=str(broken)), _patched("RETRY_PAUSE_SEC", 0.0):
            code, out = _run_cli(["--ask", str(prompt_file), "--attempts", "1"])
        assert code == 1, (code, out)
        assert out == "", repr(out)

        # Нет бинарника — тоже код 1, но без запуска.
        with _env(CLAUDE_BIN=str(Path(tmp) / "нет-такого")):
            code, out = _run_cli(["--ask", str(prompt_file)])
        assert code == 1, (code, out)
        assert out == "", repr(out)


def _check_stdin_is_closed():
    """Вызов не наследует stdin: заглушка читает из него пустоту.

    Проверка идёт отдельным процессом с данными в stdin: если бы ask() отдавал
    claude свой stdin, заглушка прочитала бы эти данные. Внутри одного процесса
    такое не проверить — stdin у селф-теста уже чей-то.
    """
    with tempfile.TemporaryDirectory() as tmp:
        stub = _stub(
            Path(tmp),
            "ok",
            f'cat > "$dir/stdin.txt"\nprintf \'%s\\n\' "{STUB_ANSWER}"\n',
        )
        prompt_file = Path(tmp) / "промпт.txt"
        prompt_file.write_text("вопрос", encoding="utf-8")
        environment = dict(os.environ, CLAUDE_BIN=str(stub))
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--ask", str(prompt_file)],
            input="ПОДСАДНЫЕ ДАННЫЕ ДЛЯ STDIN\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert proc.stdout.strip() == STUB_ANSWER, repr(proc.stdout)
        got = (stub.parent / "stdin.txt").read_text(encoding="utf-8")
        assert got == "", repr(got)


def _check_help_and_usage():
    """--help — справка на русском со всеми флагами; мусор в аргументах — код 2."""
    code, out = _run_cli(["--help"])
    assert code == 0, code
    assert "Использование:" in out, out
    for flag in (
        "--probe",
        "--ask",
        "--timeout",
        "--attempts",
        "--log-dir",
        "--tag",
        "--selftest",
        "-h, --help",
    ):
        assert flag in out, flag
    # В справке названы и форма команды, и предел длины, и коды возврата.
    assert f"{BINARY_NAME} -p <промпт>" in out, out
    assert ENV_BIN in out, out
    assert str(MAX_PROMPT_BYTES) in out, out
    assert "Коды возврата:" in out, out
    # Справка на русском: латиницей только имена флагов и переменных.
    assert "Опции:" in out and "модель не задаётся" in out, out

    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = Path(tmp) / "промпт.txt"
        prompt_file.write_text("вопрос", encoding="utf-8")
        directory = Path(tmp) / "каталог"
        directory.mkdir()
        for argv in (
            [],
            ["--чепуха"],
            ["--ask"],
            ["--ask", str(Path(tmp) / "нет.txt")],
            ["--ask", str(directory)],
            ["--ask", str(prompt_file), "--timeout"],
            ["--ask", str(prompt_file), "--timeout", "ноль"],
            ["--ask", str(prompt_file), "--timeout", "0"],
            ["--ask", str(prompt_file), "--timeout", "-1"],
            ["--ask", str(prompt_file), "--attempts", "много"],
            ["--ask", str(prompt_file), "--attempts", "0"],
            ["--ask", str(prompt_file), "--tag", "окна/вызов"],
            ["--ask", str(prompt_file), "--log-dir"],
            ["--probe", "--ask", str(prompt_file)],
        ):
            code, _ = _run_cli(argv)
            assert code == 2, (argv, code)


def _check_only_stdlib():
    """Импорты — только стандартная библиотека: ни сетевых клиентов, ни SDK.

    Проверка не косметическая: авторизацию держит сам Claude CLI, и если
    однажды кто-то решит дотянуться до API напрямую (а значит и до ключа), это
    станет видно селф-тестом, а не на продуктовом прогоне.
    """
    allowed = {
        "__future__",
        "io",
        "os",
        "re",
        "shutil",
        "signal",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "contextlib",
        "dataclasses",
        "pathlib",
    }
    source = Path(__file__).resolve().read_text(encoding="utf-8")
    imported = set(re.findall(r"^(?:import|from)\s+([A-Za-z_][\w]*)", source, re.MULTILINE))
    assert imported, "импортов не нашлось — проверка сломана"
    assert imported <= allowed, imported - allowed
    # И модель действительно нигде не задаётся флагом.
    assert "--model" not in build_command("claude", "вопрос"), "модель задана"


CHECKS = (
    ("удачный вызов: ответ, команда `-p <промпт>`, stdin закрыт", _check_success),
    ("в команде нет модели и интерактивных флагов", _check_command_shape),
    ("ненулевой код: ClaudeFailed и повтор", _check_nonzero_code),
    ("пустой stdout: ClaudeFailed без повтора", _check_empty_output),
    ("таймаут: ClaudeFailed, повтор, зависший процесс снят", _check_timeout),
    ("нет бинарника: ClaudeUnavailable без повторов", _check_missing_binary),
    ("CLAUDE_BIN сильнее PATH, пустая переменная равна незаданной", _check_binary_sources),
    ("неавторизован: ClaudeUnavailable без повторов", _check_unauthorized),
    ("предел длины промпта: отказ до запуска claude", _check_prompt_limit),
    ("log_dir: два файла на вызов, номера не перетираются", _check_logs),
    ("негодные timeout/attempts/tag дают ValueError", _check_argument_validation),
    ("--probe: коды 0 и 1, текст по-русски", _check_probe_cli),
    ("--ask печатает ответ в stdout", _check_ask_cli),
    ("stdin не наследуется (проверка отдельным процессом)", _check_stdin_is_closed),
    ("--help на русском, мусор в аргументах — код 2", _check_help_and_usage),
    ("импорты только из стандартной библиотеки", _check_only_stdlib),
)


def selftest() -> int:
    failed = 0
    for name, check_func in CHECKS:
        try:
            check_func()
        except Exception as err:  # noqa: BLE001 — селф-тест обязан дойти до конца
            failed += 1
            print(f"ПРОВАЛ  {name}: {type(err).__name__}: {err}")
        else:
            print(f"ок      {name}")
    print()
    if failed:
        print(f"{PROG}: селф-тест ПРОВАЛЕН, провалено {failed} из {len(CHECKS)} проверок")
        return 1
    print(f"{PROG}: селф-тест пройден, {len(CHECKS)} из {len(CHECKS)} проверок")
    return 0


if __name__ == "__main__":
    sys.exit(main())
