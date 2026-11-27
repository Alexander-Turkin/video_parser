#!/usr/bin/env python3
"""punctuate.py — пунктуация расшифровки окнами через ollama и предложения с таймкодами.

На вход — транскрипт схемы transcript/1 от tools/transcribe.py: сплошной поток
строчными без точек, числа словами (так говорит выбранная модель GigaAM v3_rnnt).
На выход — файл схемы punctuated/1: тот же текст, которому ollama расставил
знаки препинания, заглавные и абзацы, разбитый на предложения, и у каждого
предложения таймкод начала и конца из пословных таймкодов транскрипта.

Главное правило, из которого вырос весь инструмент: ollama не имеет права менять
слова. Он ставит пунктуацию, меняет регистр и расставляет абзацы — и всё. Поэтому
после каждого ответа последовательность слов сверяется с исходной, и расхождение
не «исправляется», а откатывает окно целиком: в файл идёт дословный текст модели
без пунктуации, окно помечается status «fallback» с причиной, а код возврата
инструмента становится ненулевым. Молча отдать текст, в котором модель
«поправила» оговорку или перевела «двадцать пять» в «25», нельзя: расшифровка
обязана остаться расшифровкой, а таймкоды — соответствовать словам.

Почему окнами. Полная запись — это десятки тысяч слов, в один промпт они не
не помещаются в контекст модели. По умолчанию используются окна по 25
сегментов, которые склеиваются строго по порядку. В
промпт каждого окна добавляется последнее предложение предыдущего — как контекст
с пометкой «не выводить», чтобы пунктуация на стыке не рвалась.

Откуда берутся таймкоды предложений. Слова окна выкладываются в один плоский
список: у сегмента с пословными таймкодами берутся они, у сегмента без них —
границы самого сегмента (так бывает с пустыми и короткими чанками). Поскольку
последовательность слов в ответе ollama уже сверена и совпадает слово в слово,
k-е слово ответа — это k-е слово списка: start предложения — время первого его
слова, end — конец последнего. Времена слов зажимаются в границы их сегментов, а
предложения — в [0, source_duration], и порядок не убывает.

Чего инструмент не делает: не ходит за саммари (tools/summarize.py), не собирает
итоговый .md (tools/render_doc.py), не чинит ошибки распознавания и не вызывает
ollama повторно ради лучшей пунктуации. Вызов ollama живёт только в
tools/ollama_client.py, своего subprocess здесь нет.

Зависимости: стандартная библиотека плюс два соседних модуля (ollama_client для
вызова, outputs для разбора транскрипта). Ни torch, ни gigaam, ни ffmpeg здесь не
нужны — весам моделей взяться неоткуда и незачем. Проверки живут в самом файле
(--selftest): pytest в окружении проекта нет.

Использование:
  python tools/punctuate.py --transcript <файл.json> --out <файл.json> [опции]
  python tools/punctuate.py --selftest
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

# ollama_client.py и outputs.py лежат рядом, а tools/ — не пакет: при запуске
# «python tools/punctuate.py» каталог скрипта уже первый в sys.path, но при
# импорте модуля откуда-то ещё — нет. Поэтому путь добавляется явно.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import ollama_client  # noqa: E402 — импорт возможен только после правки sys.path выше
import outputs  # noqa: E402 — то же самое

PROG = "punctuate.py"

# Схема файла на выходе. Это контракт с tools/summarize.py и tools/render_doc.py:
# набор полей менять нельзя, добавлять свои — можно.
SCHEMA = "punctuated/1"

# Схема входа — та же константа, что у tools/outputs.py и tools/transcribe.py.
TRANSCRIPT_SCHEMA = outputs.SCHEMA

# Сегменты транскрипта — чанки по 20–25 с. Окно в 25 сегментов
# помещается в контекст локальной модели и проверяется целиком.
DEFAULT_WINDOW = 25

# Сколько знаков после запятой остаётся у секунд в файле. Транскрипт приходит с
# точностью до миллисекунды, выдумывать точнее нечего.
ROUND_TO = 3

# Разделитель в промпте, после которого идёт текст окна и ничего больше. Нужен
# по двум причинам: ollama видит, где кончаются правила, а селф-тест может
# вытащить из промпта ровно отправленный текст.
TEXT_MARKER = "=== ТЕКСТ ОКНА ==="

# Предупреждения соседнего разборщика транскрипта должны выглядеть так, будто их
# написал работающий инструмент, а не библиотека, из которой он их позвал.
outputs.PROG = PROG


class UsageError(Exception):
    """Ошибка в аргументах командной строки."""


class InputError(Exception):
    """Вход непригоден: нет файла, не та схема, нет сегментов."""


class OllamaGone(Exception):
    """ollama недоступен совсем: остальные окна звать его уже бессмысленно."""


def die(message: str, code: int = 2) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main().

    По умолчанию 2: непригодный вход и ошибки аргументов означают «файл не
    собран», а код 1 здесь занят другим смыслом — файл записан, но в нём есть
    окна в откате.
    """
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return code


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но пользователь должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


HELP = """Использование:
  python tools/punctuate.py --transcript <файл.json> --out <файл.json> [опции]
  python tools/punctuate.py --selftest

Расставляет в расшифровке пунктуацию, заглавные и абзацы силами ollama, не меняя
ни одного слова, и пишет файл схемы %s: предложения с таймкодами
начала и конца, абзацы и окна обработки. Текст уходит в ollama окнами по
--window сегментов, по порядку и без перекрытия; после каждого ответа
последовательность слов сверяется с транскриптом. Если слова не совпали или
ollama не ответил, окно откатывается к дословному тексту модели без пунктуации,
помечается в файле и даёт ненулевой код возврата.

Опции:
  --transcript <файл>   JSON схемы %s от tools/transcribe.py (обязателен)
  --out <файл>          куда записать JSON схемы %s (обязателен)
  --ollama-model <имя>  локальная модель (по умолчанию qwen3:8b)
  --window <N>          сколько сегментов отправлять в одном окне
                        (по умолчанию %d)
  --timeout <сек>       сколько ждать ответа ollama на одно окно
                        (по умолчанию %g)
  --attempts <N>        сколько попыток делать на одно окно (по умолчанию %d;
                        повтор бывает только на таймауте и ненулевом коде)
  --log-dir <каталог>   куда складывать промпты и ответы ollama
                        (через tools/ollama_client.py, по файлу на вызов)
  --selftest            прогнать встроенные проверки и выйти (установленный
                        ollama и веса моделей не нужны)
  -h, --help            показать эту справку

Вызов ollama собирается только в tools/ollama_client.py: локальный HTTP API 127.0.0.1:11434 с
моделью qwen3:8b по умолчанию. Если Ollama или выбранная модель недоступна, файл всё равно
записывается — целиком из дословного текста модели, с пометками в каждом окне.

Коды возврата:
  0  все окна прошли сверку слов
  1  есть окна в откате (слова не совпали, ollama не ответил или недоступен);
     файл записан и валиден
  2  вход непригоден (нет файла, не та схема, нет сегментов) — --out не создаётся

Примеры:
  python tools/punctuate.py --transcript data/output/созвон/transcript.json \\
      --out data/output/созвон/punctuated.json
  python tools/punctuate.py --transcript t.json --out p.json --window 60 \\
      --log-dir data/output/созвон/ollama
""" % (
    SCHEMA,
    TRANSCRIPT_SCHEMA,
    SCHEMA,
    DEFAULT_WINDOW,
    ollama_client.DEFAULT_TIMEOUT,
    ollama_client.DEFAULT_ATTEMPTS,
)


# --------------------------------------------------------------- промпт ---

# Промпт по-русски и запрещает ровно то, что ломает расшифровку: любое изменение
# состава и порядка слов, перевод чисел между словами и цифрами, «исправление»
# оговорок и приписки от себя. Разрешены только пунктуация, регистр и абзацы.
PROMPT_RULES = """Ты расставляешь пунктуацию в расшифровке устной речи на русском языке.

Запрещено:
- менять слова: ни одно слово нельзя заменить другим, в том числе синонимом,
  исправленной формой или сокращением;
- добавлять слова, которых нет в тексте, и удалять те, которые есть;
- переводить числа из слов в цифры и из цифр в слова: «двадцать пять» остаётся
  словами, «25» остаётся цифрами;
- исправлять оговорки, повторы, обрывки фраз и ошибки речи — это расшифровка,
  в ней всё сказанное остаётся как сказано;
- переставлять слова местами и менять порядок фраз;
- дописывать комментарии, пояснения, заголовки, приветствия, выводы и любой
  текст от себя.

Разрешено только это:
- ставить знаки пунктуации;
- менять регистр букв;
- разбивать текст на абзацы пустой строкой по смысловым переходам.

Ответ — только обработанный текст и ничего кроме него: без вступления, без
пояснений, без кавычек вокруг текста, без markdown-разметки."""

PROMPT_NO_CONTEXT = """Текст ниже — начало записи."""

PROMPT_CONTEXT = """Предыдущий кусок записи закончился предложением:
«%s»
Оно дано только для согласования пунктуации на стыке — не выводить его в ответе,
обрабатывать нужно лишь текст после разделителя."""


def build_prompt(text: str, previous_sentence: str | None) -> str:
    """Собирает промпт окна: правила, контекст предыдущего окна, текст."""
    context = (
        PROMPT_CONTEXT % previous_sentence.strip()
        if previous_sentence and previous_sentence.strip()
        else PROMPT_NO_CONTEXT
    )
    return "%s\n\n%s\n\n%s\n%s\n" % (PROMPT_RULES, context, TEXT_MARKER, text)


# ------------------------------------------------- слова и их нормализация ---


def normalize_word(token: str) -> str:
    """Нормализованная форма слова для сверки.

    Нижний регистр, ё→е, выкидывается всё, кроме букв и цифр. Именно в таком
    виде сравниваются обе последовательности: пунктуация и заглавные — это
    разрешённые ollama правки, а состав слов — нет. «Привет!» и «привет» равны,
    «ещё» и «еще» равны, а «двадцать пять» и «25» — уже нет.
    """
    lowered = token.lower().replace("ё", "е")
    return "".join(char for char in lowered if char.isalnum())


def tokenize(text: str) -> list[tuple[str, str]]:
    """Текст -> список пар (слово как есть, нормализованная форма).

    Токены, от которых после нормализации ничего не осталось (одинокое тире,
    кавычки), отбрасываются: сравнивать их не с чем.
    """
    pairs = []
    for raw in text.split():
        norm = normalize_word(raw)
        if norm:
            pairs.append((raw, norm))
    return pairs


def first_mismatch(
    reference: list[tuple[str, str]], produced: list[tuple[str, str]]
) -> tuple[int, str, str] | None:
    """Первое расхождение двух последовательностей слов.

    Возвращает (позиция с единицы, было, стало) или None, если совпали. Пропажа
    и лишнее слово в конце тоже расхождение: тогда с одной стороны стоит «—».
    """
    for position, (ref, got) in enumerate(zip(reference, produced), start=1):
        if ref[1] != got[1]:
            return position, ref[0], got[0]
    if len(reference) == len(produced):
        return None
    position = min(len(reference), len(produced)) + 1
    was = reference[position - 1][0] if position <= len(reference) else "—"
    now = produced[position - 1][0] if position <= len(produced) else "—"
    return position, was, now


# -------------------------------------------------- предложения и абзацы ---

# Граница предложения: один или несколько знаков из «.!?…», за ними возможные
# закрывающие кавычки и скобки, пробел — и дальше заглавная буква или цифра
# (перед ней допустимы открывающие кавычки и скобки). Строчная буква после точки
# границей не считается: так «и т. д. потом» не разрывается посередине.
SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?…]+[»\"'”’)\]]*\s+(?=[«\"'“‘(\[]*[0-9A-ZА-ЯЁ])"
)

# Абзацы в ответе ollama разделены пустой строкой — так и просит промпт.
PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n+")


def has_content(text: str) -> bool:
    """Есть ли в тексте хоть одна буква или цифра."""
    return any(char.isalnum() for char in text)


def split_sentences(paragraph: str) -> list[str]:
    """Абзац -> список предложений.

    Внутренние переводы строк и повторные пробелы сжимаются в один пробел:
    абзац в итоговом документе — это строки предложений, и переносы из ответа
    ollama в них не нужны. Кусок без букв и цифр отдельным предложением не
    становится, но и не теряется: он прирастает к соседнему — иначе сверка слов
    увидела бы расхождение там, где претензий к ollama нет.
    """
    text = " ".join(paragraph.split())
    if not text:
        return []

    sentences: list[str] = []
    buffer = ""
    position = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        buffer += text[position:match.end()]
        position = match.end()
        if has_content(buffer):
            sentences.append(buffer.strip())
            buffer = ""
    buffer += text[position:]
    buffer = buffer.strip()
    if buffer:
        if has_content(buffer) or not sentences:
            sentences.append(buffer)
        else:
            sentences[-1] = ("%s %s" % (sentences[-1], buffer)).strip()
    return [sentence for sentence in sentences if has_content(sentence)]


def split_paragraphs(answer: str) -> list[list[str]]:
    """Ответ ollama -> абзацы, каждый из предложений. Пустые абзацы выпадают."""
    paragraphs = []
    for block in PARAGRAPH_SPLIT_RE.split(answer.strip()):
        sentences = split_sentences(block)
        if sentences:
            paragraphs.append(sentences)
    return paragraphs


# ------------------------------------------------------- окна транскрипта ---


@dataclass
class RefWord:
    """Слово транскрипта с временем и номером сегмента, из которого оно взято."""

    raw: str
    norm: str
    start: float
    end: float
    segment: int


@dataclass
class Window:
    """Окно обработки: подряд идущие сегменты и их слова в одном списке."""

    index: int
    segments: list[outputs.Segment]
    words: list[RefWord]

    @property
    def first_segment(self) -> int:
        return self.segments[0].index

    @property
    def last_segment(self) -> int:
        return self.segments[-1].index

    @property
    def text(self) -> str:
        """Дословный текст окна для промпта: слова через пробел, без пунктуации."""
        return " ".join(word.raw for word in self.words)


def _seconds(milliseconds: int) -> float:
    """Миллисекунды разборщика транскрипта -> секунды файла punctuated."""
    return round(milliseconds / 1000.0, ROUND_TO)


def _clamp(value: float, low: float, high: float) -> float:
    """Зажимает значение в [low, high]; перевёрнутые границы не считаются."""
    if high < low:
        return low
    return min(max(value, low), high)


def segment_words(segment: outputs.Segment) -> list[RefWord]:
    """Слова сегмента с временами, зажатыми в его границы.

    У сегмента без пословных таймкодов (пустой или слишком короткий чанк) все
    слова получают границы самого сегмента: это грубее, но честнее, чем выдумать
    равномерную раскладку по секундам, которой в распознавании не было.
    """
    start = _seconds(segment.start)
    end = max(start, _seconds(segment.end))

    words: list[RefWord] = []
    if segment.words:
        for word in segment.words:
            word_start = _clamp(_seconds(word.start), start, end)
            word_end = _clamp(_seconds(word.end), word_start, end)
            words.append(
                RefWord(word.text, normalize_word(word.text), word_start, word_end, segment.index)
            )
    else:
        for raw in segment.text.split():
            words.append(RefWord(raw, normalize_word(raw), start, end, segment.index))

    # Слово, от которого после нормализации ничего не осталось, в сверке не
    # участвует — значит и в тексте окна ему делать нечего.
    return [word for word in words if word.norm]


def build_windows(segments: list[outputs.Segment], size: int) -> list[Window]:
    """Режет сегменты на окна по порядку, без перекрытия, по size сегментов."""
    windows = []
    for index, at in enumerate(range(0, len(segments), size)):
        chunk = segments[at:at + size]
        words: list[RefWord] = []
        for segment in chunk:
            words.extend(segment_words(segment))
        windows.append(Window(index=index, segments=chunk, words=words))
    return windows


# ------------------------------------------------------- обработка окна ---


@dataclass
class DraftSentence:
    """Предложение до сборки файла: текст и таймкоды в секундах."""

    text: str
    start: float
    end: float
    fallback: bool


@dataclass
class WindowResult:
    """Чем закончилось окно: абзацы с предложениями, итог сверки и причина."""

    index: int
    first_segment: int
    last_segment: int
    status: str
    reason: str
    paragraphs: list[list[DraftSentence]] = field(default_factory=list)

    @property
    def fallback(self) -> bool:
        return self.status == "fallback"

    @property
    def last_sentence(self) -> str | None:
        """Последнее предложение окна — контекст для промпта следующего."""
        if not self.paragraphs or not self.paragraphs[-1]:
            return None
        return self.paragraphs[-1][-1].text


def verbatim_result(window: Window, reason: str) -> WindowResult:
    """Окно в откате: дословный текст сегментов, по предложению на сегмент.

    Пунктуации здесь нет и не будет — это ровно то, что выдала модель
    распознавания. Сегменты без текста предложениями не становятся: пустая
    строка в документе не нужна никому.
    """
    sentences = []
    for segment in window.segments:
        text = " ".join(segment.text.split())
        if not has_content(text):
            continue
        start = _seconds(segment.start)
        sentences.append(
            DraftSentence(
                text=text, start=start, end=max(start, _seconds(segment.end)), fallback=True
            )
        )
    return WindowResult(
        index=window.index,
        first_segment=window.first_segment,
        last_segment=window.last_segment,
        status="fallback",
        reason=reason,
        paragraphs=[sentences] if sentences else [],
    )


def assign_times(window: Window, paragraphs: list[list[str]]) -> list[list[DraftSentence]]:
    """Раскладывает таймкоды по предложениям окна.

    Вызывается только после успешной сверки, поэтому k-е слово ответа — это k-е
    слово окна: предложение забирает столько слов, сколько в нём непустых
    токенов, start берётся у первого слова, end — у последнего.
    """
    cursor = 0
    out: list[list[DraftSentence]] = []
    for sentences in paragraphs:
        block: list[DraftSentence] = []
        for sentence in sentences:
            count = len(tokenize(sentence))
            if count == 0:  # pragma: no cover — split_sentences такого не отдаёт
                continue
            words = window.words[cursor:cursor + count]
            cursor += count
            block.append(
                DraftSentence(
                    text=sentence,
                    start=words[0].start,
                    end=max(words[0].start, words[-1].end),
                    fallback=False,
                )
            )
        if block:
            out.append(block)
    return out


def process_window(
    window: Window,
    previous_sentence: str | None,
    *,
    timeout: float,
    attempts: int,
    log_dir: str | None,
    model: str = ollama_client.DEFAULT_MODEL,
) -> WindowResult:
    """Одно окно: промпт, вызов ollama, сверка слов и таймкоды.

    Поднимает OllamaGone, если ollama недоступен совсем (нет бинарника, не
    авторизован) — звать его по следующим окнам в таком прогоне бессмысленно.
    Любая другая неудача и любое расхождение слов превращаются в откат окна, а
    не в исключение: остальные окна обрабатываются как обычно.
    """
    if not window.words:
        # Все сегменты окна пустые (так бывает с тишиной): просить пунктуацию не
        # для чего, и терять нечего — откатом это не считается.
        return WindowResult(
            index=window.index,
            first_segment=window.first_segment,
            last_segment=window.last_segment,
            status="ok",
            reason="в окне нет слов: сегменты пустые",
        )

    prompt = build_prompt(window.text, previous_sentence)
    try:
        answer = ollama_client.ask(
            prompt,
            model=model,
            timeout=timeout,
            attempts=attempts,
            log_dir=log_dir,
            tag="окно%03d" % window.index,
        )
    except ollama_client.OllamaUnavailable as exc:
        raise OllamaGone(str(exc)) from None
    except ollama_client.OllamaFailed as exc:
        warn("окно %d: ollama не ответил: %s" % (window.index, exc))
        return verbatim_result(window, "ollama не ответил: %s" % exc)

    paragraphs = split_paragraphs(answer)
    produced = tokenize(" ".join(
        sentence for sentences in paragraphs for sentence in sentences
    ))
    reference = [(word.raw, word.norm) for word in window.words]

    mismatch = first_mismatch(reference, produced)
    if mismatch is not None:
        position, was, now = mismatch
        sys.stderr.write(
            "%s: окно %d: слова не совпали с транскриптом на позиции %d: «%s» → «%s» "
            "(слов в транскрипте %d, в ответе ollama %d)\n"
            % (PROG, window.index, position, was, now, len(reference), len(produced))
        )
        return verbatim_result(
            window,
            "слова не совпали с транскриптом (позиция %d: «%s» → «%s»)" % (position, was, now),
        )

    return WindowResult(
        index=window.index,
        first_segment=window.first_segment,
        last_segment=window.last_segment,
        status="ok",
        reason="",
        paragraphs=assign_times(window, paragraphs),
    )


# ---------------------------------------------------------- сборка файла ---


@dataclass
class Report:
    """Итоги прогона: то же, что печатается в stdout и решает код возврата."""

    windows: int = 0
    fallback_windows: list[WindowResult] = field(default_factory=list)
    paragraphs: int = 0
    sentences: int = 0
    words: int = 0


def build_payload(
    results: list[WindowResult],
    *,
    source: str,
    source_duration: float | None,
    transcript_path: str,
    model: str,
) -> tuple[dict, Report]:
    """Собирает объект схемы punctuated/1 из результатов окон.

    Склейка строго в порядке окон, нумерация предложений и абзацев сквозная.
    Здесь же держатся обещания про таймкоды: они не убывают и лежат внутри
    [0, source_duration]. Границы сегментов уже соблюдены на уровне слов (см.
    segment_words), и единственный случай, когда предложение может из них
    выйти, — немонотонный транскрипт; про него говорится предупреждением.
    """
    report = Report(windows=len(results))
    limit = source_duration if source_duration and source_duration > 0 else None

    sentences_out: list[dict] = []
    paragraphs_out: list[dict] = []
    windows_out: list[dict] = []
    paragraph_texts: list[str] = []

    previous_start = 0.0
    shifted = 0
    for result in results:
        windows_out.append(
            {
                "index": result.index,
                "segments": [result.first_segment, result.last_segment],
                "status": result.status,
                "reason": result.reason,
            }
        )
        if result.fallback:
            report.fallback_windows.append(result)

        for block in result.paragraphs:
            if not block:
                continue
            first_index = len(sentences_out)
            texts: list[str] = []
            for sentence in block:
                start = sentence.start if limit is None else _clamp(sentence.start, 0.0, limit)
                start = max(start, 0.0)
                end = sentence.end if limit is None else _clamp(sentence.end, 0.0, limit)
                if start < previous_start:
                    # Транскрипт не монотонен: честнее подтянуть предложение к
                    # предыдущему, чем отдать документ, в котором таймкоды
                    # прыгают назад.
                    start = previous_start
                    shifted += 1
                end = max(end, start)
                previous_start = start

                sentences_out.append(
                    {
                        "index": len(sentences_out),
                        "start": round(start, ROUND_TO),
                        "end": round(end, ROUND_TO),
                        "text": sentence.text,
                        "window": result.index,
                        "fallback": sentence.fallback,
                    }
                )
                texts.append(sentence.text)

            paragraphs_out.append(
                {
                    "index": len(paragraphs_out),
                    "window": result.index,
                    "sentences": [first_index, len(sentences_out) - 1],
                }
            )
            paragraph_texts.append(" ".join(texts))

    if shifted:
        warn(
            "таймкоды %d предложений подтянуты вперёд: в транскрипте слова идут не "
            "по возрастанию времени" % shifted
        )

    text = "\n\n".join(paragraph_texts)
    report.paragraphs = len(paragraphs_out)
    report.sentences = len(sentences_out)
    report.words = len(text.split())

    payload = {
        "schema": SCHEMA,
        "source": source,
        "source_duration": None if source_duration is None else round(source_duration, ROUND_TO),
        "transcript": transcript_path,
        "model": model,
        "windows": windows_out,
        "paragraphs": paragraphs_out,
        "sentences": sentences_out,
        "text": text,
        "stats": {
            "windows": report.windows,
            "fallback_windows": len(report.fallback_windows),
            "paragraphs": report.paragraphs,
            "sentences": report.sentences,
            "words": report.words,
        },
    }
    return payload, report


def punctuate(
    windows: list[Window],
    *,
    timeout: float,
    attempts: int,
    log_dir: str | None,
    model: str = ollama_client.DEFAULT_MODEL,
) -> list[WindowResult]:
    """Проходит окна по порядку, передавая следующему контекст предыдущего."""
    available, detail = ollama_client.available(model)
    gone: str | None = None if available else detail
    if gone is not None:
        warn("ollama недоступен (%s) — пунктуации не будет, в файл пойдёт дословный текст" % gone)

    results: list[WindowResult] = []
    previous_sentence: str | None = None
    for window in windows:
        if gone is not None:
            results.append(verbatim_result(window, "ollama недоступен: %s" % gone))
        else:
            try:
                result = process_window(
                    window,
                    previous_sentence,
                    timeout=timeout,
                    attempts=attempts,
                    log_dir=log_dir,
                    model=model,
                )
            except OllamaGone as exc:
                gone = str(exc)
                warn(
                    "ollama недоступен (%s) — остальные окна остаются без пунктуации" % gone
                )
                result = verbatim_result(window, "ollama недоступен: %s" % gone)
            results.append(result)
        if results[-1].last_sentence:
            previous_sentence = results[-1].last_sentence
    return results


# ------------------------------------------------------------ ввод-вывод ---


def read_transcript(path: str) -> tuple[outputs.Transcript, str, str]:
    """Читает транскрипт схемы transcript/1: (разбор, source записи, модель).

    Структуру проверяет tools/outputs.parse_transcript — тот же разборщик, что у
    рендера .txt и .srt, чтобы требования к транскрипту не разъехались между
    инструментами. Поля source и model в нём не нужны и не хранятся, поэтому
    берутся из разобранного JSON напрямую.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise InputError("файл транскрипта не найден: %s" % path) from None
    except IsADirectoryError:
        raise InputError("вместо файла транскрипта указан каталог: %s" % path) from None
    except PermissionError:
        raise InputError("файл транскрипта недоступен для чтения: %s" % path) from None
    except UnicodeDecodeError:
        raise InputError("файл транскрипта не в кодировке UTF-8: %s" % path) from None
    except json.JSONDecodeError as exc:
        raise InputError(
            "файл транскрипта не разбирается как JSON: %s (строка %d, символ %d: %s)"
            % (path, exc.lineno, exc.colno, exc.msg)
        ) from None
    except OSError as exc:
        raise InputError(
            "не удалось прочитать транскрипт: %s (%s)" % (path, exc.strerror or exc)
        ) from None

    try:
        transcript = outputs.parse_transcript(data, source=path)
    except outputs.TranscriptError as exc:
        raise InputError(str(exc)) from None

    source = data.get("source")
    model = data.get("model")
    return (
        transcript,
        source if isinstance(source, str) else "",
        model if isinstance(model, str) else "",
    )


def write_payload(path: str, payload: dict) -> None:
    """Пишет punctuated.json: UTF-8 без эскейпов, отступ 2, перевод строки в конце."""
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise InputError(
            "не удалось записать файл %s (%s)" % (path, exc.strerror or exc)
        ) from None


def print_report(path: str, report: Report) -> None:
    """Краткий отчёт в stdout — он же попадает в лог прогона."""
    sys.stdout.write("================ ПУНКТУАЦИЯ ================\n")
    sys.stdout.write("Окон:          %d\n" % report.windows)
    sys.stdout.write("В откате:      %d\n" % len(report.fallback_windows))
    for result in report.fallback_windows:
        sys.stdout.write(
            "  * окно %d, сегменты %d–%d: %s\n"
            % (result.index, result.first_segment, result.last_segment, result.reason)
        )
    sys.stdout.write("Абзацев:       %d\n" % report.paragraphs)
    sys.stdout.write("Предложений:   %d\n" % report.sentences)
    sys.stdout.write("Слов:          %d\n" % report.words)
    sys.stdout.write("Файл:          %s\n" % path)


# --------------------------------------------------- разбор аргументов ---

_FLAGS_VALUE = {
    "--transcript": "transcript",
    "--out": "out",
    "--window": "window",
    "--timeout": "timeout",
    "--attempts": "attempts",
    "--log-dir": "log_dir",
    "--ollama-model": "ollama_model",
}


def parse_args(argv: list[str]) -> dict:
    """Разбирает аргументы так же, как остальные инструменты проекта.

    Своим разбором, а не argparse: тот печатает usage и свои ошибки
    по-английски, а весь вывод инструментов проекта — на русском.
    """
    opts: dict = {
        "transcript": None,
        "out": None,
        "window": DEFAULT_WINDOW,
        "timeout": ollama_client.DEFAULT_TIMEOUT,
        "attempts": ollama_client.DEFAULT_ATTEMPTS,
        "log_dir": None,
        "ollama_model": ollama_client.DEFAULT_MODEL,
        "help": False,
        "selftest": False,
    }

    position = 0
    while position < len(argv):
        token = argv[position]
        inline: str | None = None
        if token.startswith("--") and "=" in token:
            token, _, inline = token.partition("=")

        if token in ("-h", "--help", "--selftest"):
            if inline is not None:
                raise UsageError("%s не принимает значение" % token)
            opts["help" if token != "--selftest" else "selftest"] = True
            position += 1
            continue

        if token in _FLAGS_VALUE:
            if inline is None:
                if position + 1 >= len(argv):
                    raise UsageError("%s требует значение" % token)
                value = argv[position + 1]
                position += 2
            else:
                value = inline
                position += 1
            if not value:
                raise UsageError("%s требует непустое значение" % token)
            name = _FLAGS_VALUE[token]
            if name == "window":
                opts[name] = _as_int(token, value, minimum=1)
            elif name == "attempts":
                opts[name] = _as_int(token, value, minimum=1)
            elif name == "timeout":
                opts[name] = _as_float(token, value)
            else:
                opts[name] = value
            continue

        if token.startswith("-"):
            raise UsageError("неизвестная опция: %s (см. --help)" % token)
        raise UsageError("лишний аргумент: %s (см. --help)" % token)

    return opts


def _as_int(flag: str, value: str, *, minimum: int) -> int:
    try:
        number = int(value)
    except ValueError:
        raise UsageError("%s ожидает целое число, получено: %s" % (flag, value)) from None
    if number < minimum:
        raise UsageError(
            "%s должно быть не меньше %d, получено: %s" % (flag, minimum, value)
        )
    return number


def _as_float(flag: str, value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise UsageError("%s ожидает число секунд, получено: %s" % (flag, value)) from None
    if not math.isfinite(number) or number <= 0:
        raise UsageError("%s должен быть больше нуля, получено: %s" % (flag, value))
    return number


def check_args(opts: dict) -> None:
    """Проверяет набор аргументов целиком — то, что нельзя проверить по одному."""
    if not opts["transcript"]:
        raise UsageError("не указан --transcript (см. --help)")
    if not opts["out"]:
        raise UsageError("не указан --out (см. --help)")
    # Перетереть транскрипт — это потерять часы распознавания: собрать его
    # заново нечем, а punctuated.json из него выводится за минуты.
    if os.path.abspath(opts["out"]) == os.path.abspath(opts["transcript"]):
        raise UsageError(
            "--out указывает на тот же файл, что и --transcript: %s" % opts["out"]
        )


# ------------------------------------------------------------- селф-тест ---
#
# Установленный ollama здесь не нужен и намеренно не используется: все вызовы
# идут на подставной ollama — скрипт на PATH, который механически расставляет
# точки и заглавные, а по указанию из mode.txt портит слова. Так проверяется и
# удачное окно, и каждый вид расхождения, и полное отсутствие ollama.

# Заглушка — это пара файлов: исполняемый `ollama` на /bin/sh и питонья начинка
# рядом с ним. Шебанг с путём к интерпретатору здесь не годится: путь к проекту
# может содержать пробел (а он содержит), и ядро разорвало бы шебанг по нему.
# Обёртка же запускает интерпретатор по абсолютному пути в кавычках, а каталог
# берёт подстановкой оболочки ${0%/*} — внешних команд вроде dirname не нужно,
# часть проверок подменяет PATH целиком.
_STUB_WRAPPER = '''#!/bin/sh
exec "%(python)s" "${0%%/*}/ollama.py" "$@"
'''

# Промпт каждого вызова сохраняется рядом с заглушкой — по нему проверяется и
# состав правил, и контекст предыдущего окна.
_STUB_SOURCE = '''import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
MARKER = %(marker)r
WORDS_PER_SENTENCE = 7
SENTENCES_PER_PARAGRAPH = 3


def main():
    args = sys.argv[1:]
    number = len(list(HERE.glob("prompt-*.txt"))) + 1
    prompt = args[1] if len(args) > 1 and args[0] == "-p" else ""
    (HERE / ("prompt-%%03d.txt" %% number)).write_text(prompt, encoding="utf-8")

    rows = (HERE / "mode.txt").read_text(encoding="utf-8").splitlines()
    modes = [row.strip() for row in rows if row.strip()]
    mode = modes[number - 1] if number <= len(modes) else modes[-1]

    words = prompt.split(MARKER)[-1].split()
    if mode == "изменить слово" and len(words) > 2:
        words[2] = "выдумка"
    elif mode == "добавить слова":
        words = words + ["и", "ещё", "пара", "слов", "от", "меня"]
    elif mode == "потерять слово" and len(words) > 1:
        del words[1]
    elif mode == "ненулевой код":
        sys.stderr.write("подставной ollama: так и было задумано\\n")
        sys.exit(9)

    sentences = []
    for at in range(0, len(words), WORDS_PER_SENTENCE):
        chunk = words[at:at + WORDS_PER_SENTENCE]
        if not chunk:
            continue
        body = " ".join(chunk)
        sentences.append(body[0].upper() + body[1:] + ".")
    answer = "\\n\\n".join(
        " ".join(sentences[at:at + SENTENCES_PER_PARAGRAPH])
        for at in range(0, len(sentences), SENTENCES_PER_PARAGRAPH)
    )
    if mode == "приписка":
        answer = "Готово! Вот текст с пунктуацией:\\n\\n" + answer
    sys.stdout.write(answer + "\\n")


main()
'''

# Слова фикстуры: обычная устная речь, с «ещё» (проверка ё→е) и числами словами
# (проверка запрета на цифры — заглушка их не трогает, но сверка обязана держать
# их за слова).
_POOL = [
    "итак", "давайте", "разберём", "как", "работает", "окно", "пунктуации", "ещё",
    "раз", "смотрите", "двадцать", "пять", "процентов", "текста", "уходит", "в",
    "откат", "когда", "модель", "меняет", "слова", "поэтому", "мы", "сверяем",
    "последовательность", "слов", "после", "каждого", "ответа", "и", "держим",
    "таймкоды", "на", "месте", "без", "исключений", "вот", "так", "примерно",
    "это", "и", "выглядит", "в", "работе", "каждый", "день", "у", "нас",
]

_SELFTEST_WORDS_PER_SEGMENT = 8


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@contextmanager
def _env(**values: str | None):
    """Временно правит os.environ: None — убрать переменную."""
    # Existing CLI fixtures are used only inside this self-test.
    if "PATH" in values and "OLLAMA_BIN" in values:
        values["OLLAMA_TEST_BIN"] = os.path.join(values["PATH"].split(os.pathsep)[0], "ollama")
        values.pop("OLLAMA_BIN")
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


def _make_stub(root: Path, modes: list[str]) -> Path:
    """Создаёт каталог с подставным ollama и файлом режимов. Возвращает каталог."""
    directory = root / "подставной-ollama"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ollama.py").write_text(
        _STUB_SOURCE % {"marker": TEXT_MARKER}, encoding="utf-8"
    )
    stub = directory / "ollama"
    stub.write_text(_STUB_WRAPPER % {"python": sys.executable}, encoding="utf-8")
    stub.chmod(0o755)
    (directory / "mode.txt").write_text("\n".join(modes) + "\n", encoding="utf-8")
    return directory


def _stub_prompts(directory: Path) -> list[str]:
    """Промпты всех вызовов подставного ollama, по порядку."""
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("prompt-*.txt"))
    ]


def _fixture_transcript(segments: int = 7, *, without_words: tuple[int, ...] = (5,)) -> dict:
    """Транскрипт схемы transcript/1 в памяти: сегменты по 8 слов с таймкодами.

    Сегмент k занимает [k*10, k*10+9] секунд, слова идут по одному в секунду.
    Сегменты из without_words остаются без пословных таймкодов — это случай
    «границы берутся у сегмента».
    """
    items = []
    pool_at = 0
    for index in range(segments):
        start = index * 10.0
        words = []
        for position in range(_SELFTEST_WORDS_PER_SEGMENT):
            word = _POOL[pool_at % len(_POOL)]
            pool_at += 1
            words.append(
                {
                    "text": word,
                    "start": round(start + position, 3),
                    "end": round(start + position + 0.8, 3),
                }
            )
        item = {
            "index": index,
            "start": start,
            "end": start + 9.0,
            "text": " ".join(word["text"] for word in words),
        }
        item["words"] = [] if index in without_words else words
        items.append(item)

    return {
        "schema": TRANSCRIPT_SCHEMA,
        "source": "data/output/созвон команды/созвон команды.webm",
        "source_duration": segments * 10.0,
        "model": "v3_rnnt",
        "device": "cpu",
        "text": " ".join(item["text"] for item in items),
        "metrics": {"chunks": segments},
        "segments": items,
    }


def _write_json(path: Path, payload: object) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Сквозной прогон main() с перехватом вывода."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _transcript_words(data: dict) -> list[str]:
    """Нормализованная последовательность слов транскрипта — эталон для сверки."""
    words = []
    for segment in data["segments"]:
        source = (
            [word["text"] for word in segment["words"]]
            if segment.get("words")
            else segment["text"].split()
        )
        words.extend(normalize_word(word) for word in source if normalize_word(word))
    return words


def _file_words(payload: dict) -> list[str]:
    """Нормализованная последовательность слов из предложений файла."""
    words = []
    for sentence in payload["sentences"]:
        words.extend(norm for _, norm in tokenize(sentence["text"]))
    return words


EXPECTED_KEYS = [
    "schema",
    "source",
    "source_duration",
    "transcript",
    "model",
    "windows",
    "paragraphs",
    "sentences",
    "text",
    "stats",
]


def _validate_payload(payload: dict, transcript: dict, path: str) -> None:
    """Общие для всех прогонов обещания файла punctuated/1.

    Проверяется и структура (ровно те поля, что в схеме), и обещания про
    таймкоды: не убывают, лежат внутри [0, source_duration] и внутри границ
    своего сегмента. Плюс согласованность: абзацы ссылаются на существующие
    предложения, stats совпадает с содержимым, text собирается из абзацев.
    """
    _check(
        list(payload.keys()) == EXPECTED_KEYS,
        "поля файла не те и/или не в том порядке: %r" % list(payload.keys()),
    )
    _check(payload["schema"] == SCHEMA, "не та схема: %r" % payload["schema"])
    _check(payload["source"] == transcript["source"], "source не из транскрипта")
    _check(payload["transcript"] == path, "в поле transcript не путь ко входу")
    _check(payload["model"] == transcript["model"], "model не из транскрипта")
    duration = transcript["source_duration"]
    _check(payload["source_duration"] == duration, "source_duration не из транскрипта")

    # Таймкоды: округление, монотонность, границы записи и своего сегмента.
    bounds = {
        segment["index"]: (segment["start"], segment["end"])
        for segment in transcript["segments"]
    }
    windows = {window["index"]: window for window in payload["windows"]}
    previous = -1.0
    for sentence in payload["sentences"]:
        where = "предложение %d" % sentence["index"]
        for key in ("start", "end"):
            value = sentence[key]
            _check(isinstance(value, (int, float)), "%s: %s не число" % (where, key))
            _check(
                round(float(value), ROUND_TO) == value,
                "%s: %s округлён не до %d знаков: %r" % (where, key, ROUND_TO, value),
            )
        _check(sentence["start"] <= sentence["end"], "%s: end раньше start" % where)
        _check(sentence["start"] >= previous, "%s: таймкод пошёл назад" % where)
        previous = sentence["start"]
        _check(0.0 <= sentence["start"] <= duration, "%s: start вне записи" % where)
        _check(0.0 <= sentence["end"] <= duration, "%s: end вне записи" % where)

        window = windows[sentence["window"]]
        first, last = window["segments"]
        low = min(bounds[number][0] for number in range(first, last + 1))
        high = max(bounds[number][1] for number in range(first, last + 1))
        _check(
            low <= sentence["start"] <= high and low <= sentence["end"] <= high,
            "%s: таймкод вышел за границы сегментов своего окна" % where,
        )
        _check(has_content(sentence["text"]), "%s: текст без букв и цифр" % where)

    # Абзацы: непрерывные диапазоны, покрывающие все предложения по одному разу.
    covered = []
    for paragraph in payload["paragraphs"]:
        first, last = paragraph["sentences"]
        _check(first <= last, "абзац %d: границы перевёрнуты" % paragraph["index"])
        _check(
            paragraph["window"] in windows,
            "абзац %d: ссылка на окно, которого нет" % paragraph["index"],
        )
        covered.extend(range(first, last + 1))
    _check(
        covered == list(range(len(payload["sentences"]))),
        "абзацы не покрывают предложения подряд и по одному разу: %r" % covered[:20],
    )

    # Текст: абзацы через пустую строку, слова те же, что в предложениях.
    blocks = []
    for paragraph in payload["paragraphs"]:
        first, last = paragraph["sentences"]
        blocks.append(
            " ".join(payload["sentences"][number]["text"] for number in range(first, last + 1))
        )
    _check(payload["text"] == "\n\n".join(blocks), "поле text собрано не из абзацев")

    stats = payload["stats"]
    _check(
        list(stats.keys())
        == ["windows", "fallback_windows", "paragraphs", "sentences", "words"],
        "поля stats не те: %r" % list(stats.keys()),
    )
    _check(stats["windows"] == len(payload["windows"]), "stats.windows не то число")
    _check(
        stats["fallback_windows"]
        == len([item for item in payload["windows"] if item["status"] == "fallback"]),
        "stats.fallback_windows не то число",
    )
    _check(stats["paragraphs"] == len(payload["paragraphs"]), "stats.paragraphs не то число")
    _check(stats["sentences"] == len(payload["sentences"]), "stats.sentences не то число")
    _check(stats["words"] == len(payload["text"].split()), "stats.words не то число")

    for window in payload["windows"]:
        _check(
            window["status"] in ("ok", "fallback"),
            "окно %d: неизвестный status %r" % (window["index"], window["status"]),
        )
        _check(isinstance(window["reason"], str), "окно %d: reason не строка" % window["index"])
        if window["status"] == "fallback":
            _check(window["reason"], "окно %d в откате без причины" % window["index"])
        for sentence in payload["sentences"]:
            if sentence["window"] == window["index"]:
                _check(
                    sentence["fallback"] == (window["status"] == "fallback"),
                    "предложение %d: флаг fallback не совпал с окном" % sentence["index"],
                )


def _test_successful_run() -> None:
    """Все окна прошли сверку: пунктуация, абзацы, предложения, таймкоды, код 0."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        stub_dir = _make_stub(root, ["ок"])
        transcript = _fixture_transcript()
        path = _write_json(root / "transcript.json", transcript)
        out = str(root / "итог прогона" / "punctuated.json")
        logs = str(root / "логи ollama")

        with _env(PATH="%s%s%s" % (stub_dir, os.pathsep, os.environ["PATH"]), OLLAMA_BIN=None):
            code, report, stderr = _run(
                ["--transcript", path, "--out", out, "--window", "3", "--log-dir", logs]
            )
        _check(code == 0, "удачный прогон вернул код %d (stderr: %s)" % (code, stderr.strip()))
        _check(os.path.isfile(out), "файл не записан: %s" % out)

        payload = _read_json(out)
        _validate_payload(payload, transcript, path)

        # Окна: по порядку, без перекрытия, по 3 сегмента; 7 сегментов -> 3 окна.
        got = [tuple(window["segments"]) for window in payload["windows"]]
        _check(got == [(0, 2), (3, 5), (6, 6)], "окна порезаны не так: %r" % got)
        _check(
            all(window["status"] == "ok" for window in payload["windows"]),
            "в удачном прогоне есть окна в откате",
        )
        _check(
            all(window["reason"] == "" for window in payload["windows"]),
            "в удачном прогоне у окна есть причина отката",
        )
        _check(
            [window["index"] for window in payload["windows"]] == [0, 1, 2],
            "окна пронумерованы не по порядку",
        )

        # Слова не изменились ни одним словом, порядок склейки — по окнам.
        _check(
            _file_words(payload) == _transcript_words(transcript),
            "последовательность слов в файле не совпала с транскриптом",
        )
        _check(
            all(not sentence["fallback"] for sentence in payload["sentences"]),
            "в удачном прогоне предложения помечены откатом",
        )

        # Пунктуация и заглавные на месте, предложений больше одного на окно.
        first = payload["sentences"][0]
        _check(first["text"][0].isupper(), "первое предложение не с заглавной: %r" % first["text"])
        _check(first["text"].endswith("."), "первое предложение без точки: %r" % first["text"])
        _check(len(payload["sentences"]) >= 9, "предложений всего %d" % len(payload["sentences"]))
        _check(payload["paragraphs"] and len(payload["paragraphs"]) >= 4,
               "абзацев всего %d" % len(payload["paragraphs"]))
        # В одном окне больше одного абзаца — значит пустая строка в ответе
        # ollama действительно превратилась в абзац.
        by_window: dict[int, int] = {}
        for paragraph in payload["paragraphs"]:
            by_window[paragraph["window"]] = by_window.get(paragraph["window"], 0) + 1
        _check(max(by_window.values()) >= 2, "разбиения на абзацы внутри окна нет: %r" % by_window)

        # Таймкоды от начала записи: первое предложение в первой секунде,
        # последнее — в последнем сегменте.
        _check(first["start"] == 0.0, "первое предложение начинается с %r" % first["start"])
        last = payload["sentences"][-1]
        _check(60.0 <= last["start"] <= 69.0, "последнее предложение в %r" % last["start"])

        # Сегмент без пословных таймкодов (index 5): предложение, целиком
        # собранное из его слов, получает границы сегмента.
        inside = [
            sentence
            for sentence in payload["sentences"]
            if sentence["start"] == 50.0 and sentence["end"] == 59.0
        ]
        _check(inside, "у сегмента без пословных таймкодов не взяты его границы")

        # Отчёт в stdout — по-русски и со всеми числами.
        for needle in ("Окон:          3", "В откате:      0", "Предложений:", "Слов:", "Абзацев:"):
            _check(needle in report, "в отчёте нет строки %r" % needle)
        _check(out in report, "в отчёте нет пути к файлу")

        # Промпт: правила, запреты и контекст предыдущего окна с пометкой.
        prompts = _stub_prompts(stub_dir)
        _check(len(prompts) == 3, "вызовов ollama было %d, а окон 3" % len(prompts))
        for needle in (
            "менять слова",
            "добавлять слова",
            "удалять",
            "цифры",
            "оговорки",
            "переставлять слова",
            "комментарии",
            "знаки пунктуации",
            "регистр букв",
            "абзацы пустой строкой",
            "только обработанный текст",
            TEXT_MARKER,
        ):
            _check(needle in prompts[0], "в промпте нет запрета или разрешения: %r" % needle)
        _check("начало записи" in prompts[0], "в первом промпте нет пометки о начале записи")
        for number in (1, 2):
            _check(
                "не выводить его в ответе" in prompts[number],
                "в промпте окна %d нет пометки «не выводить»" % number,
            )
        # Контекстом идёт именно последнее предложение предыдущего окна.
        previous_window = [
            sentence for sentence in payload["sentences"] if sentence["window"] == 0
        ]
        _check(
            previous_window[-1]["text"] in prompts[1],
            "в промпте второго окна нет последнего предложения первого",
        )
        # А текст окна в промпте — дословный, без пунктуации и заглавных.
        window_text = prompts[1].split(TEXT_MARKER)[-1].strip()
        _check(
            window_text == " ".join(
                word for segment in transcript["segments"][3:6]
                for word in segment["text"].split()
            ),
            "в промпте второго окна не дословный текст его сегментов",
        )

        # --log-dir: промпты и ответы сложены через ollama_client.
        names = sorted(os.listdir(logs))
        _check(len(names) == 6, "в каталоге логов %d файлов, а вызовов 3: %r" % (len(names), names))
        _check(
            any(name.startswith("окно000") for name in names),
            "логи не названы по окнам: %r" % names,
        )


def _test_changed_word() -> None:
    """Окно с изменённым словом откатывается, остальные обрабатываются как обычно."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        # Второе окно из трёх портит слово, остальные отвечают честно.
        stub_dir = _make_stub(root, ["ок", "изменить слово", "ок"])
        transcript = _fixture_transcript()
        path = _write_json(root / "transcript.json", transcript)
        out = str(root / "punctuated.json")

        with _env(PATH="%s%s%s" % (stub_dir, os.pathsep, os.environ["PATH"]), OLLAMA_BIN=None):
            code, report, stderr = _run(
                ["--transcript", path, "--out", out, "--window", "3"]
            )
        _check(code == 1, "откат окна вернул код %d, ожидался 1" % code)

        payload = _read_json(out)
        _validate_payload(payload, transcript, path)

        statuses = [window["status"] for window in payload["windows"]]
        _check(statuses == ["ok", "fallback", "ok"], "статусы окон: %r" % statuses)
        reason = payload["windows"][1]["reason"]
        _check("слова не совпали" in reason, "причина отката не про слова: %r" % reason)
        _check("выдумка" in reason, "в причине нет подставленного слова: %r" % reason)
        _check("позиция" in reason, "в причине нет позиции расхождения: %r" % reason)

        # В stderr — номер окна, позиция и пара «было → стало».
        _check("окно 1" in stderr, "в stderr нет номера окна: %s" % stderr)
        _check("позиции 3" in stderr, "в stderr не та позиция расхождения: %s" % stderr)
        _check("→ «выдумка»" in stderr, "в stderr нет пары «было → стало»: %s" % stderr)

        # Откатившееся окно: по предложению на сегмент, дословно и без точек.
        rolled = [
            sentence for sentence in payload["sentences"] if sentence["window"] == 1
        ]
        _check(len(rolled) == 3, "в откатившемся окне %d предложений, а сегментов 3" % len(rolled))
        for sentence, segment in zip(rolled, transcript["segments"][3:6]):
            _check(sentence["text"] == segment["text"], "текст отката не дословный")
            _check(sentence["fallback"] is True, "предложение отката без флага fallback")
            _check(sentence["start"] == segment["start"], "таймкод отката не от сегмента")
            _check(sentence["end"] == segment["end"], "конец отката не от сегмента")
        # Выдуманного слова в файле нет вовсе.
        _check("выдумка" not in payload["text"], "подставленное слово попало в файл")
        # Соседние окна пунктуацию получили.
        clean = [sentence for sentence in payload["sentences"] if sentence["window"] == 0]
        _check(clean[0]["text"].endswith("."), "чистое окно осталось без пунктуации")
        _check(
            all(not sentence["fallback"] for sentence in clean),
            "чистое окно помечено откатом",
        )
        # Слова всё равно совпадают с транскриптом слово в слово.
        _check(
            _file_words(payload) == _transcript_words(transcript),
            "после отката последовательность слов разъехалась",
        )
        _check("В откате:      1" in report, "в отчёте не одно окно в откате")
        _check("окно 1, сегменты 3–5" in report, "в отчёте нет диапазона сегментов отката")


def _test_invented_words() -> None:
    """Выдуманные добавленные слова и приписка от себя тоже откатывают окно."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        transcript = _fixture_transcript(segments=2, without_words=())
        path = _write_json(root / "transcript.json", transcript)

        cases = [
            ("добавить слова", "лишние слова в конце"),
            ("приписка", "вступление от ollama"),
            ("потерять слово", "пропавшее слово"),
        ]
        for mode, what in cases:
            stub_dir = _make_stub(root / mode, [mode])
            out = str(root / ("punctuated-%s.json" % mode))
            with _env(
                PATH="%s%s%s" % (stub_dir, os.pathsep, os.environ["PATH"]), OLLAMA_BIN=None
            ):
                code, _, stderr = _run(["--transcript", path, "--out", out, "--window", "2"])
            _check(code == 1, "случай %r вернул код %d, ожидался 1" % (what, code))

            payload = _read_json(out)
            _validate_payload(payload, transcript, path)
            _check(
                payload["windows"][0]["status"] == "fallback",
                "случай %r: окно не откатилось" % what,
            )
            _check(
                "слова не совпали" in payload["windows"][0]["reason"],
                "случай %r: причина не про слова: %r" % (what, payload["windows"][0]["reason"]),
            )
            _check("позиции" in stderr, "случай %r: в stderr нет позиции" % what)
            _check(
                _file_words(payload) == _transcript_words(transcript),
                "случай %r: в файле не дословная последовательность слов" % what,
            )
            # Ни одного выдуманного слова в файле.
            for invented in ("Готово", "пара", "меня"):
                _check(
                    invented not in payload["text"],
                    "случай %r: выдуманное слово %r попало в файл" % (what, invented),
                )


def _test_ollama_failure() -> None:
    """Ненулевой код ollama по окну — тоже откат, а не падение инструмента."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        stub_dir = _make_stub(root, ["ненулевой код", "ок"])
        transcript = _fixture_transcript(segments=4, without_words=())
        path = _write_json(root / "transcript.json", transcript)
        out = str(root / "punctuated.json")

        with _env(PATH="%s%s%s" % (stub_dir, os.pathsep, os.environ["PATH"]), OLLAMA_BIN=None):
            # --attempts 1: повторять заведомо падающую заглушку незачем, а пауза
            # между попытками живёт в ollama_client и съела бы время селф-теста.
            code, report, stderr = _run(
                ["--transcript", path, "--out", out, "--window", "2", "--attempts", "1"]
            )
        _check(code == 1, "неответивший ollama дал код %d, ожидался 1" % code)

        payload = _read_json(out)
        _validate_payload(payload, transcript, path)
        statuses = [window["status"] for window in payload["windows"]]
        _check(statuses == ["fallback", "ok"], "статусы окон: %r" % statuses)
        _check(
            "ollama не ответил" in payload["windows"][0]["reason"],
            "причина отката не про неответивший ollama: %r" % payload["windows"][0]["reason"],
        )
        _check("код 9" in payload["windows"][0]["reason"], "в причине нет кода возврата ollama")
        _check("окно 0" in stderr, "в stderr нет номера неудачного окна: %s" % stderr)
        _check("В откате:      1" in report, "в отчёте не одно окно в откате")


def _test_no_ollama() -> None:
    """ollama нет вовсе: все окна в откате, файл записан и валиден, код 1."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        empty = root / "пустой-path"
        empty.mkdir()
        transcript = _fixture_transcript()
        path = _write_json(root / "transcript.json", transcript)
        out = str(root / "punctuated.json")

        with _env(PATH=str(empty), OLLAMA_BIN=None):
            code, report, stderr = _run(
                ["--transcript", path, "--out", out, "--window", "3"]
            )
        _check(code == 1, "без ollama код возврата %d, ожидался 1" % code)
        _check(os.path.isfile(out), "без ollama файл не записан")

        payload = _read_json(out)
        _validate_payload(payload, transcript, path)
        _check(
            all(window["status"] == "fallback" for window in payload["windows"]),
            "без ollama не все окна в откате: %r" % payload["windows"],
        )
        for window in payload["windows"]:
            _check("недоступен" in window["reason"], "причина не про недоступность ollama")
        _check(
            all(sentence["fallback"] for sentence in payload["sentences"]),
            "без ollama есть предложения без флага fallback",
        )
        # Каждый сегмент стал предложением со своим таймкодом, текст дословный.
        _check(
            len(payload["sentences"]) == len(transcript["segments"]),
            "предложений %d, а сегментов %d"
            % (len(payload["sentences"]), len(transcript["segments"])),
        )
        for sentence, segment in zip(payload["sentences"], transcript["segments"]):
            _check(sentence["text"] == segment["text"], "текст не дословный")
            _check(sentence["start"] == segment["start"], "таймкод не от сегмента")
        _check(
            _file_words(payload) == _transcript_words(transcript),
            "без ollama слова разъехались с транскриптом",
        )
        _check("В откате:      3" in report, "в отчёте не три окна в откате")
        _check("ollama недоступен" in stderr, "про недоступность ollama не предупредили")

        # Не авторизованный ollama — тот же исход: оставшиеся окна не вызываются.
        auth_dir = root / "неавторизованный"
        auth_dir.mkdir()
        stub = auth_dir / "ollama"
        stub.write_text(
            "#!/bin/sh\nprintf 'Invalid API key · Please run /login\\n' >&2\nexit 1\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        out2 = str(root / "punctuated-auth.json")
        with _env(PATH="%s%s%s" % (auth_dir, os.pathsep, os.environ["PATH"]), OLLAMA_BIN=None):
            code, _, stderr = _run(
                ["--transcript", path, "--out", out2, "--window", "3", "--attempts", "1"]
            )
        _check(code == 1, "неавторизованный ollama дал код %d" % code)
        payload = _read_json(out2)
        _validate_payload(payload, transcript, path)
        _check(
            all(window["status"] == "fallback" for window in payload["windows"]),
            "при неавторизованном ollama не все окна в откате",
        )
        _check("не авторизован" in stderr, "про неавторизованность не сказали: %s" % stderr)


def _test_sentence_splitting() -> None:
    """Разбиение на предложения и абзацы — на разметке, а не на угадывании."""
    cases = [
        ("Привет всем. Начинаем работу.", ["Привет всем.", "Начинаем работу."]),
        # Строчная буква после точки — не граница: «и т. д.» не разрывается.
        ("и так далее. потом ещё", ["и так далее. потом ещё"]),
        ("Что это?! Не знаю…", ["Что это?!", "Не знаю…"]),
        ("Он сказал: «Да.» И ушёл.", ["Он сказал: «Да.»", "И ушёл."]),
        # Многоточие — такая же граница.
        ("Ну… Ладно.", ["Ну…", "Ладно."]),
        # Цифра после точки тоже начинает предложение.
        ("Было 25. 30 осталось.", ["Было 25.", "30 осталось."]),
        # Предложение без точки в конце абзаца всё равно предложение.
        ("Первое. И второе без точки", ["Первое.", "И второе без точки"]),
        # Переводы строк внутри абзаца сжимаются в пробел.
        ("Первое\nпредложение. Второе.", ["Первое предложение.", "Второе."]),
        # Кусок без букв и цифр отдельным предложением не становится.
        ("Текст. ...", ["Текст. ..."]),
        ("...", []),
        ("   ", []),
    ]
    for source, expected in cases:
        got = split_sentences(source)
        _check(got == expected, "разбиение %r дало %r, ожидалось %r" % (source, got, expected))

    # Абзацы: пустая строка — граница, лишние пустые строки не создают пустых
    # абзацев, и ни одно слово при разбиении не теряется.
    answer = "Первый абзац. Ещё фраза.\n\n\nВторой абзац.\n \nТретий абзац."
    paragraphs = split_paragraphs(answer)
    _check(len(paragraphs) == 3, "абзацев %d, ожидалось 3: %r" % (len(paragraphs), paragraphs))
    _check(paragraphs[0] == ["Первый абзац.", "Ещё фраза."], "первый абзац: %r" % paragraphs[0])
    words_in = [norm for _, norm in tokenize(answer)]
    words_out = [
        norm
        for block in paragraphs
        for sentence in block
        for _, norm in tokenize(sentence)
    ]
    _check(words_in == words_out, "при разбиении потерялись слова: %r" % words_out)


def _test_word_check() -> None:
    """Нормализация и поиск первого расхождения — отдельно от всего прогона."""
    _check(normalize_word("Привет!") == "привет", "знаки не выброшены")
    _check(normalize_word("ещё") == normalize_word("еще"), "ё не приведена к е")
    _check(normalize_word("Ещё,") == "еще", "регистр и запятая не сняты")
    _check(normalize_word("25") == "25", "цифры выброшены")
    _check(normalize_word("как-то") == "както", "дефис не выброшен")
    _check(normalize_word("—") == "", "тире не стало пустым токеном")
    _check(normalize_word("«»") == "", "кавычки не стали пустым токеном")
    _check([raw for raw, _ in tokenize("а — б")] == ["а", "б"], "пустой токен не отброшен")

    reference = tokenize("двадцать пять процентов текста")
    _check(first_mismatch(reference, tokenize("Двадцать пять процентов текста.")) is None,
           "пунктуация и регистр приняты за расхождение")
    _check(
        first_mismatch(reference, tokenize("двадцать пять процентов")) == (4, "текста", "—"),
        "пропажа последнего слова не найдена: %r"
        % (first_mismatch(reference, tokenize("двадцать пять процентов")),),
    )
    _check(
        first_mismatch(reference, tokenize("двадцать пять процентов текста и ещё"))[0] == 5,
        "лишние слова в конце не найдены",
    )
    # Перевод числа в цифры — изменение слова, а не оформление.
    _check(
        first_mismatch(reference, tokenize("25 процентов текста")) == (1, "двадцать", "25"),
        "перевод числа в цифры не считается расхождением",
    )
    _check(
        first_mismatch(reference, tokenize("двадцать пять поцентов текста"))
        == (3, "процентов", "поцентов"),
        "исправленная опечатка не найдена",
    )
    # «Исправленная» оговорка на первой позиции.
    _check(first_mismatch(reference, tokenize("двадцати пять процентов текста"))[0] == 1,
           "изменённая форма слова не найдена")


def _test_window_edges() -> None:
    """Края нарезки: окно больше транскрипта, окно в один сегмент, пустые сегменты."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        stub_dir = _make_stub(root, ["ок"])
        transcript = _fixture_transcript(segments=3, without_words=())
        path = _write_json(root / "transcript.json", transcript)
        environment = {
            "PATH": "%s%s%s" % (stub_dir, os.pathsep, os.environ["PATH"]),
            "OLLAMA_BIN": None,
        }

        # Окно больше, чем сегментов: одно окно на всю запись.
        out = str(root / "одно-окно.json")
        with _env(**environment):
            code, _, _ = _run(["--transcript", path, "--out", out, "--window", "1000"])
        _check(code == 0, "окно больше транскрипта дало код %d" % code)
        payload = _read_json(out)
        _validate_payload(payload, transcript, path)
        _check(len(payload["windows"]) == 1, "окон %d вместо одного" % len(payload["windows"]))
        _check(payload["windows"][0]["segments"] == [0, 2], "окно не на всю запись")

        # Окно в один сегмент: окон столько, сколько сегментов.
        out = str(root / "по-сегменту.json")
        with _env(**environment):
            code, _, _ = _run(["--transcript", path, "--out", out, "--window", "1"])
        _check(code == 0, "окно в один сегмент дало код %d" % code)
        payload = _read_json(out)
        _validate_payload(payload, transcript, path)
        _check(len(payload["windows"]) == 3, "окон %d вместо трёх" % len(payload["windows"]))

        # Пустой сегмент (тишина): окно без слов ollama не беспокоит и откатом
        # не считается — терять в нём нечего.
        quiet = _fixture_transcript(segments=3, without_words=())
        quiet["segments"][1]["text"] = ""
        quiet["segments"][1]["words"] = []
        quiet_path = _write_json(root / "тишина.json", quiet)
        out = str(root / "тишина-итог.json")
        with _env(**environment):
            code, _, _ = _run(["--transcript", quiet_path, "--out", out, "--window", "1"])
        _check(code == 0, "пустой сегмент дал код %d, ожидался 0" % code)
        payload = _read_json(out)
        _validate_payload(payload, quiet, quiet_path)
        _check(
            payload["windows"][1]["status"] == "ok",
            "пустое окно объявлено откатом: %r" % payload["windows"][1],
        )
        _check("нет слов" in payload["windows"][1]["reason"], "про пустое окно не сказано")
        _check(
            all(sentence["window"] != 1 for sentence in payload["sentences"]),
            "у пустого окна появились предложения",
        )


def _test_reread_and_contract() -> None:
    """Файл перечитывается и принимается тем, кто его потребляет."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        stub_dir = _make_stub(root, ["ок", "изменить слово"])
        transcript = _fixture_transcript(segments=4, without_words=(3,))
        path = _write_json(root / "transcript.json", transcript)
        out = str(root / "punctuated.json")

        with _env(PATH="%s%s%s" % (stub_dir, os.pathsep, os.environ["PATH"]), OLLAMA_BIN=None):
            code, _, _ = _run(["--transcript", path, "--out", out, "--window", "2"])
        _check(code == 1, "прогон вернул код %d, ожидался 1" % code)

        with open(out, "r", encoding="utf-8") as handle:
            raw = handle.read()
        # Запись: отступ 2, кириллица не заэскейплена, перевод строки в конце.
        _check('\n  "schema"' in raw, "отступ в файле не 2")
        _check("\\u" not in raw, "кириллица записана эскейпами (ensure_ascii не выключен)")
        _check(raw.endswith("\n"), "файл не заканчивается переводом строки")
        payload = json.loads(raw)
        _validate_payload(payload, transcript, path)

        # Повторный прогон того же транскрипта даёт такой же валидный файл.
        # Пишется он рядом, а не поверх: подставной ollama отвечает по списку
        # режимов, и второй прогон — это уже другое содержимое, по которому ниже
        # нельзя было бы проверять контракт со смесью «ок» и «откат».
        again = str(root / "punctuated-повтор.json")
        with _env(PATH="%s%s%s" % (stub_dir, os.pathsep, os.environ["PATH"]), OLLAMA_BIN=None):
            code, _, _ = _run(["--transcript", path, "--out", again, "--window", "2"])
        _check(code in (0, 1), "повторный прогон вернул код %d" % code)
        _validate_payload(_read_json(again), transcript, path)

        # Контракт с потребителем: tools/render_doc.py разбирает файл без жалоб.
        # Если соседнего инструмента в дереве нет, проверку пропускаем — ломать
        # селф-тест пунктуации из-за отсутствия чужого файла незачем.
        try:
            import render_doc
        except ImportError:  # pragma: no cover — render_doc лежит рядом
            sys.stdout.write("         (render_doc.py не найден, контракт не проверен)\n")
            return
        with redirect_stderr(io.StringIO()):
            parsed = render_doc.parse_punctuated(_read_json(out), path=out)
        _check(
            len(parsed.sentences) == payload["stats"]["sentences"],
            "render_doc увидел другое число предложений",
        )
        _check(
            len(parsed.paragraphs) == payload["stats"]["paragraphs"],
            "render_doc увидел другое число абзацев",
        )
        _check(
            any(window.fallback for window in parsed.windows.values()),
            "render_doc не увидел окно в откате",
        )


def _test_broken_input() -> None:
    """Непригодный вход: код 2, сообщение по-русски, --out не создаётся."""
    with tempfile.TemporaryDirectory(prefix="punctuate-selftest-") as workdir:
        root = Path(workdir)
        good = _write_json(root / "transcript.json", _fixture_transcript(segments=2))
        broken = root / "битый.json"
        broken.write_text('{"schema": "transcript/1", "segments": [', encoding="utf-8")
        wrong_schema = _write_json(
            root / "чужая.json", dict(_fixture_transcript(segments=1), schema="punctuated/1")
        )
        no_segments = _write_json(
            root / "без-сегментов.json", dict(_fixture_transcript(segments=1), segments=[])
        )
        not_object = _write_json(root / "список.json", [1, 2, 3])
        out = str(root / "не должен появиться" / "punctuated.json")

        cases = [
            (["--transcript", str(root / "нет.json"), "--out", out], "нет файла"),
            (["--transcript", str(root), "--out", out], "каталог вместо файла"),
            (["--transcript", str(broken), "--out", out], "битый JSON"),
            (["--transcript", wrong_schema, "--out", out], "не та схема"),
            (["--transcript", no_segments, "--out", out], "нет сегментов"),
            (["--transcript", not_object, "--out", out], "не объект JSON"),
            (["--out", out], "без --transcript"),
            (["--transcript", good], "без --out"),
            (["--transcript", good, "--out", good], "--out поверх транскрипта"),
            (["--transcript", good, "--out", out, "--window", "0"], "--window 0"),
            (["--transcript", good, "--out", out, "--window", "-1"], "--window -1"),
            (["--transcript", good, "--out", out, "--window", "много"], "--window не число"),
            (["--transcript", good, "--out", out, "--attempts", "0"], "--attempts 0"),
            (["--transcript", good, "--out", out, "--timeout", "0"], "--timeout 0"),
            (["--transcript", good, "--out", out, "--timeout", "ноль"], "--timeout не число"),
            (["--transcript", good, "--out"], "--out без значения"),
            (["--transcript", good, "--out", out, "--log-dir"], "--log-dir без значения"),
            (["--кривая-опция"], "неизвестная опция"),
            (["лишнее"], "лишний аргумент"),
            (["--selftest=1"], "--selftest со значением"),
        ]
        for argv, what in cases:
            code, _, stderr = _run(argv)
            _check(code == 2, "случай %r вернул код %d, ожидался 2" % (what, code))
            _check(stderr.startswith(PROG + ":"), "случай %r ничего не сказал в stderr" % what)
            _check(not os.path.exists(out), "случай %r создал файл --out" % what)

        # Существующий файл при непригодном входе не перетирается.
        kept = root / "итог.json"
        kept.write_text("старое содержимое\n", encoding="utf-8")
        code, _, _ = _run(["--transcript", str(broken), "--out", str(kept)])
        _check(code == 2, "перетирание: код %d вместо 2" % code)
        _check(
            kept.read_text(encoding="utf-8") == "старое содержимое\n",
            "старый файл перетёрт при непригодном входе",
        )


def _test_help() -> None:
    """Справка — на русском, со всеми флагами, код 0."""
    for flag in ("-h", "--help"):
        code, out, _ = _run([flag])
        _check(code == 0, "%s вернул код %d" % (flag, code))
        for needle in (
            "--transcript",
            "--out",
            "--window",
            "--timeout",
            "--attempts",
            "--log-dir",
            "--selftest",
            "-h, --help",
            "Использование:",
            "Коды возврата:",
        ):
            _check(needle in out, "в справке (%s) нет %r" % (flag, needle))
        _check(re.search(r"[а-яА-Я]", out) is not None, "справка не на русском")
        _check(str(DEFAULT_WINDOW) in out, "в справке нет значения --window по умолчанию")


def selftest() -> int:
    """Прогоняет встроенные проверки. Код возврата 0 — все прошли."""
    tests = (
        ("удачный прогон: окна, абзацы, предложения, таймкоды", _test_successful_run),
        ("окно с изменённым словом откатывается", _test_changed_word),
        ("выдуманные, пропавшие слова и приписка откатывают окно", _test_invented_words),
        ("ollama не ответил по окну: откат, а не падение", _test_ollama_failure),
        ("ollama недоступен: все окна в откате, код 1", _test_no_ollama),
        ("разбиение на предложения и абзацы", _test_sentence_splitting),
        ("нормализация слов и поиск расхождения", _test_word_check),
        ("края нарезки на окна и пустые сегменты", _test_window_edges),
        ("запись, перечитывание и контракт с render_doc", _test_reread_and_contract),
        ("непригодный вход и коды возврата", _test_broken_input),
        ("справка", _test_help),
    )
    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ punctuate.py ================\n")
    for name, test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            sys.stdout.write("[ПРОВАЛ] %s: %s\n" % (name, exc))
        except Exception as exc:  # noqa: BLE001 — любое исключение здесь это провал теста
            failed += 1
            sys.stdout.write("[ОШИБКА] %s: %s: %s\n" % (name, type(exc).__name__, exc))
        else:
            sys.stdout.write("[ок]     %s\n" % name)

    passed = len(tests) - failed
    sys.stdout.write("\nПройдено %d из %d\n" % (passed, len(tests)))
    if failed:
        sys.stderr.write("%s: селф-тест не прошёл (%d проверок упало)\n" % (PROG, failed))
        return 1
    return 0


# -------------------------------------------------------------------- main ---


def main(argv: list[str]) -> int:
    try:
        opts = parse_args(argv)
    except UsageError as exc:
        return die(str(exc))

    if opts["help"]:
        sys.stdout.write(HELP)
        return 0
    if opts["selftest"]:
        return selftest()

    try:
        check_args(opts)
        transcript, source, model = read_transcript(opts["transcript"])
    except (UsageError, InputError) as exc:
        return die(str(exc))

    windows = build_windows(transcript.segments, opts["window"])
    try:
        results = punctuate(
            windows,
            timeout=opts["timeout"],
            attempts=opts["attempts"],
            log_dir=opts["log_dir"],
            model=opts["ollama_model"],
        )
    except KeyboardInterrupt:
        return die("прервано пользователем: файл не записан", code=1)

    payload, report = build_payload(
        results,
        source=source,
        source_duration=transcript.source_duration,
        transcript_path=opts["transcript"],
        model=model,
    )
    try:
        write_payload(opts["out"], payload)
    except InputError as exc:
        return die(str(exc))

    print_report(opts["out"], report)

    # Код 1 — файл записан и валиден, но пунктуация в нём не полная: сборочный
    # скрипт обязан это заметить и сказать человеку.
    return 1 if report.fallback_windows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
