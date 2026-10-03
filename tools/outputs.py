#!/usr/bin/env python3
"""outputs.py — рендер читаемого текста (.txt) и субтитров (.srt) из транскрипта.

На вход — JSON схемы transcript/1: сегменты (один сегмент = один чанк нарезки)
с пословными таймкодами. На выход — два файла, которые просил владелец:

  * .txt — по абзацу на сегмент. Никаких таймкодов и номеров: это текст для
    чтения. Текст сегментов не переписывается — у модели v3_rnnt он идёт без
    заглавных и пунктуации, и приводить его «к виду» здесь нечем, а абзацы по
    чанкам — единственная структура, которая у такого текста вообще есть.
  * .srt — субтитры, собранные ПО СЛОВАМ, а не по сегментам. Сегмент длиной до
    24 секунд — это десятки слов; одним кадром такое читать невозможно, поэтому
    кадр закрывается по длине строк, по длительности или по паузе в речи.

Почему таймкоды форматируются своей функцией, а не gigaam.format_time(): та
отдаёт HH:MM:SS:mm (центисекунды через двоеточие) — это не формат SRT.

Внутри всё считается в целых миллисекундах. SRT всё равно хранит миллисекунды,
а целые числа избавляют от двойного дребезга: сравнение длительностей не зависит
от того, что 0.3 * 20 в double равно 6.000000000000001, и кадр не может после
округления получить end, равный start.

Зависимостей нет: только стандартная библиотека. Ни torch, ни gigaam здесь не
нужны — инструмент не распознаёт речь, а только рендерит готовый транскрипт.
Проверки живут в самом файле (--selftest): в окружении проекта нет pytest.

Использование:
  python tools/outputs.py --transcript <файл.json> [--txt <файл>] [--srt <файл>]
  python tools/outputs.py --selftest
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

PROG = "outputs.py"
SCHEMA = "transcript/1"

# Значения по умолчанию — обычная практика субтитров: две строки по 42 символа,
# кадр не длиннее шести секунд.
DEFAULT_MAX_LINE = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MAX_CUE = 6.0
DEFAULT_GAP = 0.4

HELP = """Использование:
  python tools/outputs.py --transcript <файл.json> [--txt <файл>] [--srt <файл>] [опции]
  python tools/outputs.py --selftest

Читает JSON транскрипта схемы transcript/1 (сегменты с пословными таймкодами) и
пишет читаемый текст (.txt) и субтитры (.srt). Кадры субтитров собираются по
пословным таймкодам, а не по сегментам: сегмент — это чанк нарезки до 24 секунд.

Опции:
  --transcript <файл>   JSON транскрипта схемы transcript/1
  --txt <файл>          куда записать читаемый текст (по умолчанию не пишется)
  --srt <файл>          куда записать субтитры (по умолчанию не пишется)
  --max-line <N>        максимум символов в строке субтитра (по умолчанию 42)
  --max-lines <N>       максимум строк в кадре (по умолчанию 2)
  --max-cue <сек>       максимальная длительность кадра (по умолчанию 6.0)
  --gap <сек>           пауза перед словом, с которой начинается новый кадр
                        (по умолчанию 0.4)
  --selftest            прогнать встроенные проверки и выйти
  -h, --help            показать эту справку

Нужен хотя бы один из --txt и --srt; задавать их можно независимо.
Каталоги для файлов вывода создаются при необходимости.

Примеры:
  python tools/outputs.py --transcript data/out/lecture.json \\
      --txt data/out/lecture.txt --srt data/out/lecture.srt
  python tools/outputs.py --transcript data/out/lecture.json --srt subs.srt --max-line 38
"""


class UsageError(Exception):
    """Ошибка в аргументах командной строки."""


class TranscriptError(Exception):
    """Транскрипт не читается, не той схемы или в нём нет нужных полей."""


def die(message: str) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main()."""
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return 1


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но пользователь должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


# ----------------------------------------------------------------- таймкоды ---


def to_ms(seconds: float) -> int:
    """Секунды -> целые миллисекунды (с округлением до ближайшей)."""
    return int(round(float(seconds) * 1000.0))


def format_ms(milliseconds: int) -> str:
    """Миллисекунды -> HH:MM:SS,mmm — формат таймкода SRT.

    Запятая вместо точки и ровно три знака миллисекунд — требование формата;
    плееры, читающие SRT построчно, на точку реагируют по-разному.
    """
    if milliseconds < 0:
        raise ValueError("таймкод не может быть отрицательным: %r" % (milliseconds,))
    hours, rest = divmod(int(milliseconds), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, secs, ms)


def format_timestamp(seconds: float) -> str:
    """Секунды -> HH:MM:SS,mmm. Проверяется селф-тестом на 0.0, 59.999, 3661.5."""
    return format_ms(to_ms(seconds))


# ------------------------------------------------------------ чтение входа ---


@dataclass
class Word:
    """Слово с пословным таймкодом; границы — целые миллисекунды."""

    text: str
    start: int
    end: int


@dataclass
class Segment:
    """Сегмент транскрипта — он же чанк нарезки."""

    index: int
    start: int
    end: int
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """То, что инструменту нужно от транскрипта: сегменты и длительность записи.

    Остальные поля схемы (model, device, plan, metrics, ...) здесь не нужны:
    рендер от них не зависит, и привязываться к содержимому metrics нельзя —
    его пишет соседний инструмент и состав ключей там свой.
    """

    source: str
    source_duration: float | None
    segments: list[Segment]


def _as_number(holder: dict, key: str, where: str) -> float:
    """Достаёт из словаря обязательное число, иначе — понятная ошибка."""
    if key not in holder:
        raise TranscriptError("%s: нет обязательного поля %s" % (where, key))
    value = holder[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptError(
            "%s: поле %s должно быть числом секунд, получено: %s"
            % (where, key, _typename(value))
        )
    if not math.isfinite(float(value)):
        raise TranscriptError("%s: поле %s не является конечным числом" % (where, key))
    return float(value)


def _as_text(holder: dict, key: str, where: str) -> str:
    """Достаёт из словаря обязательную строку, иначе — понятная ошибка."""
    if key not in holder:
        raise TranscriptError("%s: нет обязательного поля %s" % (where, key))
    value = holder[key]
    if not isinstance(value, str):
        raise TranscriptError(
            "%s: поле %s должно быть строкой, получено: %s"
            % (where, key, _typename(value))
        )
    return value


def _typename(value: object) -> str:
    """Имя типа по-русски — для сообщений об ошибках."""
    names = {
        type(None): "null",
        bool: "логическое значение",
        int: "число",
        float: "число",
        str: "строка",
        list: "список",
        dict: "объект",
    }
    return names.get(type(value), type(value).__name__)


def parse_transcript(data: object, source: str = "<память>") -> Transcript:
    """Проверяет структуру JSON транскрипта и приводит её к Transcript.

    Проверяется только то, без чего рендер невозможен: схема, список сегментов,
    их границы и текст, структура пословных таймкодов. Несогласованность самих
    чисел (слово выходит за границы сегмента, кадры наезжают друг на друга) здесь
    ошибкой не считается — её лечит normalize_cues(), а оценка годности
    таймкодов — дело отдельного инструмента tools/check_timecodes.py.
    """
    if not isinstance(data, dict):
        raise TranscriptError(
            "на верхнем уровне транскрипта ожидается объект JSON, получено: %s"
            % _typename(data)
        )

    if "schema" not in data:
        raise TranscriptError("в транскрипте нет поля schema (ожидается %r)" % SCHEMA)
    if data["schema"] != SCHEMA:
        raise TranscriptError(
            "не та схема транскрипта: ожидается %r, в файле %r" % (SCHEMA, data["schema"])
        )

    if "segments" not in data:
        raise TranscriptError("в транскрипте нет поля segments")
    raw_segments = data["segments"]
    if not isinstance(raw_segments, list):
        raise TranscriptError(
            "поле segments должно быть списком, получено: %s" % _typename(raw_segments)
        )
    if not raw_segments:
        raise TranscriptError("список segments пуст — рендерить нечего")

    # source_duration нужен, чтобы ни один кадр не вышел за конец записи. Его
    # отсутствие не фатально (рендер возможен и без него), но молчать об этом
    # нельзя: проверка «end не превышает длительность» в таком прогоне не
    # выполняется.
    duration: float | None = None
    if data.get("source_duration") is None:
        warn("в транскрипте нет source_duration — кадры не обрезаются по длине записи")
    else:
        duration = _as_number(data, "source_duration", "транскрипт")
        if duration <= 0:
            raise TranscriptError(
                "source_duration должен быть больше нуля, получено: %s" % duration
            )

    segments: list[Segment] = []
    for position, raw in enumerate(raw_segments):
        where = "сегмент №%d" % (position + 1)
        if not isinstance(raw, dict):
            raise TranscriptError(
                "%s должен быть объектом JSON, получено: %s" % (where, _typename(raw))
            )
        index = raw.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            where = "сегмент index=%d" % index
        else:
            index = position

        start = _as_number(raw, "start", where)
        end = _as_number(raw, "end", where)
        text = _as_text(raw, "text", where)

        raw_words = raw.get("words")
        if raw_words is None:
            raw_words = []
        if not isinstance(raw_words, list):
            raise TranscriptError(
                "%s: поле words должно быть списком, получено: %s"
                % (where, _typename(raw_words))
            )

        words: list[Word] = []
        for word_position, raw_word in enumerate(raw_words):
            word_where = "%s, слово №%d" % (where, word_position + 1)
            if not isinstance(raw_word, dict):
                raise TranscriptError(
                    "%s должно быть объектом JSON, получено: %s"
                    % (word_where, _typename(raw_word))
                )
            word_text = _as_text(raw_word, "text", word_where).strip()
            word_start = _as_number(raw_word, "start", word_where)
            word_end = _as_number(raw_word, "end", word_where)
            # Слово без текста кадру ничего не даёт, а пустой кадр в SRT
            # недопустим — такие слова просто не участвуют в сборке.
            if not word_text:
                continue
            words.append(Word(word_text, to_ms(word_start), to_ms(word_end)))

        segments.append(Segment(index, to_ms(start), to_ms(end), text, words))

    return Transcript(source=source, source_duration=duration, segments=segments)


def load_transcript(path: str) -> Transcript:
    """Читает файл транскрипта. Любая беда на входе — TranscriptError, не трейсбек."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise TranscriptError("файл транскрипта не найден: %s" % path) from None
    except IsADirectoryError:
        raise TranscriptError("вместо файла транскрипта указан каталог: %s" % path) from None
    except PermissionError:
        raise TranscriptError("файл транскрипта недоступен для чтения: %s" % path) from None
    except UnicodeDecodeError:
        raise TranscriptError("файл транскрипта не в кодировке UTF-8: %s" % path) from None
    except json.JSONDecodeError as exc:
        raise TranscriptError(
            "файл транскрипта не разбирается как JSON: %s (строка %d, символ %d: %s)"
            % (path, exc.lineno, exc.colno, exc.msg)
        ) from None
    except OSError as exc:
        raise TranscriptError(
            "не удалось прочитать файл транскрипта: %s (%s)" % (path, exc.strerror or exc)
        ) from None
    return parse_transcript(data, source=path)


# -------------------------------------------------------------- сборка SRT ---


@dataclass
class Cue:
    """Кадр субтитра: границы в миллисекундах и уже свёрстанные строки."""

    start: int
    end: int
    lines: list[str]
    words: int  # сколько слов вошло; 0 — кадр собран из текста сегмента


def wrap_words(words: list[str], max_line: int) -> list[str]:
    """Жадно укладывает слова в строки не длиннее max_line символов.

    Слова не переносятся по слогам: слово длиннее max_line занимает всю строку
    целиком. В русской речи такие слова единичны, а разрыв слова читается хуже,
    чем одна длинная строка.
    """
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_line:
            current = current + " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_cues(
    segments: list[Segment],
    *,
    max_line: int = DEFAULT_MAX_LINE,
    max_lines: int = DEFAULT_MAX_LINES,
    max_cue: float = DEFAULT_MAX_CUE,
    gap: float = DEFAULT_GAP,
) -> list[Cue]:
    """Собирает кадры субтитров из пословных таймкодов.

    Кадр закрывается перед словом, если добавление этого слова:
      * не уложилось бы в max_lines строк по max_line символов, либо
      * сделало бы кадр длиннее max_cue, либо
      * отделено от предыдущего слова паузой не меньше gap.
    Границы кадра — start первого слова и end последнего: таймкоды берутся из
    транскрипта, а не вычисляются заново.
    """
    max_cue_ms = to_ms(max_cue)
    gap_ms = to_ms(gap)

    cues: list[Cue] = []
    for segment in segments:
        if not segment.words:
            cues.extend(_cue_without_words(segment, max_line))
            continue

        current: list[Word] = []
        for word in segment.words:
            if current:
                pause = word.start - current[-1].end
                too_long = word.end - current[0].start > max_cue_ms
                fits = (
                    len(wrap_words([w.text for w in current] + [word.text], max_line))
                    <= max_lines
                )
                if pause >= gap_ms or too_long or not fits:
                    cues.append(_close_cue(current, max_line))
                    current = []
            current.append(word)
        if current:
            cues.append(_close_cue(current, max_line))

    return cues


def _close_cue(words: list[Word], max_line: int) -> Cue:
    """Закрывает накопленный кадр по границам первого и последнего слова."""
    return Cue(
        start=words[0].start,
        end=words[-1].end,
        lines=wrap_words([w.text for w in words], max_line),
        words=len(words),
    )


def _cue_without_words(segment: Segment, max_line: int) -> list[Cue]:
    """Страховочная ветка: у сегмента нет пословных таймкодов.

    GigaAM отдаёт слова всегда, но транскрипт мог прийти от другого движка —
    тогда на сегмент выдаётся ровно один кадр с его собственными границами.
    Количество строк здесь не ограничивается max_lines: кадр один, и урезать
    текст сегмента значило бы потерять расшифровку.
    """
    text = " ".join(segment.text.split())
    if not text:
        return []
    return [
        Cue(
            start=segment.start,
            end=segment.end,
            lines=wrap_words(text.split(" "), max_line),
            words=0,
        )
    ]


def normalize_cues(
    cues: list[Cue], source_duration: float | None, max_line: int
) -> list[Cue]:
    """Приводит кадры к требованиям SRT: монотонность, непустота, предел записи.

    В транскрипте таймкоды слов могут слегка наезжать друг на друга (стык чанков)
    или совпадать до миллисекунды. Плееры такое читают по-разному, поэтому здесь
    кадры выпрямляются: start не раньше конца предыдущего кадра, end минимум на
    миллисекунду больше start, ничего не выходит за конец записи.
    """
    limit = None if source_duration is None else int(source_duration * 1000)

    out: list[Cue] = []
    for cue in cues:
        if not cue.lines:
            continue
        start = max(0, cue.start)
        end = max(start, cue.end)
        if limit is not None:
            start = min(start, limit)
            end = min(end, limit)
        if out and start < out[-1].end:
            start = out[-1].end
        if end < start + 1:
            end = start + 1

        if limit is not None and end > limit:
            # Места под отдельный кадр в хвосте записи не осталось: таймкоды
            # транскрипта дошли ровно до её конца. Текст не выбрасываем —
            # дописываем в предыдущий кадр, иначе часть расшифровки исчезла бы
            # из субтитров молча.
            if out:
                merged = " ".join(" ".join(out[-1].lines + cue.lines).split())
                out[-1] = Cue(
                    start=out[-1].start,
                    end=out[-1].end,
                    lines=wrap_words(merged.split(" "), max_line),
                    words=out[-1].words + cue.words,
                )
                continue
            raise TranscriptError(
                "таймкоды транскрипта не помещаются в source_duration (%s с) — "
                "похоже, транскрипт собран не от этой записи" % source_duration
            )

        out.append(Cue(start=start, end=end, lines=list(cue.lines), words=cue.words))
    return out


def render_srt(cues: list[Cue]) -> str:
    """Собирает текст файла SRT: нумерация с 1, между кадрами одна пустая строка."""
    if not cues:
        raise TranscriptError("в транскрипте нет слов и текста — субтитры собирать не из чего")
    blocks = []
    for number, cue in enumerate(cues, start=1):
        blocks.append(
            "%d\n%s --> %s\n%s"
            % (number, format_ms(cue.start), format_ms(cue.end), "\n".join(cue.lines))
        )
    # Пустая строка — разделитель кадров, а не окончание файла, поэтому склейка
    # через "\n\n" плюс один финальный перевод строки.
    return "\n\n".join(blocks) + "\n"


def render_txt(segments: list[Segment]) -> str:
    """Собирает читаемый текст: абзац на сегмент, абзацы разделены пустой строкой.

    Текст сегмента идёт как есть — обрезаются только внешние пробелы. Регистр и
    пунктуацию не правим: чего нет в транскрипте, того инструмент не придумает.
    """
    paragraphs = [segment.text.strip() for segment in segments]
    paragraphs = [text for text in paragraphs if text]
    if not paragraphs:
        raise TranscriptError("во всех сегментах транскрипта пустой текст — писать нечего")
    return "\n\n".join(paragraphs) + "\n"


def write_text_file(path: str, content: str) -> None:
    """Пишет файл в UTF-8 с переводами строк LF.

    newline="\\n" задан явно: на Windows иначе получились бы CRLF, и «ровно одна
    пустая строка между кадрами» перестала бы быть правдой.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except OSError as exc:
        raise TranscriptError(
            "не удалось записать файл %s (%s)" % (path, exc.strerror or exc)
        ) from None


# ------------------------------------------------------------------- отчёт ---


def count_words(segments: list[Segment]) -> int:
    """Слов в транскрипте: по пословным таймкодам, а где их нет — по тексту."""
    total = 0
    for segment in segments:
        total += len(segment.words) if segment.words else len(segment.text.split())
    return total


def count_chars(segments: list[Segment]) -> int:
    """Символов в тексте сегментов (без разделителей абзацев)."""
    return sum(len(segment.text.strip()) for segment in segments)


def print_report(
    transcript: Transcript, cues: list[Cue], written: list[tuple[str, str]]
) -> None:
    """Печатает итоги рендера — то, что потом уходит в отчёт о прогоне."""
    total_ms = sum(cue.end - cue.start for cue in cues)
    sys.stdout.write("================ РЕНДЕР ТРАНСКРИПТА ================\n")
    sys.stdout.write("Транскрипт:    %s\n" % transcript.source)
    sys.stdout.write("Сегментов:     %d\n" % len(transcript.segments))
    sys.stdout.write("Кадров SRT:    %d\n" % len(cues))
    sys.stdout.write("Слов:          %d\n" % count_words(transcript.segments))
    sys.stdout.write("Символов:      %d\n" % count_chars(transcript.segments))
    sys.stdout.write(
        "Длительность кадров: %s (%.3f с)\n" % (format_ms(total_ms), total_ms / 1000.0)
    )
    sys.stdout.write("\n== Пути результатов\n")
    for kind, path in written:
        sys.stdout.write("  %-4s %s\n" % (kind, path))


# --------------------------------------------------------- разбор аргументов ---

_FLAGS_PATH = {"--transcript": "transcript", "--txt": "txt", "--srt": "srt"}
_FLAGS_INT = {"--max-line": "max_line", "--max-lines": "max_lines"}
_FLAGS_FLOAT = {"--max-cue": "max_cue", "--gap": "gap"}


def _positive_int(flag: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise UsageError("%s ожидает целое число, получено: %s" % (flag, raw)) from None
    if value < 1:
        raise UsageError("%s должен быть больше нуля, получено: %s" % (flag, raw))
    return value


def _positive_float(flag: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise UsageError("%s ожидает число секунд, получено: %s" % (flag, raw)) from None
    if not math.isfinite(value) or value <= 0:
        raise UsageError("%s должен быть больше нуля, получено: %s" % (flag, raw))
    return value


def parse_args(argv: list[str]) -> dict:
    """Разбирает аргументы так же, как это делают скрипты проекта в scripts/.

    Своим разбором, а не argparse: тот печатает usage и свои ошибки по-английски,
    а весь остальной вывод инструментов проекта — на русском.
    """
    opts: dict = {
        "transcript": None,
        "txt": None,
        "srt": None,
        "max_line": DEFAULT_MAX_LINE,
        "max_lines": DEFAULT_MAX_LINES,
        "max_cue": DEFAULT_MAX_CUE,
        "gap": DEFAULT_GAP,
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

        if token in _FLAGS_PATH or token in _FLAGS_INT or token in _FLAGS_FLOAT:
            if inline is None:
                if position + 1 >= len(argv):
                    raise UsageError("%s требует значение" % token)
                value = argv[position + 1]
                position += 2
            else:
                value = inline
                position += 1

            if token in _FLAGS_PATH:
                if not value:
                    raise UsageError("%s требует непустой путь" % token)
                opts[_FLAGS_PATH[token]] = value
            elif token in _FLAGS_INT:
                opts[_FLAGS_INT[token]] = _positive_int(token, value)
            else:
                opts[_FLAGS_FLOAT[token]] = _positive_float(token, value)
            continue

        if token.startswith("-"):
            raise UsageError("неизвестная опция: %s (см. --help)" % token)
        raise UsageError("лишний аргумент: %s (см. --help)" % token)

    return opts


def check_args(opts: dict) -> None:
    """Проверяет набор аргументов целиком — то, что нельзя проверить по одному."""
    if not opts["transcript"]:
        raise UsageError("не указан --transcript (см. --help)")
    if not opts["txt"] and not opts["srt"]:
        raise UsageError("не указано, что писать: нужен --txt и/или --srt (см. --help)")
    if opts["txt"] and opts["srt"]:
        same = os.path.abspath(opts["txt"]) == os.path.abspath(opts["srt"])
        if same:
            raise UsageError("--txt и --srt указывают на один файл: %s" % opts["txt"])


# ---------------------------------------------------------------- селф-тест ---

# Строгая регулярка разбора собственного SRT: номер кадра без ведущих нулей,
# таймкоды ровно в формате HH:MM:SS,mmm, минимум одна непустая строка текста и
# либо пустая строка-разделитель, либо конец файла.
SRT_CUE_RE = re.compile(
    r"(?P<index>[1-9][0-9]*)\n"
    r"(?P<start>[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3})"
    r" --> "
    r"(?P<end>[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3})\n"
    r"(?P<text>(?:[^\n]+\n)+)"
    r"(?:\n|\Z)"
)

# Любой намёк на таймкод — им в .txt делать нечего.
TIMECODE_IN_TEXT_RE = re.compile(r"\d{1,2}:\d{2}|-->")


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _parse_ms(timestamp: str) -> int:
    """Разбирает HH:MM:SS,mmm обратно в миллисекунды (для обратного разбора SRT)."""
    hours, minutes, tail = timestamp.split(":")
    secs, ms = tail.split(",")
    return (int(hours) * 3600 + int(minutes) * 60 + int(secs)) * 1000 + int(ms)


def parse_srt(text: str) -> list[tuple[int, int, int, list[str]]]:
    """Строгий обратный разбор SRT. Любое отступление от формата — AssertionError."""
    _check(text, "файл SRT пуст")
    _check(text.endswith("\n"), "файл SRT не заканчивается переводом строки")
    _check("\r" not in text, "в файле SRT есть возврат каретки")
    _check("\n\n\n" not in text, "между кадрами больше одной пустой строки")

    cues: list[tuple[int, int, int, list[str]]] = []
    offset = 0
    for match in SRT_CUE_RE.finditer(text):
        _check(
            match.start() == offset,
            "в файле SRT есть мусор вне кадров на позиции %d" % offset,
        )
        offset = match.end()
        cues.append(
            (
                int(match.group("index")),
                _parse_ms(match.group("start")),
                _parse_ms(match.group("end")),
                match.group("text").rstrip("\n").split("\n"),
            )
        )
    _check(offset == len(text), "хвост файла SRT не разобран (с позиции %d)" % offset)
    _check(cues, "в файле SRT не нашлось ни одного кадра")
    return cues


def _fixture_transcript() -> dict:
    """Транскрипт для проверок: собирается в памяти, без аудио и внешних файлов.

    Сегмент 1 — около шестидесяти слов средней длины с одной длинной паузой
    посередине: на нём срабатывают закрытие кадра по вёрстке и по паузе.
    Сегмент 2 — короткие слова по секунде каждое: на нём срабатывает закрытие по
    max-cue, до предела строк там далеко.
    """
    words = []
    clock = 0.0
    for number in range(60):
        duration = 0.30
        words.append(
            {"text": "слово%02d" % number, "start": round(clock, 3), "end": round(clock + duration, 3)}
        )
        # Пауза 0.90 с в середине — заметно больше gap по умолчанию (0.4 с).
        clock += duration + (0.90 if number == 29 else 0.05)
    first_end = round(clock, 3)

    slow = []
    clock = round(clock + 1.0, 3)
    for number in range(14):
        slow.append({"text": ["да", "ну", "вот", "ещё"][number % 4], "start": round(clock, 3), "end": round(clock + 1.0, 3)})
        clock = round(clock + 1.0, 3)
    second_end = round(clock, 3)

    return {
        "schema": SCHEMA,
        "source": "data/samples/fixture.wav",
        "source_duration": round(second_end + 2.0, 3),
        "model": "v3_rnnt",
        "device": "cpu",
        "plan": "data/samples/fixture.chunks.json",
        "generated_at": "2026-10-03T00:00:00+03:00",
        "text": " ".join(w["text"] for w in words) + " " + " ".join(w["text"] for w in slow),
        "segments": [
            {
                "index": 0,
                "start": 0.0,
                "end": first_end,
                "text": " ".join(w["text"] for w in words),
                "words": words,
            },
            {
                "index": 1,
                "start": words[-1]["end"],
                "end": second_end,
                "text": " ".join(w["text"] for w in slow),
                "words": slow,
            },
        ],
        "metrics": {"что_угодно": [1, 2, 3], "вложенное": {"ключ": None}},
    }


def _test_timestamps() -> None:
    """Формат таймкодов — три контрольных значения из критериев приёмки."""
    _check(format_timestamp(0.0) == "00:00:00,000", "0.0 -> %s" % format_timestamp(0.0))
    _check(
        format_timestamp(59.999) == "00:00:59,999",
        "59.999 -> %s" % format_timestamp(59.999),
    )
    _check(
        format_timestamp(3661.5) == "01:01:01,500",
        "3661.5 -> %s" % format_timestamp(3661.5),
    )
    # Запятая и ровно три знака миллисекунд — на произвольных значениях тоже.
    for seconds in (0.0005, 1.0, 61.25, 3599.999, 7322.0):
        stamp = format_timestamp(seconds)
        _check(
            re.fullmatch(r"\d{2}:\d{2}:\d{2},\d{3}", stamp) is not None,
            "таймкод не в формате HH:MM:SS,mmm: %s" % stamp,
        )
    _check(format_ms(3_600_000) == "01:00:00,000", "час -> %s" % format_ms(3_600_000))


def _test_cue_split() -> None:
    """Разбиение сегментов на кадры и все ограничения на кадр."""
    max_line, max_lines, max_cue, gap = 42, 2, 6.0, 0.4
    transcript = parse_transcript(_fixture_transcript())
    cues = normalize_cues(
        build_cues(
            transcript.segments,
            max_line=max_line,
            max_lines=max_lines,
            max_cue=max_cue,
            gap=gap,
        ),
        transcript.source_duration,
        max_line,
    )

    _check(len(cues) > 1, "сегмент из 60 слов остался одним кадром")

    previous_end = -1
    for number, cue in enumerate(cues, start=1):
        _check(cue.lines, "кадр %d пустой" % number)
        _check(all(line.strip() for line in cue.lines), "в кадре %d пустая строка" % number)
        _check(
            len(cue.lines) <= max_lines,
            "в кадре %d строк %d, предел %d" % (number, len(cue.lines), max_lines),
        )
        for line in cue.lines:
            _check(
                len(line) <= max_line or " " not in line,
                "в кадре %d строка длиной %d символов: %r" % (number, len(line), line),
            )
        _check(cue.end > cue.start, "в кадре %d end не больше start" % number)
        _check(
            cue.start >= previous_end,
            "кадр %d начинается раньше, чем кончился предыдущий" % number,
        )
        duration = cue.end - cue.start
        _check(
            duration <= to_ms(max_cue) or cue.words <= 1,
            "кадр %d длится %d мс, предел %d мс" % (number, duration, to_ms(max_cue)),
        )
        _check(
            cue.end <= to_ms(transcript.source_duration),
            "кадр %d выходит за длительность записи" % number,
        )
        previous_end = cue.end

    # Границы кадров — это границы слов, а не вычисленные заново значения.
    all_words = [word for segment in transcript.segments for word in segment.words]
    _check(cues[0].start == all_words[0].start, "start первого кадра не равен start первого слова")
    _check(cues[-1].end == all_words[-1].end, "end последнего кадра не равен end последнего слова")

    # Ни одно слово не потеряно и порядок сохранён.
    from_cues = " ".join(" ".join(cue.lines) for cue in cues).split()
    _check(
        from_cues == [word.text for word in all_words],
        "состав слов в кадрах не совпал с транскриптом (%d против %d)"
        % (len(from_cues), len(all_words)),
    )

    # Срабатывание каждого из трёх правил закрытия кадра по отдельности.
    _check(
        any(len(cue.lines) == max_lines for cue in cues),
        "закрытие кадра по вёрстке ни разу не сработало",
    )
    _check(
        any(cue.end - cue.start > to_ms(max_cue) - 1000 for cue in cues),
        "закрытие кадра по max-cue ни разу не сработало",
    )

    # Правило паузы проверяется отдельным сегментом, где ни вёрстка, ни max-cue
    # сработать не могут: четыре коротких слова с одной паузой 0.5 с посередине.
    # На кадрах из общей фикстуры это неотличимо — там пауза попадает в стык,
    # закрытый по вёрстке, и правило паузы можно было бы выключить незаметно.
    pause_segments = parse_transcript(
        {
            "schema": SCHEMA,
            "source": "x.wav",
            "source_duration": 5.0,
            "segments": [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 1.6,
                    "text": "да ну да ну",
                    "words": [
                        {"text": "да", "start": 0.0, "end": 0.2},
                        {"text": "ну", "start": 0.25, "end": 0.45},
                        {"text": "да", "start": 0.95, "end": 1.15},
                        {"text": "ну", "start": 1.20, "end": 1.40},
                    ],
                }
            ],
        }
    ).segments
    by_pause = build_cues(pause_segments, max_line=max_line, max_lines=max_lines, max_cue=max_cue, gap=gap)
    _check(
        len(by_pause) == 2,
        "пауза 0.5 с при --gap %s дала %d кадров вместо 2" % (gap, len(by_pause)),
    )
    _check(
        [len(cue.lines) for cue in by_pause] == [1, 1]
        and [cue.words for cue in by_pause] == [2, 2],
        "кадр разошёлся не по паузе: %r" % [cue.lines for cue in by_pause],
    )
    # Пауза меньше --gap кадр не закрывает.
    by_pause_wide = build_cues(
        pause_segments, max_line=max_line, max_lines=max_lines, max_cue=max_cue, gap=1.0
    )
    _check(
        len(by_pause_wide) == 1,
        "при --gap 1.0 пауза 0.5 с всё равно закрыла кадр (%d кадров)" % len(by_pause_wide),
    )

    # Другие ограничения дают другое разбиение — значит флаги действительно работают.
    narrow = build_cues(transcript.segments, max_line=20, max_lines=1, max_cue=max_cue, gap=gap)
    _check(len(narrow) > len(cues), "сужение строки не увеличило число кадров")


def _test_srt_roundtrip() -> None:
    """Обратный разбор собственного SRT строгой регуляркой."""
    transcript = parse_transcript(_fixture_transcript())
    cues = normalize_cues(
        build_cues(transcript.segments),
        transcript.source_duration,
        DEFAULT_MAX_LINE,
    )
    with tempfile.TemporaryDirectory(prefix="outputs-selftest-") as workdir:
        path = os.path.join(workdir, "subs.srt")
        write_text_file(path, render_srt(cues))
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()

    parsed = parse_srt(raw)
    _check(
        len(parsed) == len(cues),
        "разобрано %d кадров вместо %d" % (len(parsed), len(cues)),
    )
    for position, (index, start, end, lines) in enumerate(parsed):
        cue = cues[position]
        _check(index == position + 1, "нумерация кадров сбилась на %d: %d" % (position + 1, index))
        _check(start == cue.start, "кадр %d: start после разбора %d вместо %d" % (index, start, cue.start))
        _check(end == cue.end, "кадр %d: end после разбора %d вместо %d" % (index, end, cue.end))
        _check(lines == cue.lines, "кадр %d: текст после разбора не совпал" % index)
        _check(end > start, "кадр %d: end не больше start" % index)
        if position:
            _check(start >= parsed[position - 1][2], "кадр %d пересекается с предыдущим" % index)

    # Кадр без текста формат не принимает — регулярка обязана это заметить.
    broken = "1\n00:00:00,000 --> 00:00:01,000\n\n"
    try:
        parse_srt(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("строгий разбор принял кадр без текста")


def _test_without_words() -> None:
    """Ветка-страховка: у сегментов нет пословных таймкодов."""
    data = {
        "schema": SCHEMA,
        "source": "x.wav",
        "source_duration": 30.0,
        "segments": [
            {"index": 0, "start": 0.0, "end": 12.0, "text": "первый чанк без пословных таймкодов"},
            {"index": 1, "start": 12.0, "end": 24.0, "text": "второй чанк тоже", "words": []},
        ],
    }
    transcript = parse_transcript(data)
    cues = normalize_cues(build_cues(transcript.segments), transcript.source_duration, DEFAULT_MAX_LINE)
    _check(len(cues) == 2, "ожидалось по одному кадру на сегмент, получено %d" % len(cues))
    for cue, segment in zip(cues, transcript.segments):
        _check(cue.start == segment.start, "кадр не взял start сегмента")
        _check(cue.end == segment.end, "кадр не взял end сегмента")
        _check(cue.words == 0, "кадр без пословных таймкодов считает слова")
        _check(
            " ".join(cue.lines) == " ".join(segment.text.split()),
            "текст сегмента в кадре изменился",
        )
    parse_srt(render_srt(cues))

    # Смешанный транскрипт: один сегмент со словами, другой без — не падает.
    mixed = _fixture_transcript()
    mixed["segments"][1].pop("words")
    mixed_cues = build_cues(parse_transcript(mixed).segments)
    _check(len(mixed_cues) > 1, "смешанный транскрипт дал один кадр")


def _test_txt_layout() -> None:
    """Структура .txt: абзац на сегмент, без таймкодов, с финальным переводом строки."""
    data = _fixture_transcript()
    transcript = parse_transcript(data)
    with tempfile.TemporaryDirectory(prefix="outputs-selftest-") as workdir:
        path = os.path.join(workdir, "вложенный", "text.txt")
        write_text_file(path, render_txt(transcript.segments))
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()

    _check(raw.endswith("\n"), "файл .txt не заканчивается переводом строки")
    _check("\n\n\n" not in raw, "абзацы в .txt разделены больше чем одной пустой строкой")
    _check(TIMECODE_IN_TEXT_RE.search(raw) is None, "в .txt попал таймкод")
    paragraphs = raw.rstrip("\n").split("\n\n")
    _check(
        len(paragraphs) == len(transcript.segments),
        "абзацев %d, а сегментов %d" % (len(paragraphs), len(transcript.segments)),
    )
    for paragraph, segment in zip(paragraphs, transcript.segments):
        _check(paragraph == segment.text.strip(), "текст сегмента в .txt переписан")

    # Пустые сегменты не должны оставлять лишних пустых строк.
    data["segments"].insert(1, {"index": 9, "start": 1.0, "end": 1.5, "text": "   ", "words": []})
    sparse = render_txt(parse_transcript(data).segments)
    _check("\n\n\n" not in sparse, "пустой сегмент оставил лишнюю пустую строку в .txt")


def _test_duration_clamp() -> None:
    """Кадры не выходят за source_duration, даже если таймкоды слов выходят."""
    data = {
        "schema": SCHEMA,
        "source": "x.wav",
        "source_duration": 10.0,
        "segments": [
            {
                "index": 0,
                "start": 8.0,
                "end": 12.0,
                "text": "раз два",
                "words": [
                    {"text": "раз", "start": 8.0, "end": 9.5},
                    {"text": "два", "start": 9.4, "end": 12.0},
                ],
            }
        ],
    }
    transcript = parse_transcript(data)
    cues = normalize_cues(build_cues(transcript.segments), transcript.source_duration, DEFAULT_MAX_LINE)
    _check(cues, "кадры исчезли при обрезке по длительности")
    for cue in cues:
        _check(cue.end <= 10_000, "кадр вышел за source_duration: end=%d" % cue.end)
        _check(cue.end > cue.start, "кадр после обрезки стал пустым по времени")
    text = " ".join(" ".join(cue.lines) for cue in cues).split()
    _check(text == ["раз", "два"], "при обрезке по длительности потерялись слова: %r" % text)


def _test_broken_input() -> None:
    """Битый вход даёт понятное сообщение и ненулевой код, а не трейсбек."""
    cases = [
        ("не объект", []),
        ("нет schema", {"segments": []}),
        ("чужая schema", {"schema": "transcript/2", "segments": [], "source_duration": 1.0}),
        ("нет segments", {"schema": SCHEMA, "source_duration": 1.0}),
        ("segments не список", {"schema": SCHEMA, "source_duration": 1.0, "segments": {}}),
        ("segments пуст", {"schema": SCHEMA, "source_duration": 1.0, "segments": []}),
        (
            "нет text у сегмента",
            {"schema": SCHEMA, "source_duration": 1.0, "segments": [{"start": 0.0, "end": 1.0}]},
        ),
        (
            "нет start у сегмента",
            {"schema": SCHEMA, "source_duration": 1.0, "segments": [{"end": 1.0, "text": "а"}]},
        ),
        (
            "start не число",
            {
                "schema": SCHEMA,
                "source_duration": 1.0,
                "segments": [{"start": "ноль", "end": 1.0, "text": "а"}],
            },
        ),
        (
            "слово без таймкода",
            {
                "schema": SCHEMA,
                "source_duration": 1.0,
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "а", "words": [{"text": "а", "start": 0.0}]}
                ],
            },
        ),
        (
            "words не список",
            {
                "schema": SCHEMA,
                "source_duration": 1.0,
                "segments": [{"start": 0.0, "end": 1.0, "text": "а", "words": 5}],
            },
        ),
    ]
    for name, payload in cases:
        try:
            parse_transcript(payload)
        except TranscriptError as exc:
            _check(str(exc).strip() != "", "случай %r: пустое сообщение об ошибке" % name)
        else:
            raise AssertionError("случай %r прошёл проверку структуры" % name)

    with tempfile.TemporaryDirectory(prefix="outputs-selftest-") as workdir:
        good = os.path.join(workdir, "good.json")
        with open(good, "w", encoding="utf-8") as handle:
            json.dump(_fixture_transcript(), handle, ensure_ascii=False)
        broken = os.path.join(workdir, "broken.json")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write('{"schema": "transcript/1", "segments": [')

        # Сквозной прогон: main() обязан вернуть ненулевой код и написать в
        # stderr, а не вывалить трейсбек.
        for argv, what in (
            (["--transcript", broken, "--txt", os.path.join(workdir, "a.txt")], "битый JSON"),
            (["--transcript", os.path.join(workdir, "нет.json"), "--srt", "b.srt"], "нет файла"),
            (["--transcript", good], "без --txt и --srt"),
            (["--transcript", good, "--txt", "a.txt", "--max-line", "0"], "--max-line 0"),
            (["--transcript", good, "--txt", "a.txt", "--gap", "ненаю"], "--gap не число"),
            (["--кривая-опция"], "неизвестная опция"),
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = main(argv)
            _check(code != 0, "случай %r завершился кодом 0" % what)
            _check(stderr.getvalue().startswith(PROG + ":"), "случай %r ничего не написал в stderr" % what)

        # И наоборот: на целом транскрипте сквозной прогон проходит и пишет оба файла.
        txt_path = os.path.join(workdir, "out", "lecture.txt")
        srt_path = os.path.join(workdir, "out", "lecture.srt")
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = main(["--transcript", good, "--txt", txt_path, "--srt", srt_path])
        _check(code == 0, "рендер целого транскрипта вернул код %d" % code)
        _check(os.path.isfile(txt_path) and os.path.isfile(srt_path), "файлы вывода не созданы")
        report = stdout.getvalue()
        for needle in ("Сегментов:", "Кадров SRT:", "Слов:", "Символов:", "Длительность кадров:"):
            _check(needle in report, "в отчёте нет строки %r" % needle)
        _check(txt_path in report and srt_path in report, "в отчёте нет путей результатов")
        with open(srt_path, "r", encoding="utf-8") as handle:
            parse_srt(handle.read())

        # --txt и --srt задаются независимо: один флаг — один файл.
        only_txt = os.path.join(workdir, "only.txt")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main(["--transcript", good, "--txt", only_txt])
        _check(code == 0, "прогон только с --txt вернул код %d" % code)
        _check(os.path.isfile(only_txt), "файл .txt не создан")
        _check(not os.path.isfile(os.path.join(workdir, "only.srt")), "создан лишний .srt")


def selftest() -> int:
    """Прогоняет встроенные проверки. Код возврата 0 — все прошли."""
    tests = (
        ("форматирование таймкодов", _test_timestamps),
        ("разбиение сегмента на кадры", _test_cue_split),
        ("обратный разбор собственного SRT", _test_srt_roundtrip),
        ("ветка без пословных таймкодов", _test_without_words),
        ("структура .txt", _test_txt_layout),
        ("обрезка по длительности записи", _test_duration_clamp),
        ("битый вход и коды возврата", _test_broken_input),
    )
    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ outputs.py ================\n")
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
        transcript = load_transcript(opts["transcript"])
        cues = normalize_cues(
            build_cues(
                transcript.segments,
                max_line=opts["max_line"],
                max_lines=opts["max_lines"],
                max_cue=opts["max_cue"],
                gap=opts["gap"],
            ),
            transcript.source_duration,
            opts["max_line"],
        )

        # Содержимое готовится до записи: лучше упасть с внятным сообщением, чем
        # оставить на диске половину результата.
        written: list[tuple[str, str]] = []
        payload: list[tuple[str, str, str]] = []
        if opts["txt"]:
            payload.append(("txt", opts["txt"], render_txt(transcript.segments)))
        if opts["srt"]:
            payload.append(("srt", opts["srt"], render_srt(cues)))
        for kind, path, content in payload:
            write_text_file(path, content)
            written.append((kind, path))
    except (UsageError, TranscriptError) as exc:
        return die(str(exc))

    print_report(transcript, cues, written)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
