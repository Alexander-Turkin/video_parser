#!/usr/bin/env python3
#
# check_timecodes.py — объективные проверки пригодности результата прогона.
#
# Берёт JSON транскрипта схемы transcript/1 (его пишет tools/transcribe.py) и,
# если передан, собранный .srt, и прогоняет обязательные проверки допуска из
# docs/research/benchmark-plan.md, раздел 7: монотонность, корректность и
# непустоту сегментов, вложенность и монотонность пословных таймкодов, границы
# записи, полноту, покрытие, отсутствие зацикливания и валидность субтитров.
#
# Инструмент только диагностирует: ни транскрипт, ни субтитры он не правит.
# На выходе — таблица PASS/FAIL, однострочный итог и вывод о допуске. По коду
# возврата видно, годен ли результат: 0 — все проверки прошли, 1 — хотя бы одна
# FAIL, 2 — входные данные непригодны (файла нет, битый JSON, чужая схема).
#
# Использование:
#   python tools/check_timecodes.py --transcript data/out/v3_rnnt.json \
#       --srt data/out/v3_rnnt.srt --json data/out/v3_rnnt.checks.json
#   python tools/check_timecodes.py --selftest
#
# Зависимости: только стандартная библиотека. Ни torch, ни gigaam, ни ffmpeg
# здесь не нужны и не вызываются.

import argparse
import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

PROG = "check_timecodes.py"

TRANSCRIPT_SCHEMA = "transcript/1"
PLAN_SCHEMA = "chunk-plan/1"
CHECKS_SCHEMA = "checks/1"

PLAN_REFERENCE = "docs/research/benchmark-plan.md, раздел 7"

DEFAULT_TOLERANCE = 0.05

# Полнота: последний таймкод не раньше чем за столько секунд до конца записи
# (в разделе 7 это «не раньше 04:50 из 05:00»).
COMPLETENESS_SLACK = 10.0

# Покрытие: доля речи в сумме «сегменты + выброшенные паузы».
COVERAGE_MIN = 0.90

# Зацикливание: повтор подряд более этого числа раз — брак.
LOOP_MAX_REPEATS = 3

# Длина последовательности слов, повтор которой считаем зацикливанием.
PHRASE_WORDS = 5

# Сколько нарушений печатать в details, чтобы вывод не превращался в дамп.
MAX_DETAILS = 5

SRT_TIME_RE = re.compile(
    r"^(?P<h1>\d{2}):(?P<m1>\d{2}):(?P<s1>\d{2}),(?P<ms1>\d{3})"
    r" --> "
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}),(?P<ms2>\d{3})$"
)

# Всё, что не буква, не цифра и не пробел, при нормализации становится пробелом:
# пунктуация и регистр не должны мешать ловить зацикливание.
NON_WORD_RE = re.compile(r"[^0-9a-zA-Zа-яёА-ЯЁ\s]+")


class ToolError(Exception):
    """Ошибка непригодного входа: о ней говорят человеку, а не трейсбеком."""

    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


def warn(message):
    print(f"{PROG}: {message}", file=sys.stderr)


# --- мелкие форматтеры -----------------------------------------------------


def format_hms(seconds):
    """Секунды в HH:MM:SS.mmm — чтобы FAIL можно было найти в плеере."""
    total = max(0.0, float(seconds))
    hours, rest = divmod(total, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def sec(seconds):
    return f"{float(seconds):.3f} с"


def percent(value):
    return f"{value * 100:.1f} %"


def times(count):
    """«1 раз», «4 раза», «5 раз» — согласование числа со словом «раз»."""
    if count % 100 not in (11, 12, 13, 14) and count % 10 in (2, 3, 4):
        return f"{count} раза"
    return f"{count} раз"


def normalize_text(text):
    """Текст к виду, в котором сравнивают на повтор: без пунктуации и регистра."""
    lowered = NON_WORD_RE.sub(" ", str(text).lower())
    return " ".join(lowered.split())


# --- чтение входа ----------------------------------------------------------


def read_json(path, what):
    """Читает JSON. Отсутствие файла и битый JSON — непригодный вход, не трейсбек."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ToolError(f"{what} не найден: {path}") from None
    except OSError as err:
        raise ToolError(f"не удалось прочитать {what} {path}: {err}") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        raise ToolError(f"{what} {path} — не валидный JSON: {err}") from None


def read_text(path, what):
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ToolError(f"{what} не найден: {path}") from None
    except (OSError, UnicodeDecodeError) as err:
        raise ToolError(f"не удалось прочитать {what} {path}: {err}") from None


def _number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{what}: ожидалось число, получено {value!r}")
    return float(value)


def validate_transcript(transcript, path):
    """Проверяет, что вход вообще пригоден для проверок, и нормализует его.

    Здесь ловится только то, без чего проверки бессмысленны: чужая схема, битая
    структура, отсутствие source_duration. Содержательные дефекты (пустой текст,
    поехавшие таймкоды) — это FAIL проверок, а не непригодный вход.
    """
    if not isinstance(transcript, dict):
        raise ToolError(f"транскрипт {path} не является объектом JSON")
    schema = transcript.get("schema")
    if schema != TRANSCRIPT_SCHEMA:
        raise ToolError(
            f"ожидалась схема транскрипта {TRANSCRIPT_SCHEMA!r}, в файле — {schema!r}"
        )

    source_duration = transcript.get("source_duration")
    if source_duration is None:
        raise ToolError(
            "в транскрипте нет source_duration — без длительности записи нельзя "
            "проверить границы и полноту"
        )
    source_duration = _number(source_duration, "source_duration")
    if source_duration <= 0:
        raise ToolError(f"source_duration должен быть больше нуля, получено {source_duration}")

    raw_segments = transcript.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ToolError("в транскрипте нет сегментов: поле segments пусто или отсутствует")

    segments = []
    for position, segment in enumerate(raw_segments):
        if not isinstance(segment, dict):
            raise ToolError(f"сегмент на позиции {position}: ожидался объект JSON")
        number = segment.get("index", position)
        for field in ("start", "end", "text"):
            if field not in segment:
                raise ToolError(f"сегмент {number}: нет обязательного поля {field!r}")
        if not isinstance(segment["text"], str):
            raise ToolError(f"сегмент {number}: text должен быть строкой")

        raw_words = segment.get("words") or []
        if not isinstance(raw_words, list):
            raise ToolError(f"сегмент {number}: words должен быть списком")
        words = []
        for word_position, word in enumerate(raw_words):
            if not isinstance(word, dict):
                raise ToolError(
                    f"сегмент {number}, слово {word_position}: ожидался объект JSON"
                )
            for field in ("text", "start", "end"):
                if field not in word:
                    raise ToolError(
                        f"сегмент {number}, слово {word_position}: нет поля {field!r}"
                    )
            words.append(
                {
                    "text": str(word["text"]),
                    "start": _number(word["start"], f"сегмент {number}: start слова"),
                    "end": _number(word["end"], f"сегмент {number}: end слова"),
                }
            )

        segments.append(
            {
                "index": number,
                "position": position,
                "start": _number(segment["start"], f"сегмент {number}: start"),
                "end": _number(segment["end"], f"сегмент {number}: end"),
                "text": segment["text"],
                "words": words,
            }
        )

    return {"source_duration": source_duration, "segments": segments}


def load_plan_info(transcript, transcript_path):
    """Пробует найти plan.json, на который ссылается транскрипт.

    Данных о выброшенных паузах в транскрипте нет, поэтому покрытие считается по
    разрывам между сегментами, а план — только сверка. План может быть недоступен
    (его путь относителен каталогу запуска прогона), и это не ошибка: тогда
    проверки честно скажут, что работали без плана.
    """
    reference = transcript.get("plan")
    if not isinstance(reference, str) or not reference:
        return {"path": None, "reason": "транскрипт не ссылается на plan.json"}

    candidates = [Path(reference).expanduser()]
    if not candidates[0].is_absolute():
        candidates.append(Path(transcript_path).parent / reference)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            plan = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
            return {"path": None, "reason": f"план {candidate} не читается: {err}"}
        if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
            return {
                "path": None,
                "reason": f"план {candidate} не схемы {PLAN_SCHEMA}",
            }
        dropped = []
        for item in plan.get("dropped") or []:
            if isinstance(item, dict) and "start" in item and "end" in item:
                try:
                    dropped.append(
                        {
                            "start": _number(item["start"], "план: start паузы"),
                            "end": _number(item["end"], "план: end паузы"),
                        }
                    )
                except ToolError:
                    continue
        dropped_seconds = plan.get("dropped_seconds")
        if isinstance(dropped_seconds, bool) or not isinstance(
            dropped_seconds, (int, float)
        ):
            dropped_seconds = sum(item["end"] - item["start"] for item in dropped) or None
        return {
            "path": candidate,
            "reason": None,
            "dropped": dropped,
            "dropped_seconds": None if dropped_seconds is None else float(dropped_seconds),
        }
    return {"path": None, "reason": f"файл плана {reference} недоступен"}


# --- разбор SRT ------------------------------------------------------------


def parse_srt(content):
    """Строгий разбор SRT. Возвращает (кадры, ошибки формата).

    Строгий — значит ровно так, как пишет tools/outputs.py: номер кадра на
    отдельной строке, таймкод HH:MM:SS,mmm --> HH:MM:SS,mmm, затем непустой
    текст, между кадрами пустая строка. Всё остальное — ошибка формата, а не
    повод угадывать.
    """
    frames = []
    errors = []
    text = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    blocks = [block for block in re.split(r"\n[ \t]*\n", text.strip("\n")) if block.strip()]
    if not blocks:
        errors.append("файл субтитров пуст")
        return frames, errors

    for position, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 3:
            errors.append(
                f"кадр {position}: ожидались номер, таймкод и текст, "
                f"получено строк: {len(lines)}"
            )
            continue
        number_line, time_line = lines[0].strip(), lines[1].strip()
        if not number_line.isdigit():
            errors.append(f"кадр {position}: первая строка не номер кадра: {lines[0]!r}")
            continue
        match = SRT_TIME_RE.match(time_line)
        if match is None:
            errors.append(
                f"кадр {position}: таймкод не в формате HH:MM:SS,mmm --> HH:MM:SS,mmm: "
                f"{time_line!r}"
            )
            continue
        body = "\n".join(lines[2:])
        if not body.strip():
            errors.append(f"кадр {position}: пустой текст")
            continue

        groups = match.groupdict()
        start = (
            int(groups["h1"]) * 3600
            + int(groups["m1"]) * 60
            + int(groups["s1"])
            + int(groups["ms1"]) / 1000.0
        )
        end = (
            int(groups["h2"]) * 3600
            + int(groups["m2"]) * 60
            + int(groups["s2"])
            + int(groups["ms2"]) / 1000.0
        )
        frames.append(
            {
                "position": position,
                "number": int(number_line),
                "start": start,
                "end": end,
                "text": body.strip(),
            }
        )
    return frames, errors


# --- проверки --------------------------------------------------------------


def check(name, problems, ok_details):
    """Собирает результат проверки из списка нарушений."""
    if problems:
        shown = problems[:MAX_DETAILS]
        if len(problems) > MAX_DETAILS:
            shown = shown + [f"… и ещё нарушений: {len(problems) - MAX_DETAILS}"]
        return {"name": name, "passed": False, "details": "\n".join(shown)}
    return {"name": name, "passed": True, "details": ok_details}


def check_monotonic_segments(data, tolerance):
    segments = data["segments"]
    problems = []
    previous = None
    for segment in segments:
        if previous is not None and segment["start"] < previous["end"] - tolerance:
            problems.append(
                f"сегмент {segment['index']}: start {sec(segment['start'])} "
                f"({format_hms(segment['start'])}) меньше end предыдущего сегмента "
                f"{previous['index']} — {sec(previous['end'])}; "
                f"наложение {sec(previous['end'] - segment['start'])}"
            )
        previous = segment
    return check(
        "монотонность сегментов",
        problems,
        f"сегментов: {len(segments)}, идут по возрастанию, допуск {sec(tolerance)}",
    )


def check_valid_segments(data):
    problems = []
    for segment in data["segments"]:
        if segment["end"] <= segment["start"]:
            problems.append(
                f"сегмент {segment['index']}: end {sec(segment['end'])} не больше "
                f"start {sec(segment['start'])}"
            )
        if not segment["text"].strip():
            problems.append(
                f"сегмент {segment['index']} ({format_hms(segment['start'])}–"
                f"{format_hms(segment['end'])}): пустой текст"
            )
    total = sum(segment["end"] - segment["start"] for segment in data["segments"])
    return check(
        "корректность сегментов",
        problems,
        f"у всех сегментов end > start и непустой текст, речи {sec(total)}",
    )


def check_words_inside(data, tolerance):
    problems = []
    words = 0
    for segment in data["segments"]:
        words += len(segment["words"])
        for position, word in enumerate(segment["words"]):
            if word["start"] < segment["start"] - tolerance:
                problems.append(
                    f"сегмент {segment['index']}, слово {position} {word['text']!r}: "
                    f"start {sec(word['start'])} раньше начала сегмента "
                    f"{sec(segment['start'])}"
                )
            if word["end"] > segment["end"] + tolerance:
                problems.append(
                    f"сегмент {segment['index']}, слово {position} {word['text']!r}: "
                    f"end {sec(word['end'])} позже конца сегмента {sec(segment['end'])}"
                )
    return check(
        "вложенность пословных таймкодов",
        problems,
        f"слов: {words}, все в границах своих сегментов, допуск {sec(tolerance)}",
    )


def check_words_monotonic(data, tolerance):
    problems = []
    for segment in data["segments"]:
        previous = None
        for position, word in enumerate(segment["words"]):
            if word["end"] < word["start"]:
                problems.append(
                    f"сегмент {segment['index']}, слово {position} {word['text']!r}: "
                    f"end {sec(word['end'])} меньше start {sec(word['start'])}"
                )
            if previous is not None and word["start"] < previous["end"] - tolerance:
                problems.append(
                    f"сегмент {segment['index']}, слово {position} {word['text']!r}: "
                    f"start {sec(word['start'])} меньше end предыдущего слова "
                    f"{previous['text']!r} — {sec(previous['end'])}"
                )
            previous = word
    return check(
        "монотонность пословных таймкодов",
        problems,
        f"внутри каждого сегмента слова идут по возрастанию, допуск {sec(tolerance)}",
    )


def check_bounds(data, tolerance):
    source_duration = data["source_duration"]
    problems = []
    limit = source_duration + tolerance
    for segment in data["segments"]:
        if segment["end"] > limit:
            problems.append(
                f"сегмент {segment['index']}: end {sec(segment['end'])} "
                f"({format_hms(segment['end'])}) выходит за длительность записи "
                f"{sec(source_duration)} (допуск {sec(tolerance)})"
            )
    last_end = max(segment["end"] for segment in data["segments"])
    return check(
        "границы записи",
        problems,
        f"последний end {sec(last_end)} ≤ {sec(source_duration)} + допуск {sec(tolerance)}",
    )


def check_completeness(data, tolerance, plan_info):
    """Полнота: распознан весь фрагмент, хвост не потерян.

    Нарезка выбрасывает паузы длиннее 2 с, поэтому законный хвост тишины тоже
    окажется выброшенным — и формально последний таймкод будет раньше конца
    записи. Если такой хвост подтверждён планом, проверка проходит, но с явной
    оговоркой в выводе: молча считать её пройденной нельзя.
    """
    source_duration = data["source_duration"]
    last_end = max(segment["end"] for segment in data["segments"])
    threshold = source_duration - COMPLETENESS_SLACK
    tail = source_duration - last_end
    where = (
        f"последний end {sec(last_end)} ({format_hms(last_end)}) при записи "
        f"{sec(source_duration)}, порог {sec(threshold)}"
    )

    if last_end >= threshold - tolerance:
        # Отрицательный хвост означает выход за запись — это дело проверки границ.
        tail_note = (
            f"хвост {sec(tail)}"
            if tail >= 0
            else f"последний end за пределами записи на {sec(-tail)} — "
            f"см. проверку «границы записи»"
        )
        return check("полнота", [], f"{where}; {tail_note}")

    # Хвост короче порога — ищем ему законное объяснение в плане нарезки.
    tail_pause = None
    for pause in plan_info.get("dropped") or []:
        if pause["start"] <= last_end + tolerance and pause["end"] >= source_duration - tolerance:
            tail_pause = pause
            break
    if tail_pause is not None:
        return check(
            "полнота",
            [],
            f"{where};\nОГОВОРКА: хвост {sec(tail)} выброшен нарезкой как длинная пауза "
            f"{format_hms(tail_pause['start'])}–{format_hms(tail_pause['end'])} "
            f"(по плану {plan_info['path']}) — речь не потеряна, "
            f"но формально порог {sec(threshold)} не достигнут",
        )

    reason = (
        f"план нарезки недоступен ({plan_info.get('reason')}), "
        f"подтвердить выброшенную паузу нечем"
        if plan_info.get("path") is None
        else f"в плане {plan_info['path']} нет выброшенной паузы, покрывающей этот хвост"
    )
    return check("полнота", [f"{where}; {reason}"], "")


def check_coverage(data, plan_info):
    """Покрытие: сколько речи в сумме «сегменты + выброшенные паузы».

    Данных о выброшенных паузах в транскрипте нет, поэтому они оцениваются как
    разрывы между соседними сегментами. Сами разрывы нарушением не считаются:
    выбрасывать паузы длиннее 2 с — решение владельца, дыры в таймкодах законны.
    """
    segments = data["segments"]
    speech = sum(max(0.0, segment["end"] - segment["start"]) for segment in segments)
    gaps = []
    previous = None
    for segment in segments:
        if previous is not None:
            gap = segment["start"] - previous["end"]
            if gap > 0:
                gaps.append({"start": previous["end"], "end": segment["start"], "duration": gap})
        previous = segment
    dropped = sum(gap["duration"] for gap in gaps)
    total = speech + dropped

    numbers = (
        f"речь {sec(speech)}, разрывов {len(gaps)} на {sec(dropped)}, "
        f"итого {sec(total)}, покрытие {percent(speech / total) if total > 0 else '—'} "
        f"при пороге {percent(COVERAGE_MIN)}"
    )
    if gaps:
        widest = max(gaps, key=lambda gap: gap["duration"])
        numbers += (
            f"\nсамый длинный разрыв {sec(widest['duration'])}: "
            f"{format_hms(widest['start'])}–{format_hms(widest['end'])}"
        )
    if plan_info.get("path") is None:
        numbers += (
            f"\nпаузы оценены только по разрывам между сегментами: "
            f"{plan_info.get('reason')}"
        )
    else:
        plan_dropped = plan_info.get("dropped_seconds")
        if plan_dropped is None:
            numbers += f"\nплан {plan_info['path']} найден, но dropped_seconds в нём нет"
        else:
            numbers += (
                f"\nпо плану {plan_info['path']} выброшено {sec(plan_dropped)}; "
                f"расхождение с оценкой по разрывам {sec(abs(plan_dropped - dropped))} "
                f"(хвостовая и головная паузы в разрывы между сегментами не попадают)"
            )

    if total <= 0:
        return check("покрытие", [f"нулевая суммарная длительность: {numbers}"], "")
    coverage = speech / total
    if coverage < COVERAGE_MIN:
        return check(
            "покрытие",
            [
                f"покрытие {percent(coverage)} ниже порога {percent(COVERAGE_MIN)}: "
                f"{numbers}"
            ],
            "",
        )
    return check("покрытие", [], numbers)


def word_stream(data):
    """Сквозной поток нормализованных слов с привязкой ко времени и сегменту.

    Пословные таймкоды есть не всегда (модель без word_timestamps их не даёт),
    поэтому при их отсутствии слова берутся из текста сегмента: проверка на
    зацикливание нужна и в этом случае.
    """
    stream = []
    for segment in data["segments"]:
        if segment["words"]:
            for word in segment["words"]:
                token = normalize_text(word["text"])
                if token:
                    stream.append((token, word["start"], segment["index"]))
        else:
            for token in normalize_text(segment["text"]).split():
                stream.append((token, segment["start"], segment["index"]))
    return stream


def longest_text_run(segments):
    """Самая длинная цепочка подряд идущих сегментов с одинаковым текстом."""
    best = None
    run_start = None
    run_length = 0
    previous = None
    for segment in segments:
        token = normalize_text(segment["text"])
        if token and token == previous:
            run_length += 1
        else:
            run_start, run_length = segment, 1
        if token and (best is None or run_length > best[0]):
            best = (run_length, run_start, segment, token)
        previous = token
    return best


def longest_phrase_run(stream, size=PHRASE_WORDS):
    """Самый длинный подряд идущий повтор одной последовательности из size слов."""
    best = None
    total = len(stream)
    for index in range(total - size + 1):
        phrase = tuple(item[0] for item in stream[index : index + size])
        count = 1
        cursor = index + size
        while cursor + size <= total and (
            tuple(item[0] for item in stream[cursor : cursor + size]) == phrase
        ):
            count += 1
            cursor += size
        if count > 1 and (best is None or count > best[0]):
            best = (count, index, cursor, phrase)
    return best


def check_no_looping(data):
    problems = []
    segments = data["segments"]

    text_run = longest_text_run(segments)
    if text_run and text_run[0] > LOOP_MAX_REPEATS:
        count, first, last, token = text_run
        problems.append(
            f"текст сегмента повторён {times(count)} подряд (предел {LOOP_MAX_REPEATS}): "
            f"сегменты {first['index']}–{last['index']}, "
            f"{format_hms(first['start'])}–{format_hms(last['end'])}, текст {token!r}"
        )

    stream = word_stream(data)
    phrase_run = longest_phrase_run(stream)
    if phrase_run and phrase_run[0] > LOOP_MAX_REPEATS:
        count, index, cursor, phrase = phrase_run
        start_time = stream[index][1]
        end_time = stream[cursor - 1][1]
        problems.append(
            f"последовательность из {PHRASE_WORDS} слов повторена {times(count)} подряд "
            f"(предел {LOOP_MAX_REPEATS}): {' '.join(phrase)!r}, "
            f"сегмент {stream[index][2]}, "
            f"{format_hms(start_time)}–{format_hms(end_time)}"
        )

    max_text_run = text_run[0] if text_run else 0
    max_phrase_run = phrase_run[0] if phrase_run else 0
    return check(
        "отсутствие зацикливания",
        problems,
        f"максимум подряд: один и тот же текст сегмента {times(max_text_run)}, "
        f"последовательность из {PHRASE_WORDS} слов {times(max_phrase_run)} "
        f"(предел {LOOP_MAX_REPEATS}); слов в потоке: {len(stream)}",
    )


def check_srt(content, srt_path, data, tolerance):
    source_duration = data["source_duration"]
    frames, problems = parse_srt(content)
    previous = None
    for frame in frames:
        if frame["number"] != frame["position"]:
            problems.append(
                f"кадр {frame['position']}: номер {frame['number']} — "
                f"нумерация должна идти подряд с 1"
            )
        if frame["end"] <= frame["start"]:
            problems.append(
                f"кадр {frame['position']}: end {sec(frame['end'])} не больше "
                f"start {sec(frame['start'])}"
            )
        if previous is not None and frame["start"] < previous["end"] - tolerance:
            problems.append(
                f"кадр {frame['position']}: start {sec(frame['start'])} пересекается с "
                f"кадром {previous['position']}, который кончается в {sec(previous['end'])}"
            )
        if frame["end"] > source_duration + tolerance:
            problems.append(
                f"кадр {frame['position']}: end {sec(frame['end'])} "
                f"({format_hms(frame['end'])}) выходит за длительность записи "
                f"{sec(source_duration)}"
            )
        previous = frame

    last_end = frames[-1]["end"] if frames else 0.0
    return check(
        "валидность SRT",
        problems,
        f"кадров: {len(frames)}, нумерация подряд с 1, текст непустой, пересечений нет, "
        f"последний end {sec(last_end)} ≤ {sec(source_duration)} + допуск {sec(tolerance)}"
        f"\nфайл: {srt_path}",
    )


def run_checks(data, tolerance, plan_info, srt_content=None, srt_path=None):
    """Весь набор проверок допуска из раздела 7 плана замера."""
    checks = [
        check_monotonic_segments(data, tolerance),
        check_valid_segments(data),
        check_words_inside(data, tolerance),
        check_words_monotonic(data, tolerance),
        check_bounds(data, tolerance),
        check_completeness(data, tolerance, plan_info),
        check_coverage(data, plan_info),
        check_no_looping(data),
    ]
    if srt_content is not None:
        checks.append(check_srt(srt_content, srt_path, data, tolerance))
    return checks


# --- вывод -----------------------------------------------------------------


def print_report(checks, data, transcript_path, srt_path, tolerance, plan_info):
    segments = data["segments"]
    words = sum(len(segment["words"]) for segment in segments)
    print(f"Проверки пригодности результата — {PLAN_REFERENCE}")
    print(f"  транскрипт: {transcript_path}")
    print(f"  субтитры:   {srt_path if srt_path else '— (не передан --srt)'}")
    print(f"  план:       {plan_info['path'] or '— ' + str(plan_info.get('reason'))}")
    print(
        f"  запись:     {sec(data['source_duration'])} "
        f"({format_hms(data['source_duration'])}), сегментов: {len(segments)}, "
        f"слов: {words}, допуск: {sec(tolerance)}"
    )
    print()

    width = max(len(item["name"]) for item in checks)
    for item in checks:
        status = "PASS" if item["passed"] else "FAIL"
        lines = [line for line in str(item["details"]).split("\n") if line]
        head = lines[0] if lines else ""
        print(f"{status}  {item['name']:<{width}}  {head}")
        for line in lines[1:]:
            print(f"      {'':<{width}}  {line}")

    passed = sum(1 for item in checks if item["passed"])
    print()
    print(f"ИТОГО: пройдено {passed} из {len(checks)} проверок")
    if passed == len(checks):
        print(
            "ДОПУСК ПРОЙДЕН: таймкоды годны, результат можно брать в сравнение по WER"
        )
    else:
        print(
            "ДОПУСК НЕ ПРОЙДЕН: результат в сравнение по WER не берётся, каким бы "
            "низким WER ни был"
        )
    print(f"  критерии допуска — {PLAN_REFERENCE}")


def build_machine_report(checks, transcript_path, srt_path):
    return {
        "schema": CHECKS_SCHEMA,
        "transcript": str(transcript_path),
        "srt": None if srt_path is None else str(srt_path),
        "passed": all(item["passed"] for item in checks),
        "checks": [
            {"name": item["name"], "passed": item["passed"], "details": item["details"]}
            for item in checks
        ],
    }


def write_machine_report(report, json_path):
    """Машинночитаемый итог: '-' — в stdout, иначе в файл."""
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(json_path) == "-":
        print()
        print(payload, end="")
        return None
    path = Path(json_path).expanduser()
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


# --- прогон ----------------------------------------------------------------


def run(args):
    if args.tolerance < 0:
        raise ToolError(f"--tolerance не может быть отрицательным: {args.tolerance}")

    transcript_path = Path(args.transcript).expanduser()
    transcript = read_json(transcript_path, "транскрипт")
    data = validate_transcript(transcript, transcript_path)
    plan_info = load_plan_info(transcript, transcript_path)

    srt_path = None
    srt_content = None
    if args.srt:
        srt_path = Path(args.srt).expanduser()
        srt_content = read_text(srt_path, "файл субтитров")

    checks = run_checks(
        data, args.tolerance, plan_info, srt_content=srt_content, srt_path=srt_path
    )
    print_report(checks, data, transcript_path, srt_path, args.tolerance, plan_info)

    if args.json_path:
        report = build_machine_report(checks, transcript_path, srt_path)
        written = write_machine_report(report, args.json_path)
        if written is not None:
            print(f"  машинночитаемый итог — {written}")

    return 0 if all(item["passed"] for item in checks) else 1


# --- селф-тест -------------------------------------------------------------


def _words(texts, start, end):
    """Раскладывает слова равномерно внутри [start, end] — фикстура для проверок."""
    step = (end - start) / max(1, len(texts))
    items = []
    for number, text in enumerate(texts):
        items.append(
            {
                "text": text,
                "start": round(start + number * step, 3),
                "end": round(start + (number + 1) * step, 3),
            }
        )
    return items


def _segment(index, start, end, text):
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "words": _words(text.split(), start, end),
    }


def _good_transcript():
    """Заведомо годный транскрипт: все проверки на нём обязаны проходить.

    Разрыв 2,5 с между первым и вторым сегментом — законная выброшенная пауза,
    а не дефект: нарезка выбрасывает паузы длиннее 2 с.
    """
    segments = [
        _segment(0, 0.0, 20.0, "первый сегмент про таймкоды и границы"),
        _segment(1, 22.5, 44.0, "второй сегмент идёт после выброшенной паузы"),
        _segment(2, 44.0, 58.0, "третий сегмент заканчивает фрагмент записи"),
    ]
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "source": "data/samples/fragment.wav",
        "source_duration": 60.0,
        "model": "v3_rnnt",
        "device": "cpu",
        "plan": "data/chunks/plan.json",
        "generated_at": "2026-10-03T12:00:00+03:00",
        "text": " ".join(segment["text"] for segment in segments),
        "segments": segments,
        "metrics": {},
    }


def _srt_for(transcript):
    """Собирает корректный SRT по пословным таймкодам — как tools/outputs.py.

    Кадров больше, чем сегментов: кадр собирается по словам, и это ожидаемо.
    """

    def stamp(value):
        total = max(0.0, float(value))
        hours, rest = divmod(total, 3600.0)
        minutes, secs = divmod(rest, 60.0)
        whole = int(secs)
        millis = int(round((secs - whole) * 1000))
        if millis == 1000:  # округление вверх не должно давать ,1000
            whole, millis = whole + 1, 0
        return f"{int(hours):02d}:{int(minutes):02d}:{whole:02d},{millis:03d}"

    blocks = []
    number = 0
    for segment in transcript["segments"]:
        words = segment["words"]
        for position in range(0, len(words), 3):
            group = words[position : position + 3]
            number += 1
            blocks.append(
                f"{number}\n{stamp(group[0]['start'])} --> {stamp(group[-1]['end'])}\n"
                + " ".join(word["text"] for word in group)
            )
    return "\n\n".join(blocks) + "\n"


def _fixture(directory, transcript, srt=None, plan=None, name="transcript"):
    """Кладёт фикстуры на диск и возвращает пути (транскрипт, srt)."""
    directory = Path(directory)
    if plan is not None:
        (directory / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False), encoding="utf-8"
        )
        transcript = dict(transcript, plan="plan.json")
    transcript_path = directory / f"{name}.json"
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    srt_path = None
    if srt is not None:
        srt_path = directory / f"{name}.srt"
        srt_path.write_text(srt, encoding="utf-8")
    return transcript_path, srt_path


def _run_cli(argv):
    """Прогон main() с перехватом вывода: возвращает (код возврата, stdout).

    stderr тоже перехватывается и отбрасывается: часть фикстур намеренно
    непригодна, и их сообщения об ошибках не должны мешать читать селф-тест.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
        code = main(argv)
    return code, buffer.getvalue()


def _statuses(output):
    """Разбирает таблицу вывода в {имя проверки: PASS|FAIL}."""
    statuses = {}
    for line in output.split("\n"):
        if line.startswith(("PASS  ", "FAIL  ")):
            statuses[line[6:].split("  ")[0].strip()] = line[:4]
    return statuses


def _expect_only_failed(output, name):
    """Проверяет, что упала ровно та проверка, которой адресована фикстура."""
    statuses = _statuses(output)
    assert statuses.get(name) == "FAIL", f"проверка {name!r} должна была упасть: {statuses}"
    others = {key: value for key, value in statuses.items() if key != name and value == "FAIL"}
    assert not others, f"упали посторонние проверки: {others}"


def _expect_error(func, *fragments):
    try:
        func()
    except ToolError as err:
        message = str(err)
        for fragment in fragments:
            assert fragment in message, f"в сообщении {message!r} нет {fragment!r}"
        return message
    raise AssertionError("ожидалась ToolError, но вызов прошёл без ошибки")


def _check_good_passes():
    transcript = _good_transcript()
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, srt_path = _fixture(tmp, transcript, srt=_srt_for(transcript))
        code, output = _run_cli(["--transcript", str(transcript_path), "--srt", str(srt_path)])
        statuses = _statuses(output)
        assert len(statuses) == 9, statuses
        assert set(statuses.values()) == {"PASS"}, statuses
        assert "ИТОГО: пройдено 9 из 9 проверок" in output, output
        assert "ДОПУСК ПРОЙДЕН" in output, output
        assert PLAN_REFERENCE in output, output
        assert code == 0, (code, output)

        # Без --srt проверок на одну меньше, но их всё равно не меньше восьми.
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 0, (code, output)
        statuses = _statuses(output)
        assert len(statuses) == 8, statuses
        assert "валидность SRT" not in statuses, statuses


def _check_not_monotonic():
    transcript = _good_transcript()
    # Второй сегмент заезжает на первый: 19.0 < 20.0.
    transcript["segments"][1]["start"] = 19.0
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "монотонность сегментов")
        assert "сегмент 1" in output and "19.000" in output, output


def _check_empty_text():
    transcript = _good_transcript()
    transcript["segments"][1]["text"] = "   "
    transcript["segments"][1]["words"] = []
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "корректность сегментов")
        assert "пустой текст" in output and "сегмент 1" in output, output


def _check_beyond_source_duration():
    transcript = _good_transcript()
    transcript["segments"][2]["end"] = 65.0
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "границы записи")
        assert "65.000" in output and "60.000" in output, output


def _check_word_outside_segment():
    transcript = _good_transcript()
    transcript["segments"][0]["words"][-1]["end"] = 21.0
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "вложенность пословных таймкодов")
        assert "21.000" in output, output


def _check_word_not_monotonic():
    transcript = _good_transcript()
    words = transcript["segments"][1]["words"]
    words[2]["start"] = words[0]["start"]
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "монотонность пословных таймкодов")


def _check_looping_segments():
    """Фраза, повторённая 4 раза подряд отдельными сегментами."""
    segments = [
        _segment(0, 0.0, 10.0, "начало фрагмента про зацикливание"),
        _segment(1, 10.0, 20.0, "и так далее"),
        _segment(2, 20.0, 30.0, "и так далее"),
        _segment(3, 30.0, 40.0, "и так далее"),
        _segment(4, 40.0, 55.0, "и так далее"),
    ]
    transcript = dict(_good_transcript(), segments=segments)
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "отсутствие зацикливания")
        assert "4 раза подряд" in output, output
        assert "и так далее" in output, output


def _check_looping_phrase():
    """Пятисловная последовательность, повторённая 4 раза внутри одного сегмента."""
    phrase = "давайте посмотрим на этот слайд"
    segments = [
        _segment(0, 0.0, 20.0, "первый сегмент без повторов вообще"),
        {
            "index": 1,
            "start": 22.5,
            "end": 58.0,
            "text": " ".join([phrase] * 4),
            "words": _words(" ".join([phrase] * 4).split(), 22.5, 58.0),
        },
    ]
    transcript = dict(_good_transcript(), segments=segments)
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "отсутствие зацикливания")
        assert "из 5 слов повторена 4 раза подряд" in output, output

        # То же без пословных таймкодов: поток слов берётся из текста сегмента.
        for segment in transcript["segments"]:
            segment["words"] = []
        transcript_path, _ = _fixture(tmp, transcript, name="без-слов")
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "отсутствие зацикливания")


def _check_looping_tolerated():
    """Три повтора подряд — предел, а не нарушение."""
    segments = [
        _segment(0, 0.0, 15.0, "и так далее"),
        _segment(1, 15.0, 30.0, "и так далее"),
        _segment(2, 30.0, 45.0, "и так далее"),
        _segment(3, 45.0, 58.0, "дальше другой текст сегмента"),
    ]
    transcript = dict(_good_transcript(), segments=segments)
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 0, (code, output)


def _check_srt_broken_timecode():
    transcript = _good_transcript()
    srt = _srt_for(transcript).replace("00:00:00,000", "00:00:0,0", 1)
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, srt_path = _fixture(tmp, transcript, srt=srt)
        code, output = _run_cli(["--transcript", str(transcript_path), "--srt", str(srt_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "валидность SRT")
        assert "кадр 1" in output and "HH:MM:SS,mmm" in output, output


def _check_srt_broken_numbering():
    transcript = _good_transcript()
    blocks = _srt_for(transcript).strip("\n").split("\n\n")
    assert len(blocks) >= 3, blocks
    lines = blocks[2].split("\n")
    lines[0] = "7"  # нумерация сбита: 1, 2, 7, 4, …
    blocks[2] = "\n".join(lines)
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, srt_path = _fixture(tmp, transcript, srt="\n\n".join(blocks) + "\n")
        code, output = _run_cli(["--transcript", str(transcript_path), "--srt", str(srt_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "валидность SRT")
        assert "нумерация должна идти подряд с 1" in output, output


def _check_srt_other_defects():
    transcript = _good_transcript()
    good = _srt_for(transcript)
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        data = validate_transcript(json.loads(transcript_path.read_text("utf-8")), transcript_path)

        # Пустой текст кадра.
        broken = good.split("\n\n")
        broken[0] = "\n".join(broken[0].split("\n")[:2])
        result = check_srt("\n\n".join(broken), "x.srt", data, DEFAULT_TOLERANCE)
        assert not result["passed"], result
        assert "кадр 1" in result["details"], result

        # Кадр за пределами записи.
        beyond = good + "\n99\n00:01:05,000 --> 00:01:07,000\nхвост\n"
        result = check_srt(beyond, "x.srt", data, DEFAULT_TOLERANCE)
        assert not result["passed"], result
        assert "выходит за длительность записи" in result["details"], result

        # Пересечение кадров.
        frames, errors = parse_srt(good)
        assert not errors, errors
        assert len(frames) > len(transcript["segments"]), "кадров больше, чем сегментов"
        overlap = (
            "1\n00:00:00,000 --> 00:00:10,000\nпервый\n\n"
            "2\n00:00:05,000 --> 00:00:12,000\nвторой\n"
        )
        result = check_srt(overlap, "x.srt", data, DEFAULT_TOLERANCE)
        assert not result["passed"], result
        assert "пересекается" in result["details"], result

        # Пустой файл субтитров.
        result = check_srt("\n\n", "x.srt", data, DEFAULT_TOLERANCE)
        assert not result["passed"], result
        assert "пуст" in result["details"], result


def _check_completeness_tail():
    """Выброшенный хвост: с планом — PASS с оговоркой, без плана — FAIL."""
    segments = [
        _segment(0, 0.0, 20.0, "первый сегмент про таймкоды и границы"),
        _segment(1, 22.5, 45.0, "второй сегмент кончается до конца записи"),
    ]
    transcript = dict(_good_transcript(), segments=segments)
    plan = {
        "schema": PLAN_SCHEMA,
        "source_duration": 60.0,
        "chunks": [],
        "dropped": [{"start": 20.0, "end": 22.5}, {"start": 45.0, "end": 60.0}],
        "dropped_seconds": 17.5,
    }
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript, plan=plan)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 0, (code, output)
        assert _statuses(output)["полнота"] == "PASS", output
        assert "ОГОВОРКА" in output and "длинная пауза" in output, output
        # Покрытие считается по разрывам, а план печатается как сверка.
        assert "по плану" in output and "17.500" in output, output

        # Тот же транскрипт без доступного плана: объяснить хвост нечем.
        transcript_path, _ = _fixture(tmp, transcript, name="без-плана")
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "полнота")
        assert "недоступен" in output, output


def _check_coverage_numbers():
    """Покрытие печатает цифры и падает, когда речи меньше 90 %."""
    segments = [
        _segment(0, 0.0, 5.0, "короткая реплика в начале"),
        _segment(1, 50.0, 58.0, "короткая реплика в конце записи"),
    ]
    transcript = dict(_good_transcript(), segments=segments)
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 1, (code, output)
        _expect_only_failed(output, "покрытие")
        # 13 с речи при разрыве 45 с — это 22,4 % покрытия.
        assert "речь 13.000 с" in output, output
        assert "22.4 %" in output and "90.0 %" in output, output
        assert "самый длинный разрыв 45.000 с" in output, output


def _check_tolerance_flag():
    """--tolerance: мелкое наложение в пределах допуска нарушением не считается."""
    transcript = _good_transcript()
    transcript["segments"][2]["start"] = 43.96  # на 0,04 с раньше конца сегмента 1
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, _ = _fixture(tmp, transcript)
        code, output = _run_cli(["--transcript", str(transcript_path)])
        assert code == 0, (code, output)
        code, output = _run_cli(
            ["--transcript", str(transcript_path), "--tolerance", "0.01"]
        )
        assert code == 1, (code, output)
        _expect_only_failed(output, "монотонность сегментов")


def _check_machine_report():
    transcript = _good_transcript()
    with tempfile.TemporaryDirectory() as tmp:
        transcript_path, srt_path = _fixture(tmp, transcript, srt=_srt_for(transcript))
        out_path = Path(tmp) / "итог" / "checks.json"
        code, output = _run_cli(
            [
                "--transcript",
                str(transcript_path),
                "--srt",
                str(srt_path),
                "--json",
                str(out_path),
            ]
        )
        assert code == 0, (code, output)
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert set(report) == {"schema", "transcript", "srt", "passed", "checks"}, report
        assert report["schema"] == CHECKS_SCHEMA
        assert report["transcript"] == str(transcript_path)
        assert report["srt"] == str(srt_path)
        assert report["passed"] is True
        assert len(report["checks"]) == 9, report["checks"]
        for item in report["checks"]:
            assert set(item) == {"name", "passed", "details"}, item
            assert isinstance(item["passed"], bool)
        raw = out_path.read_text(encoding="utf-8")
        assert "\\u" not in raw, "ensure_ascii=False: кириллица пишется как есть"

        # Без --srt поле srt — null, а итог при FAIL — false.
        transcript["segments"][1]["text"] = ""
        transcript["segments"][1]["words"] = []
        transcript_path, _ = _fixture(tmp, transcript, name="плохой")
        out_path = Path(tmp) / "плохой.checks.json"
        code, _ = _run_cli(
            ["--transcript", str(transcript_path), "--json", str(out_path)]
        )
        assert code == 1, code
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["srt"] is None, report
        assert report["passed"] is False, report
        assert len(report["checks"]) == 8, report["checks"]


def _check_bad_input():
    """Непригодный вход — код 2, а не трейсбек и не FAIL."""
    transcript = _good_transcript()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert _run_cli(["--transcript", str(tmp / "нет.json")])[0] == 2

        broken = tmp / "битый.json"
        broken.write_text("{не json", encoding="utf-8")
        assert _run_cli(["--transcript", str(broken)])[0] == 2

        alien = tmp / "чужая-схема.json"
        alien.write_text(
            json.dumps(dict(transcript, schema="transcript/2"), ensure_ascii=False),
            encoding="utf-8",
        )
        code, _ = _run_cli(["--transcript", str(alien)])
        assert code == 2, code

        # Нет --srt-файла — это тоже непригодный вход, а не FAIL проверки.
        transcript_path, _ = _fixture(tmp, transcript)
        code, _ = _run_cli(
            ["--transcript", str(transcript_path), "--srt", str(tmp / "нет.srt")]
        )
        assert code == 2, code

        # Не указан --transcript.
        assert _run_cli([])[0] == 2

        # Отрицательный допуск.
        code, _ = _run_cli(
            ["--transcript", str(transcript_path), "--tolerance", "-1"]
        )
        assert code == 2, code

    _expect_error(
        lambda: validate_transcript(dict(transcript, source_duration=None), "x.json"),
        "source_duration",
    )
    _expect_error(
        lambda: validate_transcript(dict(transcript, segments=[]), "x.json"), "нет сегментов"
    )
    _expect_error(
        lambda: validate_transcript(
            dict(transcript, segments=[{"index": 0, "start": 0.0, "end": 1.0}]), "x.json"
        ),
        "text",
    )
    _expect_error(
        lambda: validate_transcript(
            dict(
                transcript,
                segments=[{"index": 0, "start": "ноль", "end": 1.0, "text": "а"}],
            ),
            "x.json",
        ),
        "ожидалось число",
    )
    _expect_error(lambda: validate_transcript([], "x.json"), "не является объектом JSON")


def _check_normalization():
    assert normalize_text("Привет, МИР!") == "привет мир"
    assert normalize_text("  два   пробела  ") == "два пробела"
    assert normalize_text("...") == ""
    # Пунктуация и регистр не должны прятать зацикливание.
    assert normalize_text("Да!") == normalize_text("да,")


def _check_stdlib_only():
    """Инструмент обязан обходиться стандартной библиотекой.

    Имена импортов берём из разбора собственного файла, а не грепом: слова
    «torch» и «ffmpeg» законно встречаются в комментариях.
    """
    import ast

    for module in ("torch", "torchaudio", "gigaam", "numpy"):
        assert module not in sys.modules, f"{module} не должен импортироваться"

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    stdlib = getattr(sys, "stdlib_module_names", None)
    assert stdlib, "нужен Python 3.10+: без sys.stdlib_module_names проверку не сделать"
    outsiders = sorted(name for name in imported if name not in stdlib)
    assert not outsiders, f"импортируется не стандартная библиотека: {outsiders}"
    # Внешние процессы не запускаются: ни ffmpeg, ни что-либо ещё.
    assert "subprocess" not in imported and "os" not in imported, sorted(imported)


CHECKS = (
    ("годный транскрипт и его SRT проходят все проверки", _check_good_passes),
    ("немонотонные сегменты", _check_not_monotonic),
    ("пустой текст сегмента", _check_empty_text),
    ("end за пределами source_duration", _check_beyond_source_duration),
    ("слово вне границ сегмента", _check_word_outside_segment),
    ("слова внутри сегмента не по возрастанию", _check_word_not_monotonic),
    ("фраза, повторённая 4 раза подряд сегментами", _check_looping_segments),
    ("пятисловная последовательность 4 раза подряд", _check_looping_phrase),
    ("три повтора подряд — ещё не зацикливание", _check_looping_tolerated),
    ("битый SRT: сломанный таймкод", _check_srt_broken_timecode),
    ("битый SRT: сбитая нумерация", _check_srt_broken_numbering),
    ("битый SRT: пустой текст, пересечение, выход за запись", _check_srt_other_defects),
    ("полнота: выброшенный хвост с планом и без", _check_completeness_tail),
    ("покрытие: цифры и падение ниже 90 %", _check_coverage_numbers),
    ("--tolerance меняет вердикт", _check_tolerance_flag),
    ("--json пишет итог схемы checks/1", _check_machine_report),
    ("непригодный вход даёт код 2", _check_bad_input),
    ("нормализация текста для сравнения", _check_normalization),
    ("только стандартная библиотека", _check_stdlib_only),
)


def selftest():
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
        print(f"селф-тест: провалено {failed} из {len(CHECKS)} проверок")
        return 1
    print(f"селф-тест: пройдено {len(CHECKS)} из {len(CHECKS)} проверок")
    return 0


# --- точка входа -----------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python tools/check_timecodes.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Объективные проверки пригодности результата прогона по JSON схемы\n"
            "transcript/1 и, если передан, по собранному .srt.\n\n"
            f"Критерии допуска — {PLAN_REFERENCE}:\n"
            "монотонность, корректность и непустота сегментов, вложенность и\n"
            "монотонность пословных таймкодов, границы записи, полнота, покрытие,\n"
            "отсутствие зацикливания и валидность субтитров.\n"
            "Инструмент только диагностирует и ничего не правит."
        ),
        epilog=(
            "Примеры:\n"
            "  python tools/check_timecodes.py --transcript data/out/v3_rnnt.json\n"
            "  python tools/check_timecodes.py --transcript data/out/v3_rnnt.json \\\n"
            "      --srt data/out/v3_rnnt.srt --json data/out/v3_rnnt.checks.json\n"
            "  python tools/check_timecodes.py --selftest\n\n"
            "Как считается покрытие: данных о выброшенных паузах в транскрипте нет,\n"
            "поэтому они оцениваются как разрывы между соседними сегментами, а поля\n"
            "dropped и dropped_seconds из plan.json (путь берётся из поля plan)\n"
            "печатаются как сверка. Если план недоступен, инструмент говорит об этом\n"
            "в выводе. Дыры в таймкодах сами по себе нарушением не считаются:\n"
            "нарезка выбрасывает паузы длиннее 2 с.\n\n"
            "Коды возврата: 0 — все проверки прошли, 1 — хотя бы одна FAIL,\n"
            "2 — входные данные непригодны (файла нет, битый JSON, чужая схема)."
        ),
    )
    parser.add_argument(
        "--transcript",
        metavar="ПУТЬ",
        help="JSON транскрипта схемы transcript/1 (обязателен для проверки)",
    )
    parser.add_argument(
        "--srt",
        metavar="ПУТЬ",
        help="собранный файл субтитров .srt (необязательно)",
    )
    parser.add_argument(
        "--json",
        metavar="ПУТЬ",
        dest="json_path",
        help="куда писать машинночитаемый итог схемы checks/1 ('-' — в stdout)",
    )
    parser.add_argument(
        "--tolerance",
        metavar="СЕКУНДЫ",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "допуск при сравнении таймкодов, секунды "
            f"(по умолчанию {DEFAULT_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="прогнать встроенные проверки на синтетике и выйти",
    )
    parser.add_argument("-h", "--help", action="help", help="показать эту справку и выйти")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        if not args.transcript:
            raise ToolError("не указан --transcript (см. --help)")
        return run(args)
    except ToolError as err:
        warn(str(err))
        return err.code


if __name__ == "__main__":
    raise SystemExit(main())
