#!/usr/bin/env python3
"""
align_timecodes.py — сдвиг таймкодов транскрипта относительно меток эталона.

Что считает. Берёт эталон со строками «[ММ:СС] Роль: текст» и JSON транскрипта
схемы transcript/1 (его пишет tools/transcribe.py) и отвечает на один вопрос:
на сколько секунд расходится время реплики в эталоне и время тех же слов в
транскрипте. На выходе — медиана сдвига, максимум по модулю, 90-й процентиль,
доля меток, которые сопоставить не удалось, и таблица по строкам эталона.
Решение владельца: сдвиг таймкодов замеряется отдельной метрикой, медиана и
максимум по записи.

Как сопоставляются слова. Оба текста раскладываются в поток слов — эталон
построчно, транскрипт по пословным таймкодам, — и выравниваются монотонно через
difflib.SequenceMatcher: позиции совпавших слов идут по порядку и назад не
отматываются. Для каждой метки берётся первое сопоставленное слово её строки,
сдвиг = время этого слова в транскрипте − секунды метки. Монотонное
выравнивание нужно потому, что эталон прогонялся через нейросеть для
суммаризации: часть формулировок расходится с произнесённым, и с обеих сторон
есть пропуски и лишние слова.

Почему доля несопоставленных меток — отдельное число, а не ошибка. Нарезчик по
умолчанию выбрасывает паузы длиннее 2 с (tools/chunker.py, --drop-pause),
поэтому в транскрипте есть дыры, и реплика эталона может не иметь соответствия
вообще. Такая метка помечается как несопоставленная и в статистику сдвига не
входит.

Нормализация текста берётся из tools/score.py импортом, а не копированием:
сдвиг и WER обязаны считаться по одному и тому же тексту, иначе два инструмента
разойдутся, и расхождение заметят не сразу. Оттуда же берутся разбор строки
«[ММ:СС] Роль: текст» и удаление пометок в квадратных скобках.

Чего не делает. Не считает WER и CER (это tools/score.py), не трогает модели, не
ходит в сеть и не правит ни эталон, ни транскрипт: единственный файл, который он
пишет, — указанный в --json.

Зависимости: стандартная библиотека плюс то, что уже тянет score.py (num2words —
и только если в тексте есть цифры; jiwer здесь не нужен).

Использование:
  python tools/align_timecodes.py --ref эталон.txt --transcript v3_rnnt.json
  python tools/align_timecodes.py --ref эталон.txt --transcript v3_rnnt.json \\
      --json shift.json --rows 0
  python tools/align_timecodes.py --selftest

Пути с пробелами, скобками и кириллицей поддерживаются — передавайте в кавычках.
"""

from __future__ import annotations

import io
import json
import re
import statistics
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

# score.py лежит рядом, а tools/ — не пакет: при запуске «python tools/…» каталог
# скрипта уже первый в sys.path, но при импорте модуля откуда-то ещё — нет.
# Поэтому путь добавляется явно, иначе нормализацию пришлось бы копировать.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import score  # noqa: E402 — импорт возможен только после правки sys.path выше

PROG = "align_timecodes.py"

# Версия формата машинночитаемого итога.
SCHEMA = "timeshift/1"

# Схема входного транскрипта — та же константа, что у score.py и transcribe.py.
TRANSCRIPT_SCHEMA = score.TRANSCRIPT_SCHEMA

PLAN_REFERENCE = "docs/research/benchmark-plan.md, раздел 7"

# Сколько строк таблицы печатать по умолчанию (0 — все).
DEFAULT_ROWS = 30

# Метки эталона округлены до целой секунды и ставятся на начало реплики, поэтому
# сдвиг меньше этого порога ничего не значит: его не отличить от квантования.
QUANTIZATION_SEC = 2.0

PRECISION_NOTE = (
    "Оговорка о точности: метки эталона округлены до целых секунд и отмечают "
    "начало реплики, а не первое слово, поэтому сдвиг меньше 1–2 с неотличим от "
    "квантования. Значимым считается только систематический сдвиг больше этого "
    f"порога (здесь — {QUANTIZATION_SEC:.0f} с)."
)

# Метка времени в начале строки: [ММ:СС] и [ЧЧ:ММ:СС]. Форма та же, что у
# score.TIMECODE_RE, но с группами — здесь нужны сами числа, а не только факт
# метки. Совпадение форм проверяется селф-тестом.
MARK_RE = re.compile(r"^\[\s*(\d{1,3}):([0-5]\d)(?::([0-5]\d))?\s*\]\s*")


# --------------------------------------------------------------- ошибки ---


class ToolError(Exception):
    """Ошибка окружения или данных: сообщение уже на русском и готово в stderr."""

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


# ------------------------------------------------------- разбор эталона ---


def mark_seconds(match: re.Match) -> float:
    """Секунды из метки: [ММ:СС] — минуты и секунды, [ЧЧ:ММ:СС] — ещё и часы."""
    first, second, third = match.group(1), match.group(2), match.group(3)
    if third is None:
        return float(int(first) * 60 + int(second))
    return float(int(first) * 3600 + int(second) * 60 + int(third))


@dataclass
class RefLine:
    """Непустая строка эталона: её метка и место её слов в общем потоке слов."""

    number: int  # номер непустой строки, с 1 — чтобы человек нашёл её в файле
    mark: str | None  # «[12:34]» как в файле, None — строка без метки
    seconds: float | None
    token_start: int  # индекс первого слова строки в потоке (включительно)
    token_end: int  # индекс за последним словом строки

    @property
    def marked(self) -> bool:
        return self.mark is not None


@dataclass
class Reference:
    """Эталон, разложенный в поток слов с привязкой слов к строкам."""

    lines: list[RefLine] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    notes_removed: int = 0

    @property
    def marked_lines(self) -> list[RefLine]:
        return [line for line in self.lines if line.marked]


def parse_reference(raw: str) -> Reference:
    """Разбирает эталон построчно: метка, ярлык роли, слова после нормализации.

    Строки без метки времени слова в поток отдают (это та же речь, и без них
    выравнивание поехало бы), но своей метки у них нет и в статистику сдвига они
    не попадают. Сам файл эталона не трогается ни при каких условиях: его правка
    после получения метрик обнуляет сравнение (раздел 3 протокола).
    """
    result = Reference()
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        number = len(result.lines) + 1
        mark: str | None = None
        seconds: float | None = None

        stamp = MARK_RE.match(line)
        if stamp:
            mark = re.sub(r"\s+", "", stamp.group(0).strip())
            seconds = mark_seconds(stamp)
            line = line[stamp.end():]
            # Ярлык роли снимается только сразу после метки и только короткий —
            # правило и регулярка те же, что в score.py.
            role = score.ROLE_RE.match(line)
            if role and 1 <= len(role.group(1).split()) <= score.MAX_ROLE_WORDS:
                line = line[role.end():]

        clean, notes = score.strip_notes(line)
        result.notes_removed += notes
        token_start = len(result.tokens)
        result.tokens.extend(score.normalize(clean).split())
        result.lines.append(
            RefLine(
                number=number,
                mark=mark,
                seconds=seconds,
                token_start=token_start,
                token_end=len(result.tokens),
            )
        )
    return result


# --------------------------------------------------- чтение транскрипта ---


@dataclass
class Hypothesis:
    """Транскрипт, разложенный в поток слов с временем начала каждого слова."""

    tokens: list[str] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)
    segments: int = 0
    segments_without_words: int = 0


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{what}: ожидалось число, получено {value!r}", code=2)
    return float(value)


def read_transcript(path: Path) -> Hypothesis:
    """Читает JSON схемы transcript/1 и раскладывает слова в поток.

    Одно слово транскрипта может дать несколько слов потока: нормализация
    разворачивает «2024» в четыре слова. Всем им достаётся время начала того
    слова, из которого они получились, — иначе поток и времена разъехались бы.
    Сегменты без пословных таймкодов в поток не идут: сопоставлять нечем, но их
    число печатается, чтобы несопоставленные метки не выглядели загадкой.
    """
    if not path.exists():
        raise ToolError(f"--transcript: файл не найден: {path}", code=2)
    if path.is_dir():
        raise ToolError(f"--transcript: это каталог, а не файл: {path}", code=2)
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"--transcript: файл не читается как UTF-8: {path}", code=2) from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"--transcript: битый JSON ({exc}): {path}", code=2) from None
    if not isinstance(data, dict):
        raise ToolError(
            f"--transcript: в JSON ожидался объект, а не {type(data).__name__}", code=2
        )
    schema = data.get("schema")
    if schema != TRANSCRIPT_SCHEMA:
        raise ToolError(
            f"--transcript: ожидалась схема {TRANSCRIPT_SCHEMA}, в файле: {schema!r}", code=2
        )
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ToolError("--transcript: поле segments отсутствует или не список", code=2)

    result = Hypothesis(segments=len(segments))
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ToolError(f"--transcript: сегмент {position}: ожидался объект JSON", code=2)
        number = segment.get("index", position)
        words = segment.get("words") or []
        if not isinstance(words, list):
            raise ToolError(f"--transcript: сегмент {number}: words должен быть списком", code=2)
        if not words:
            result.segments_without_words += 1
            continue
        for word_position, word in enumerate(words):
            if not isinstance(word, dict):
                raise ToolError(
                    f"--transcript: сегмент {number}, слово {word_position}: "
                    "ожидался объект JSON",
                    code=2,
                )
            for required in ("text", "start"):
                if required not in word:
                    raise ToolError(
                        f"--transcript: сегмент {number}, слово {word_position}: "
                        f"нет поля {required!r}",
                        code=2,
                    )
            start = _number(
                word["start"], f"--transcript: сегмент {number}, слово {word_position}: start"
            )
            for token in score.normalize(str(word["text"])).split():
                result.tokens.append(token)
                result.starts.append(start)
    return result


# ---------------------------------------------------------- сопоставление ---


def align(ref_tokens: list[str], hyp_tokens: list[str]) -> dict[int, int]:
    """Монотонная раскладка: индекс слова эталона → индекс слова транскрипта.

    SequenceMatcher выдаёт совпавшие блоки строго по возрастанию в обоих
    последовательностях, поэтому порядок слов не нарушается по построению:
    если слово эталона сопоставлено, то все сопоставленные слова после него
    стоят в транскрипте правее. autojunk=False обязателен: на длинной записи
    эвристика «популярных» элементов выбросила бы из сравнения самые частые
    слова («и», «что», «вот») и выравнивание развалилось бы.
    """
    matcher = SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    mapping: dict[int, int] = {}
    for ref_index, hyp_index, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapping[ref_index + offset] = hyp_index + offset
    return mapping


@dataclass
class LineShift:
    """Сдвиг по одной метке эталона."""

    mark: str
    mark_seconds: float
    hyp_seconds: float | None
    shift: float | None
    matched: bool

    def as_dict(self) -> dict:
        return {
            "mark": self.mark,
            "mark_seconds": round(self.mark_seconds, 3),
            "hyp_seconds": None if self.hyp_seconds is None else round(self.hyp_seconds, 3),
            "shift": None if self.shift is None else round(self.shift, 3),
            "matched": self.matched,
        }


def percentile(values: list[float], q: float) -> float | None:
    """Процентиль с линейной интерполяцией (как numpy по умолчанию).

    Своя реализация, а не statistics.quantiles: тому нужно не меньше двух
    значений, а меток в коротком фрагменте может оказаться одна.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass
class Shift:
    """Всё, что инструмент насчитал по паре «эталон — транскрипт»."""

    lines: list[LineShift] = field(default_factory=list)
    ref_words: int = 0
    hyp_words: int = 0
    aligned_words: int = 0
    segments: int = 0
    segments_without_words: int = 0
    notes_removed: int = 0

    @property
    def marks(self) -> int:
        return len(self.lines)

    @property
    def shifts(self) -> list[float]:
        return [line.shift for line in self.lines if line.shift is not None]

    @property
    def matched(self) -> int:
        return len(self.shifts)

    @property
    def unmatched(self) -> int:
        return self.marks - self.matched

    @property
    def unmatched_share(self) -> float:
        return self.unmatched / self.marks if self.marks else 0.0

    @property
    def median_shift(self) -> float | None:
        return statistics.median(self.shifts) if self.shifts else None

    @property
    def max_abs_shift(self) -> float | None:
        return max((abs(value) for value in self.shifts), default=None)

    @property
    def p90_shift(self) -> float | None:
        return percentile(self.shifts, 0.9)

    @property
    def p90_abs_shift(self) -> float | None:
        return percentile([abs(value) for value in self.shifts], 0.9)

    @property
    def worst_line(self) -> LineShift | None:
        """Метка с наибольшим |сдвигом| — с неё человек и начнёт разбираться."""
        matched = [line for line in self.lines if line.shift is not None]
        if not matched:
            return None
        return max(matched, key=lambda line: abs(line.shift))

    @property
    def direction(self) -> str:
        """Знак медианы словами: кто из двух отстаёт."""
        median = self.median_shift
        if median is None:
            return "неизвестно: сопоставленных меток нет"
        if median > 0:
            return "транскрипт отстаёт от эталона"
        if median < 0:
            return "транскрипт опережает эталон"
        return "сдвига нет"

    @property
    def within_quantization(self) -> bool:
        median = self.median_shift
        return median is not None and abs(median) < QUANTIZATION_SEC


def compute(reference: Reference, hypothesis: Hypothesis) -> Shift:
    """Считает сдвиг по каждой метке эталона."""
    mapping = align(reference.tokens, hypothesis.tokens)
    result = Shift(
        ref_words=len(reference.tokens),
        hyp_words=len(hypothesis.tokens),
        aligned_words=len(mapping),
        segments=hypothesis.segments,
        segments_without_words=hypothesis.segments_without_words,
        notes_removed=reference.notes_removed,
    )
    for line in reference.marked_lines:
        assert line.mark is not None and line.seconds is not None  # marked_lines
        hyp_seconds: float | None = None
        # Первое сопоставленное слово строки: именно оно отмечает её начало,
        # дальше по строке время уже уехало на длину реплики.
        for index in range(line.token_start, line.token_end):
            if index in mapping:
                hyp_seconds = hypothesis.starts[mapping[index]]
                break
        result.lines.append(
            LineShift(
                mark=line.mark,
                mark_seconds=line.seconds,
                hyp_seconds=hyp_seconds,
                shift=None if hyp_seconds is None else hyp_seconds - line.seconds,
                matched=hyp_seconds is not None,
            )
        )
    return result


# -------------------------------------------------------------- вывод ---


def fmt_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} с"


def fmt_shift(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f} с"


def print_report(shift: Shift, ref_path: Path, transcript_path: Path, rows: int) -> None:
    print("Сдвиг таймкодов относительно меток эталона")
    print(f"  эталон:     {ref_path}")
    print(f"  транскрипт: {transcript_path}")
    print(
        f"  слов после нормализации: эталон {shift.ref_words}, транскрипт "
        f"{shift.hyp_words}, сопоставлено позиций {shift.aligned_words}"
    )
    print(
        f"  сегментов: {shift.segments}"
        f" (без пословных таймкодов: {shift.segments_without_words}),"
        f" пометок в скобках удалено: {shift.notes_removed}"
    )
    print()

    print(f"меток в эталоне:        {shift.marks}")
    print(f"сопоставлено:           {shift.matched}")
    print(
        f"не сопоставлено:        {shift.unmatched}"
        f" ({shift.unmatched_share * 100:.1f} % меток)"
    )
    print(f"медиана сдвига:         {fmt_shift(shift.median_shift)} — {shift.direction}")
    worst = shift.worst_line
    worst_mark = f" (метка {worst.mark})" if worst is not None else ""
    print(f"максимум |сдвига|:      {fmt_seconds(shift.max_abs_shift)}{worst_mark}")
    print(f"90-й процентиль сдвига: {fmt_shift(shift.p90_shift)}")
    print(f"90-й процентиль |сдвига|: {fmt_seconds(shift.p90_abs_shift)}")
    if shift.within_quantization:
        print(
            f"медиана меньше {QUANTIZATION_SEC:.0f} с: систематического сдвига не видно, "
            "это уровень квантования меток"
        )
    print()

    if shift.lines:
        shown = shift.lines if rows == 0 else shift.lines[:rows]
        print(f"по строкам: показано {len(shown)} из {shift.marks}")
        print(f"  {'метка':<10}{'эталон, с':>12}{'транскрипт, с':>16}{'сдвиг, с':>12}")
        for line in shown:
            if line.matched:
                print(
                    f"  {line.mark:<10}{line.mark_seconds:>12.2f}"
                    f"{line.hyp_seconds:>16.2f}{line.shift:>+12.2f}"
                )
            else:
                print(
                    f"  {line.mark:<10}{line.mark_seconds:>12.2f}{'—':>16}{'—':>12}"
                    "  не сопоставлено"
                )
    else:
        print("по строкам: меток нет")

    print()
    print(PRECISION_NOTE)
    if shift.unmatched:
        print(
            "Несопоставленные метки — ожидаемое явление, а не сбой: нарезчик по "
            "умолчанию выбрасывает паузы длиннее 2 с (tools/chunker.py, "
            "--drop-pause), и части реплик эталона в транскрипте просто нет."
        )


def build_machine_report(shift: Shift, ref_path: Path, transcript_path: Path) -> dict:
    """Итог схемы timeshift/1. Все времена — секунды от начала записи."""

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 3)

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "units": "seconds",
        "reference": PLAN_REFERENCE,
        "ref": str(ref_path),
        "transcript": str(transcript_path),
        "marks": shift.marks,
        "matched": shift.matched,
        "unmatched": shift.unmatched,
        "unmatched_share": round(shift.unmatched_share, 6),
        "median_shift": rounded(shift.median_shift),
        "max_abs_shift": rounded(shift.max_abs_shift),
        "p90_shift": rounded(shift.p90_shift),
        "p90_abs_shift": rounded(shift.p90_abs_shift),
        "direction": shift.direction,
        "within_quantization": shift.within_quantization,
        "quantization_seconds": QUANTIZATION_SEC,
        "words": {
            "ref": shift.ref_words,
            "hyp": shift.hyp_words,
            "aligned": shift.aligned_words,
        },
        "segments": {
            "total": shift.segments,
            "without_words": shift.segments_without_words,
        },
        "note": PRECISION_NOTE,
        "lines": [line.as_dict() for line in shift.lines],
    }


def write_machine_report(report: dict, target: str) -> None:
    """Пишет итог в файл или в stdout по «-»."""
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if target == "-":
        sys.stdout.write(text)
        return
    path = Path(target)
    if path.parent and not path.parent.exists():
        raise ToolError(f"--json: каталога нет: {path.parent}", code=2)
    path.write_text(text, encoding="utf-8")
    print(f"машинночитаемый итог: {path}")


# ----------------------------------------------------------------- прогон ---


@dataclass
class Options:
    ref: Path | None = None
    transcript: Path | None = None
    json_path: str | None = None
    rows: int = DEFAULT_ROWS
    selftest: bool = False
    help: bool = False


def read_reference(path: Path) -> Reference:
    if not path.exists():
        raise ToolError(f"--ref: файл не найден: {path}", code=2)
    if path.is_dir():
        raise ToolError(f"--ref: это каталог, а не файл: {path}", code=2)
    try:
        # utf-8-sig, а не utf-8: эталон приходит из редактора, и BOM в начале
        # файла съел бы метку первой строки — она перестала бы быть меткой, а
        # это как раз первая реплика записи.
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ToolError(f"--ref: файл не читается как UTF-8: {path}", code=2) from None
    return parse_reference(raw)


def run(opts: Options) -> int:
    assert opts.ref is not None and opts.transcript is not None  # проверено в main()

    reference = read_reference(opts.ref)
    hypothesis = read_transcript(opts.transcript)
    if not reference.marked_lines:
        raise ToolError(
            "в эталоне нет ни одной метки [ММ:СС]: сопоставлять нечего "
            "(проверьте --ref: не тот файл или метки в другом формате)",
            code=1,
        )

    shift = compute(reference, hypothesis)
    print_report(shift, opts.ref, opts.transcript, opts.rows)
    if opts.json_path:
        report = build_machine_report(shift, opts.ref, opts.transcript)
        write_machine_report(report, opts.json_path)

    if shift.matched == 0:
        # Сдвиг посчитать не из чего: либо транскрипт пуст, либо тексты не
        # пересекаются вовсе. Доля несопоставленных уже напечатана.
        warn(
            "ни одна метка не сопоставлена: статистики сдвига нет "
            f"(слов в транскрипте: {shift.hyp_words})"
        )
        return 1
    return 0


# ---------------------------------------------------------------- CLI ---


USAGE = f"""Использование: python tools/align_timecodes.py --ref <эталон> \
--transcript <transcript.json> [опции]

Считает, на сколько секунд расходится время реплики в эталоне и время тех же
слов в транскрипте: медиана сдвига, максимум по модулю, 90-й процентиль, доля
несопоставленных меток и таблица по строкам. Слова эталона и транскрипта
выравниваются монотонно (difflib.SequenceMatcher) по тексту, нормализованному
той же функцией, что и в tools/score.py; для каждой метки берётся первое
сопоставленное слово её строки.

Опции:
  --ref <путь>         эталонная расшифровка .txt со строками «[ММ:СС] Роль:
                       текст» (обязателен, кроме --selftest). Поддерживаются и
                       метки [ЧЧ:ММ:СС]; ярлык роли снимается только сразу
                       после метки и только если он не длиннее трёх слов
  --transcript <путь>  JSON схемы {TRANSCRIPT_SCHEMA} от tools/transcribe.py
                       (обязателен, кроме --selftest). Берутся пословные
                       таймкоды segments[].words[]
  --json <путь>        куда писать машинночитаемый итог схемы {SCHEMA}
                       («-» — в stdout)
  --rows <N>           сколько строк таблицы печатать (по умолчанию
                       {DEFAULT_ROWS}, 0 — все). В --json всегда все
  --selftest           прогнать встроенные проверки на синтетике и выйти
  -h, --help           показать эту справку

Знак сдвига: сдвиг = время слова в транскрипте − секунды метки. Положительная
медиана означает, что транскрипт отстаёт от эталона, отрицательная — что
опережает.

{PRECISION_NOTE}

Несопоставленные метки нормальны: нарезчик по умолчанию выбрасывает паузы
длиннее 2 с (tools/chunker.py, --drop-pause), поэтому часть реплик эталона в
транскрипте отсутствует. Такие метки в статистику сдвига не входят, их доля
печатается отдельным числом.

Коды возврата: 0 — сдвиг посчитан, 1 — считать не из чего (в эталоне нет меток,
пустой транскрипт, ни одна метка не сопоставлена), 2 — вход непригоден (файла
нет, битый JSON, чужая схема, ошибка в аргументах).

Инструмент ничего не правит: ни эталон, ни транскрипт. Единственный файл,
который он пишет, — указанный в --json.

Примеры:
  python tools/align_timecodes.py --ref "эталон.txt" \\
      --transcript data/transcripts/v3_rnnt.json
  python tools/align_timecodes.py --ref эталон.txt --transcript v3_rnnt.json \\
      --rows 0 --json -
  python tools/align_timecodes.py --selftest
"""


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
        if arg == "--ref":
            opts.ref = Path(_need_value(argv, index, arg))
            index += 2
            continue
        if arg == "--transcript":
            opts.transcript = Path(_need_value(argv, index, arg))
            index += 2
            continue
        if arg == "--json":
            opts.json_path = _need_value(argv, index, arg)
            index += 2
            continue
        if arg == "--rows":
            raw = _need_value(argv, index, arg)
            try:
                opts.rows = int(raw)
            except ValueError:
                raise UsageError(f"--rows ожидает целое число, получено: {raw}") from None
            if opts.rows < 0:
                raise UsageError(f"--rows не может быть отрицательным, получено: {raw}")
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
        if opts.ref is None:
            raise UsageError("не указан --ref (см. --help)")
        if opts.transcript is None:
            raise UsageError("не указан --transcript (см. --help)")
        return run(opts)
    except ToolError as exc:
        return die(str(exc), exc.code)
    except KeyboardInterrupt:
        return die("прервано пользователем")


# ------------------------------------------------------------ селф-тест ---
#
# Фикстуры — только синтетические слова: ни строчки из реальной записи здесь
# быть не должно, файл лежит в git, а запись — нет. Цифр в фикстурах тоже нет:
# тогда нормализация обходится без num2words, и селф-тест проходит даже на
# неполном окружении.

# Три реплики по три слова: так видно, что берётся именно первое слово строки.
REF_LINES = (
    ("[00:00]", "Менеджер", "альфа бета гамма"),
    ("[00:10]", "Клиент", "дельта эпсилон дзета"),
    ("[00:20]", "Менеджер", "эта тета йота"),
)


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Прогон main() с перехватом вывода: возвращает (код возврата, stdout).

    stderr тоже перехватывается и отбрасывается: часть фикстур намеренно
    непригодна, и её сообщения не должны мешать читать селф-тест.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
        code = main(argv)
    return code, buffer.getvalue()


def _fixture(directory: str, name: str, content: str) -> Path:
    path = Path(directory) / name
    path.write_text(content, encoding="utf-8")
    return path


def _ref_text(lines=REF_LINES) -> str:
    return "".join(f"{mark} {role}: {text}\n" for mark, role, text in lines)


def _segment(index: int, words: list[tuple[str, float]]) -> dict:
    """Сегмент transcript/1 из пар (слово, время начала)."""
    return {
        "index": index,
        "start": words[0][1],
        "end": round(words[-1][1] + 0.4, 3),
        "text": " ".join(text for text, _ in words),
        "words": [
            {"text": text, "start": start, "end": round(start + 0.4, 3)}
            for text, start in words
        ],
    }


def _transcript(segments: list[dict], source_duration: float = 60.0) -> dict:
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "source": "синтетика.wav",
        "source_duration": source_duration,
        "text": " ".join(segment["text"] for segment in segments),
        "segments": segments,
    }


def _shifted_transcript(starts: tuple[float, ...]) -> dict:
    """Транскрипт тех же трёх реплик, каждая начинается в заданное время."""
    segments = []
    for number, (start, (_, _, text)) in enumerate(zip(starts, REF_LINES)):
        words = [(word, round(start + 0.5 * position, 3))
                 for position, word in enumerate(text.split())]
        segments.append(_segment(number, words))
    return _transcript(segments)


def _check_known_shifts():
    """Заранее известные сдвиги +1 / +3 / +5: медиана 3, максимум 5, p90 4,6."""
    reference = parse_reference(_ref_text())
    # Метки 0 / 10 / 20, первые слова реплик в транскрипте — 1 / 13 / 25.
    hypothesis_data = _shifted_transcript((1.0, 13.0, 25.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = _fixture(tmp, "hyp.json", json.dumps(hypothesis_data, ensure_ascii=False))
        hypothesis = read_transcript(path)
    shift = compute(reference, hypothesis)
    assert shift.marks == 3, shift.marks
    assert shift.matched == 3, shift.matched
    assert shift.unmatched == 0, shift.unmatched
    assert [round(value, 3) for value in shift.shifts] == [1.0, 3.0, 5.0], shift.shifts
    assert shift.median_shift == 3.0, shift.median_shift
    assert shift.max_abs_shift == 5.0, shift.max_abs_shift
    assert abs(shift.p90_shift - 4.6) < 1e-9, shift.p90_shift
    assert abs(shift.p90_abs_shift - 4.6) < 1e-9, shift.p90_abs_shift
    assert shift.direction == "транскрипт отстаёт от эталона", shift.direction
    assert not shift.within_quantization, shift.median_shift
    assert shift.worst_line.mark == "[00:20]", shift.worst_line


def _check_known_shifts_cli():
    """Те же сдвиги через CLI: код 0 и все обязательные строки вывода."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", _ref_text())
        hyp = _fixture(
            tmp, "hyp.json", json.dumps(_shifted_transcript((1.0, 13.0, 25.0)))
        )
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp)])
        assert code == 0, (code, out)
        for fragment in (
            "меток в эталоне:        3",
            "сопоставлено:           3",
            "не сопоставлено:        0 (0.0 % меток)",
            "медиана сдвига:         +3.00 с — транскрипт отстаёт от эталона",
            "максимум |сдвига|:      5.00 с (метка [00:20])",
            "90-й процентиль сдвига: +4.60 с",
            "90-й процентиль |сдвига|: 4.60 с",
            "по строкам: показано 3 из 3",
            "Оговорка о точности:",
            "неотличим от квантования",
        ):
            assert fragment in out, (fragment, out)
        # Таблица: метка, секунды эталона, секунды транскрипта, сдвиг.
        assert "[00:10]          10.00           13.00       +3.00" in out, out
        # Ни эталон, ни транскрипт не изменились, лишних файлов не появилось.
        assert ref.read_text(encoding="utf-8") == _ref_text()
        assert json.loads(hyp.read_text(encoding="utf-8")) == _shifted_transcript(
            (1.0, 13.0, 25.0)
        )
        assert sorted(item.name for item in Path(tmp).iterdir()) == ["hyp.json", "ref.txt"]


def _check_negative_shift():
    """Отрицательная медиана: транскрипт опережает эталон."""
    reference = parse_reference(_ref_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = _fixture(tmp, "hyp.json", json.dumps(_shifted_transcript((0.0, 5.0, 14.0))))
        hypothesis = read_transcript(path)
    shift = compute(reference, hypothesis)
    assert [round(value, 3) for value in shift.shifts] == [0.0, -5.0, -6.0], shift.shifts
    assert shift.median_shift == -5.0, shift.median_shift
    assert shift.max_abs_shift == 6.0, shift.max_abs_shift
    assert shift.direction == "транскрипт опережает эталон", shift.direction


def _check_unmatched_line():
    """Строка, слов которой в транскрипте нет, помечается несопоставленной."""
    ref_lines = REF_LINES + (("[00:30]", "Клиент", "каппа лямбда мю"),)
    reference = parse_reference(_ref_text(ref_lines))
    with tempfile.TemporaryDirectory() as tmp:
        # В транскрипте только три первых реплики: четвёртой нет вообще —
        # ровно то, что бывает после выброшенной паузы.
        hyp = _fixture(tmp, "hyp.json", json.dumps(_shifted_transcript((1.0, 11.0, 21.0))))
        hypothesis = read_transcript(hyp)
        shift = compute(reference, hypothesis)
        assert shift.marks == 4, shift.marks
        assert shift.matched == 3, shift.matched
        assert shift.unmatched == 1, shift.unmatched
        assert abs(shift.unmatched_share - 0.25) < 1e-9, shift.unmatched_share
        last = shift.lines[-1]
        assert last.mark == "[00:30]" and not last.matched, last
        assert last.hyp_seconds is None and last.shift is None, last
        # Несопоставленная метка в статистику сдвига не входит.
        assert shift.median_shift == 1.0, shift.median_shift
        assert shift.max_abs_shift == 1.0, shift.max_abs_shift

        ref = _fixture(tmp, "ref.txt", _ref_text(ref_lines))
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp)])
        assert code == 0, (code, out)
        assert "не сопоставлено:        1 (25.0 % меток)" in out, out
        assert "не сопоставлено" in out and "[00:30]" in out, out
        assert "Несопоставленные метки — ожидаемое явление" in out, out
        # Медиана 1 с меньше порога квантования — об этом сказано отдельно.
        assert "систематического сдвига не видно" in out, out


def _check_empty_transcript():
    """Пустой транскрипт: все метки несопоставлены, код 1, без деления на ноль."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = _transcript([{"index": 0, "start": 0.0, "end": 1.0, "text": "", "words": []}])
        ref = _fixture(tmp, "ref.txt", _ref_text())
        hyp = _fixture(tmp, "hyp.json", json.dumps(empty))
        hypothesis = read_transcript(hyp)
        assert hypothesis.tokens == [], hypothesis.tokens
        assert hypothesis.segments_without_words == 1, hypothesis

        shift = compute(parse_reference(_ref_text()), hypothesis)
        assert shift.matched == 0 and shift.unmatched == 3, shift
        assert shift.unmatched_share == 1.0, shift.unmatched_share
        assert shift.median_shift is None and shift.max_abs_shift is None, shift
        assert shift.p90_shift is None and shift.p90_abs_shift is None, shift
        assert shift.direction == "неизвестно: сопоставленных меток нет", shift.direction

        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp)])
        assert code == 1, (code, out)
        assert "сопоставлено:           0" in out, out
        assert "не сопоставлено:        3 (100.0 % меток)" in out, out
        assert "медиана сдвига:         — — неизвестно" in out, out
        assert "без пословных таймкодов: 1" in out, out

        # Совсем без сегментов — тот же исход, код 1.
        bare = _fixture(tmp, "bare.json", json.dumps(_transcript([])))
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(bare)])
        assert code == 1, (code, out)


def _check_monotonic_alignment():
    """Раскладка монотонная: перестановка слов не даёт сопоставления назад."""
    ref_tokens = "альфа бета гамма дельта эпсилон дзета".split()
    # Транскрипт «переставил» блоки: вторая половина оказалась перед первой.
    hyp_tokens = "дельта эпсилон дзета альфа бета гамма".split()
    mapping = align(ref_tokens, hyp_tokens)
    keys = sorted(mapping)
    values = [mapping[key] for key in keys]
    assert values == sorted(values), mapping
    assert len(set(values)) == len(values), mapping
    # Сопоставился только один из двух блоков — второй пришлось бы брать назад.
    assert len(mapping) == 3, mapping

    # На обычном тексте раскладка попадает в те же позиции.
    mapping = align(ref_tokens, ["шум"] + ref_tokens)
    assert mapping == {index: index + 1 for index in range(len(ref_tokens))}, mapping

    # Частые слова из сравнения не выбрасываются: autojunk=False.
    filler = ["и"] * 300
    mapping = align(filler + ["альфа"], filler + ["альфа"])
    assert len(mapping) == 301, len(mapping)


def _check_reference_parsing():
    """Метки [ММ:СС] и [ЧЧ:ММ:СС], ярлыки ролей, строки без метки."""
    reference = parse_reference(
        "[00:05] Менеджер: альфа бета\n"
        "\n"
        "[01:30] Клиент два: гамма\n"
        "[01:02:03] Голос из зала: дельта\n"
        "[02:00] А теперь смотрите внимательно: эпсилон\n"
        "продолжение без метки\n"
    )
    assert len(reference.lines) == 5, reference.lines
    marked = reference.marked_lines
    assert [line.mark for line in marked] == [
        "[00:05]",
        "[01:30]",
        "[01:02:03]",
        "[02:00]",
    ], marked
    # Секунды всегда float: в JSON у mark_seconds должен быть один тип.
    assert [line.seconds for line in marked] == [5.0, 90.0, 3723.0, 120.0], marked
    assert all(isinstance(line.seconds, float) for line in marked), marked
    # Ярлык роли снят только там, где он короткий; «А теперь смотрите
    # внимательно:» — четыре слова, это уже речь, её слова остаются.
    assert reference.tokens == [
        "альфа", "бета", "гамма", "дельта",
        "а", "теперь", "смотрите", "внимательно", "эпсилон",
        "продолжение", "без", "метки",
    ], reference.tokens
    # Строка без метки слова в поток отдаёт, но своей метки у неё нет.
    assert reference.lines[-1].mark is None, reference.lines[-1]
    assert reference.lines[-1].token_start == 9, reference.lines[-1]
    # Форма метки здесь и в score.py совпадает: обе регулярки видят одно и то же.
    for line in ("[00:05] раз", "[01:02:03] два", "[123:45] три"):
        assert bool(MARK_RE.match(line)) == bool(score.TIMECODE_RE.match(line)), line
    for line in ("[1:99] четыре", "без метки", "[00:05 пять"):
        assert bool(MARK_RE.match(line)) == bool(score.TIMECODE_RE.match(line)), line


def _check_normalization_is_shared():
    """Нормализация — та самая, что в score.py: регистр, ё, пунктуация, дефис."""
    # Импорт, а не копия: один и тот же объект функции.
    assert score.normalize("Ещё раз, по-русски!") == "еще раз по русски"
    # Поэтому расхождение в регистре и ё сопоставлению не мешает.
    reference = parse_reference("[00:10] Менеджер: Ещё раз — по-русски!\n")
    assert reference.tokens == ["еще", "раз", "по", "русски"], reference.tokens
    with tempfile.TemporaryDirectory() as tmp:
        words = [("ЕЩЕ,", 12.0), ("раз", 12.5), ("ПО-РУССКИ!", 13.0)]
        hyp = _fixture(tmp, "hyp.json", json.dumps(_transcript([_segment(0, words)])))
        hypothesis = read_transcript(hyp)
        # «ПО-РУССКИ!» нормализация разворачивает в два слова, и обоим достаётся
        # время исходного слова — иначе поток и времена разъехались бы.
        assert hypothesis.tokens == ["еще", "раз", "по", "русски"], hypothesis.tokens
        assert hypothesis.starts == [12.0, 12.5, 13.0, 13.0], hypothesis.starts
        shift = compute(reference, hypothesis)
        assert shift.matched == 1 and shift.shifts == [2.0], shift


def _check_bom_reference():
    """BOM в начале эталона не съедает метку первой строки.

    Проверка не умозрительная: настоящий эталон записи сохранён с BOM, и без
    utf-8-sig первая реплика теряла метку (меток было 109 вместо 110), а ярлык
    роли первой строки уходил в поток слов.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", "﻿" + _ref_text())
        reference = read_reference(ref)
        assert len(reference.marked_lines) == 3, reference.marked_lines
        assert reference.marked_lines[0].mark == "[00:00]", reference.marked_lines[0]
        # Ярлык роли первой строки снят, значит метка была распознана.
        assert reference.tokens[0] == "альфа", reference.tokens[:3]

        hyp = _fixture(tmp, "hyp.json", json.dumps(_shifted_transcript((1.0, 13.0, 25.0))))
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp)])
        assert code == 0, (code, out)
        assert "меток в эталоне:        3" in out, out


def _check_percentile():
    """Процентиль считается линейной интерполяцией и переживает один элемент."""
    assert percentile([], 0.9) is None
    assert percentile([7.0], 0.9) == 7.0
    assert percentile([1.0, 2.0], 0.5) == 1.5
    assert abs(percentile([1.0, 3.0, 5.0], 0.9) - 4.6) < 1e-9
    assert percentile([5.0, 1.0, 3.0], 0.0) == 1.0
    assert percentile([5.0, 1.0, 3.0], 1.0) == 5.0
    assert abs(percentile([float(n) for n in range(1, 11)], 0.9) - 9.1) < 1e-9


def _check_machine_report():
    """--json пишет объект схемы timeshift/1 со всеми обязательными полями."""
    ref_lines = REF_LINES + (("[00:30]", "Клиент", "каппа лямбда мю"),)
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", _ref_text(ref_lines))
        hyp = _fixture(tmp, "hyp.json", json.dumps(_shifted_transcript((1.0, 13.0, 25.0))))
        target = Path(tmp) / "shift.json"
        code, out = _run_cli(
            ["--ref", str(ref), "--transcript", str(hyp), "--json", str(target)]
        )
        assert code == 0, (code, out)
        report = json.loads(target.read_text(encoding="utf-8"))
        assert report["schema"] == SCHEMA, report["schema"]
        for key in (
            "marks",
            "matched",
            "unmatched",
            "median_shift",
            "max_abs_shift",
            "p90_shift",
            "lines",
        ):
            assert key in report, key
        assert (report["marks"], report["matched"], report["unmatched"]) == (4, 3, 1), report
        assert report["median_shift"] == 3.0, report["median_shift"]
        assert report["max_abs_shift"] == 5.0, report["max_abs_shift"]
        assert report["p90_shift"] == 4.6, report["p90_shift"]
        assert abs(report["unmatched_share"] - 0.25) < 1e-9, report["unmatched_share"]
        assert report["direction"] == "транскрипт отстаёт от эталона", report["direction"]
        # Строк в JSON всегда все, а не только напечатанные.
        assert len(report["lines"]) == 4, report["lines"]
        assert set(report["lines"][0]) == {
            "mark",
            "mark_seconds",
            "hyp_seconds",
            "shift",
            "matched",
        }, report["lines"][0]
        assert report["lines"][0] == {
            "mark": "[00:00]",
            "mark_seconds": 0.0,
            "hyp_seconds": 1.0,
            "shift": 1.0,
            "matched": True,
        }, report["lines"][0]
        assert report["lines"][-1] == {
            "mark": "[00:30]",
            "mark_seconds": 30.0,
            "hyp_seconds": None,
            "shift": None,
            "matched": False,
        }, report["lines"][-1]
        # Метки в итоге идут в том же порядке, что в эталоне.
        seconds = [line["mark_seconds"] for line in report["lines"]]
        assert seconds == sorted(seconds), seconds
        # Оговорка о точности есть и в машинном итоге.
        assert "квантования" in report["note"], report["note"]

        # «-» отправляет тот же объект в stdout и файла не создаёт.
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp), "--json", "-"])
        assert code == 0, (code, out)
        assert f'"schema": "{SCHEMA}"' in out, out


def _table_marks(out: str) -> list[str]:
    """Метки из строк таблицы вывода — только из них, не из сводки."""
    return [
        line.split()[0]
        for line in out.splitlines()
        if re.match(r"^  \[\d", line)
    ]


def _check_rows_flag():
    """--rows режет таблицу, но не список меток; 0 — печатать все."""
    many = tuple(
        (f"[00:{number:02d}]", "Менеджер", f"слово{chr(1072 + number)} повтор{number}")
        for number in range(6)
    )
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", _ref_text(many))
        words = [
            (word, float(number * 10 + position))
            for number, (_, _, text) in enumerate(many)
            for position, word in enumerate(text.split())
        ]
        hyp = _fixture(tmp, "hyp.json", json.dumps(_transcript([_segment(0, words)], 120.0)))
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp), "--rows", "2"])
        assert code == 0, (code, out)
        assert "по строкам: показано 2 из 6" in out, out
        # Считаем именно строки таблицы: метка «[00:05]» встречается и в строке
        # максимума, поэтому искать её по всему выводу нельзя.
        assert _table_marks(out) == ["[00:00]", "[00:01]"], _table_marks(out)
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp), "--rows", "0"])
        assert code == 0, (code, out)
        assert "по строкам: показано 6 из 6" in out, out
        assert _table_marks(out) == [f"[00:{number:02d}]" for number in range(6)], out
        # По умолчанию строк больше, чем здесь меток, — таблица целиком.
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp)])
        assert code == 0, (code, out)
        assert len(_table_marks(out)) == 6, out


def _check_no_marks():
    """Эталон без меток времени — считать нечего, код 1."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", "альфа бета гамма\nдельта эпсилон\n")
        hyp = _fixture(tmp, "hyp.json", json.dumps(_shifted_transcript((1.0, 13.0, 25.0))))
        code, out = _run_cli(["--ref", str(ref), "--transcript", str(hyp)])
        assert code == 1, (code, out)
        assert "медиана" not in out, out


def _check_bad_input():
    """Ошибки в аргументах и непригодный вход дают код 2, --help — код 0."""
    code, out = _run_cli(["--help"])
    assert code == 0, code
    for flag in ("--ref", "--transcript", "--json", "--rows", "--selftest", "-h, --help"):
        assert flag in out, flag
    assert "Использование:" in out, out
    assert "квантования" in out, out

    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", _ref_text())
        good = json.dumps(_shifted_transcript((1.0, 13.0, 25.0)))
        alien = _fixture(tmp, "alien.json", json.dumps({"schema": "chunk-plan/1"}))
        broken = _fixture(tmp, "broken.json", "{не json")
        no_segments = _fixture(tmp, "no_segments.json", json.dumps({"schema": TRANSCRIPT_SCHEMA}))
        bad_word = _transcript([_segment(0, [("альфа", 0.0)])])
        bad_word["segments"][0]["words"][0].pop("start")
        no_start = _fixture(tmp, "no_start.json", json.dumps(bad_word))
        hyp = _fixture(tmp, "hyp.json", good)

        for argv in (
            [],
            ["--ref", str(ref)],
            ["--transcript", str(hyp)],
            ["--ref", "нет.txt", "--transcript", str(hyp)],
            ["--ref", str(ref), "--transcript", "нет.json"],
            ["--ref", str(ref), "--transcript", str(alien)],
            ["--ref", str(ref), "--transcript", str(broken)],
            ["--ref", str(ref), "--transcript", str(no_segments)],
            ["--ref", str(ref), "--transcript", str(no_start)],
            ["--ref", str(ref), "--transcript", str(hyp), "--json", "нет/такого/shift.json"],
            ["--ref"],
            ["--ref", str(ref), "--transcript", str(hyp), "--rows", "много"],
            ["--ref", str(ref), "--transcript", str(hyp), "--rows", "-1"],
            ["--чепуха"],
        ):
            code, _ = _run_cli(argv)
            assert code == 2, (argv, code)


CHECKS = (
    ("известные сдвиги +1 / +3 / +5: медиана, максимум, p90", _check_known_shifts),
    ("те же сдвиги через CLI, входы не изменены", _check_known_shifts_cli),
    ("отрицательная медиана: транскрипт опережает", _check_negative_shift),
    ("несопоставимая строка: доля и исключение из статистики", _check_unmatched_line),
    ("пустой транскрипт: 100 % несопоставленных, код 1", _check_empty_transcript),
    ("раскладка монотонная, autojunk выключен", _check_monotonic_alignment),
    ("разбор меток [ММ:СС] и [ЧЧ:ММ:СС], ярлыки ролей", _check_reference_parsing),
    ("нормализация взята из score.py импортом", _check_normalization_is_shared),
    ("BOM в эталоне не съедает первую метку", _check_bom_reference),
    ("процентиль с линейной интерполяцией", _check_percentile),
    ("--json пишет итог схемы timeshift/1", _check_machine_report),
    ("--rows режет таблицу, 0 — все строки", _check_rows_flag),
    ("эталон без меток — код 1", _check_no_marks),
    ("аргументы и непригодный вход дают код 2", _check_bad_input),
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
