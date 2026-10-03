#!/usr/bin/env python3
"""
chunker.py — строит план нарезки 16 кГц моно WAV на чанки под GigaAM.

Зачем это нужно. gigaam 0.2.0 в transcribe() сравнивает длину входа с
LONGFORM_THRESHOLD = 25 * 16000 сэмплов и на более длинном входе бросает
ValueError, то есть запись длиннее 25 секунд модели не отдать. Штатный
transcribe_longform() из gigaam нам не подходит: он тянет pyannote и требует
HF_TOKEN. Поэтому режем сами — по тишине, одним проходом ffmpeg silencedetect,
без сети, без токенов и без моделей.

Как режем. Точки разреза — центры интервалов тишины: в центре паузы шансы
разорвать слово минимальны. Паузы длиннее --drop-pause выбрасываются целиком
(предыдущий чанк заканчивается на silence_start + pad, следующий начинается на
silence_end - pad). Это осознанное решение владельца: так быстрее, а в SRT на
месте длинных пауз будут дыры. Короткие паузы остаются внутри звука и работают
только как кандидаты на разрез. Если на отрезке подходящей тишины нет (сплошная
речь), разрез делается жёстко по --max-chunk — это заметно в плане по
cut_reason "hard_max".

Почему два лимита. --max-chunk (22 с) — рабочая длина, к которой стремимся;
--hard-max-chunk (24 с) — абсолютный предел с запасом к 25 с модели, который не
имеет права нарушить ни один чанк, в том числе после присоединения короткого
хвоста к предыдущему чанку.

Что на выходе. plan.json с параметрами прогона, найденной тишиной, чанками и
выброшенными паузами (формат описан в schema "chunk-plan/1"), а без --dry-run —
ещё и сами чанки в подкаталоге chunks/ рядом с планом, каждый отдельным вызовом
ffmpeg в 16 кГц моно pcm_s16le. План и чанки — единственный способ обмена с
соседними инструментами: ничего друг у друга они не импортируют.

Значения start/end в dropped — это фактически выброшенный участок записи
(пауза, уже уменьшенная на pad с обеих сторон), поэтому
coverage_seconds + dropped_seconds = source_duration: по плану всегда видно,
что ни одна секунда записи не потерялась молча.

Зависимости: только стандартная библиотека Python (ни torch, ни torchaudio, ни
gigaam здесь не импортируются) плюс ffmpeg/ffprobe в PATH для реального прогона.
Логика планирования — чистая функция plan_chunks(): она не вызывает ffmpeg и не
читает файлы, поэтому её проверяет встроенный --selftest на синтетических
списках тишины, без аудио и без ffmpeg вообще.

Использование:
  python tools/chunker.py --input <wav> [--outdir data/transcripts] [--dry-run]
  python tools/chunker.py --selftest

Пути с пробелами, скобками и кириллицей поддерживаются — передавайте в кавычках.
"""

from __future__ import annotations

import bisect
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

PROG = "chunker.py"

# Версия формата plan.json. Инструменты-читатели сверяются именно с этой строкой.
SCHEMA = "chunk-plan/1"

# Абсолютный минимум длины чанка. Короче секунды модели отдавать нечего: это
# обрывок слова или щелчок, поэтому такой чанк либо прирастает к предыдущему,
# либо честно уходит в dropped.
MIN_CHUNK_FLOOR = 1.0

# Все времена в плане — секунды с тремя знаками: миллисекунда мельче, чем
# точность и silencedetect, и нарезки ffmpeg.
TIME_DIGITS = 3

# Допуск для сравнения секунд: разницы меньше микросекунды не существует ни для
# ffmpeg, ни для модели, а на float-арифметике они возникают сами.
EPS = 1e-6


# --------------------------------------------------------------- ошибки ---


class ToolError(Exception):
    """Ошибка окружения или данных: сообщение уже на русском и готово в stderr."""


class UsageError(ToolError):
    """Ошибка в аргументах командной строки."""


def die(message: str) -> int:
    """Печатает понятную ошибку и возвращает код выхода (трейсбека не будет)."""
    print(f"{PROG}: {message}", file=sys.stderr)
    return 1


# ------------------------------------------------------------ параметры ---


@dataclass(frozen=True)
class PlanParams:
    """Параметры планирования. Порядок полей = порядок ключей в plan.json."""

    noise: str = "-35dB"
    min_silence: float = 0.4
    max_chunk: float = 22.0
    hard_max_chunk: float = 24.0
    min_chunk: float = 8.0
    drop_pause: float = 2.0
    pad: float = 0.1

    def validate(self) -> None:
        """Отсекает заведомо бессмысленные наборы до запуска ffmpeg."""
        if not self.noise.strip():
            raise UsageError("--noise не может быть пустым (пример: -35dB)")
        for name in ("min_silence", "max_chunk", "hard_max_chunk", "min_chunk"):
            if getattr(self, name) <= 0:
                raise UsageError(f"--{name.replace('_', '-')} должен быть больше нуля")
        if self.drop_pause < 0:
            raise UsageError("--drop-pause не может быть отрицательным")
        if self.pad < 0:
            raise UsageError("--pad не может быть отрицательным")
        if self.max_chunk > self.hard_max_chunk:
            raise UsageError(
                "--max-chunk не может быть больше --hard-max-chunk "
                f"({self.max_chunk} > {self.hard_max_chunk})"
            )
        if self.min_chunk > self.max_chunk:
            raise UsageError(
                "--min-chunk не может быть больше --max-chunk "
                f"({self.min_chunk} > {self.max_chunk})"
            )
        if self.hard_max_chunk < MIN_CHUNK_FLOOR:
            raise UsageError(
                f"--hard-max-chunk меньше минимальной длины чанка ({MIN_CHUNK_FLOOR} с)"
            )

    def as_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class Chunk:
    """Чанк плана. Порядок полей = порядок ключей в plan.json."""

    index: int
    start: float
    end: float
    duration: float
    file: str
    cut_reason: str


@dataclass
class Dropped:
    """Выброшенный участок записи. Порядок полей = порядок ключей в plan.json."""

    start: float
    end: float
    duration: float
    reason: str


@dataclass
class _Raw:
    """Черновой чанк до нумерации и округления; segment — номер отрезка звука."""

    start: float
    end: float
    reason: str
    segment: int
    dur: float = field(init=False)

    def __post_init__(self) -> None:
        self.dur = self.end - self.start


def _round(value: float) -> float:
    return round(float(value), TIME_DIGITS)


# ------------------------------------------- чистая логика планирования ---


def normalize_silences(
    duration: float, silences: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Приводит интервалы тишины к порядку: обрезает по записи, сортирует, сливает.

    silencedetect умеет выдать start чуть меньше нуля и end за концом файла
    (фильтр работает по кадрам, а не по точным границам), а при нестандартных
    параметрах — и пересекающиеся интервалы. Планировщик рассчитывает на
    непересекающийся отсортированный список, поэтому разбираемся здесь один раз.
    """
    clean: list[tuple[float, float]] = []
    for start, end in silences:
        s = max(0.0, min(float(start), duration))
        e = max(0.0, min(float(end), duration))
        if e - s > EPS:
            clean.append((s, e))
    clean.sort()

    merged: list[tuple[float, float]] = []
    for start, end in clean:
        if merged and start <= merged[-1][1] + EPS:
            # Соприкасающиеся интервалы — это одна пауза: иначе пауза в 3 с,
            # разбитая на две по 1,5 с, не попала бы под --drop-pause.
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]],
    params: PlanParams,
) -> tuple[list[Chunk], list[Dropped]]:
    """Строит план нарезки. Чистая функция: ни ffmpeg, ни файлов, ни времени.

    Вход: длительность записи в секундах, интервалы тишины и параметры.
    Выход: список чанков (уже пронумерованных, с именами файлов) и список
    выброшенных участков. Вместе они покрывают запись целиком.
    """
    params.validate()
    duration = float(duration)
    if duration <= 0:
        raise ToolError(f"длительность записи должна быть больше нуля, получено: {duration}")

    quiet = normalize_silences(duration, silences)
    segments, dropped = _split_by_long_pauses(duration, quiet, params)

    # Кандидаты на разрез — центры коротких пауз: в центре паузы разрыв слова
    # наименее вероятен. Длинные паузы в кандидаты не попадают: они уже вырезаны.
    cut_points = sorted(
        (s + e) / 2.0 for s, e in quiet if e - s <= params.drop_pause + EPS
    )

    raw: list[_Raw] = []
    for number, (start, end, end_reason) in enumerate(segments):
        raw.extend(_cut_segment(start, end, end_reason, number, cut_points, params))

    raw = _merge_short_tails(raw, params)
    raw, floor_dropped = _fix_tiny_chunks(raw, quiet, params)
    dropped.extend(floor_dropped)

    chunks: list[Chunk] = []
    for index, item in enumerate(raw):
        start = _round(item.start)
        end = _round(item.end)
        chunks.append(
            Chunk(
                index=index,
                start=start,
                end=end,
                duration=_round(end - start),
                file=f"chunks/chunk_{index:03d}.wav",
                cut_reason=item.reason,
            )
        )

    dropped.sort(key=lambda d: d.start)
    return chunks, dropped


def _split_by_long_pauses(
    duration: float, quiet: list[tuple[float, float]], params: PlanParams
) -> tuple[list[tuple[float, float, str]], list[Dropped]]:
    """Делит запись на отрезки звука, выбрасывая паузы длиннее --drop-pause.

    Возвращает отрезки (start, end, причина окончания) и выброшенные участки.
    Причина окончания отрезка — это cut_reason последнего чанка в нём: "pause",
    если дальше выброшенная пауза, и "end", если дальше конец записи.
    """
    segments: list[tuple[float, float, str]] = []
    dropped: list[Dropped] = []
    cursor = 0.0

    for start, end in quiet:
        pause = end - start
        if pause <= params.drop_pause + EPS:
            continue

        # pad возвращает в звук немного тишины по краям, чтобы разрез не съел
        # дыхание и начало слова. На третью часть паузы он ограничен только для
        # защиты от абсурдных настроек (--pad больше самой паузы): при значениях
        # по умолчанию (пауза > 2 с, pad = 0,1 с) ограничение не срабатывает и
        # границы получаются ровно silence_start + pad и silence_end - pad.
        pad = min(params.pad, pause / 3.0)

        # Паузу, начинающуюся там же, где кончился предыдущий выброс, pad-ом не
        # раздвигаем: иначе перед ней появился бы обрывок звука в один pad.
        seg_end = max(cursor, start + pad) if start > cursor + EPS else cursor
        if seg_end - cursor > EPS:
            segments.append((cursor, seg_end, "pause"))

        if end >= duration - EPS:
            # Пауза упирается в конец записи: звука после неё нет, и pad оставил
            # бы в плане обрывок в 0,1 с, который всё равно пришлось бы выбросить.
            next_start = duration
        else:
            next_start = end - pad
        next_start = max(next_start, seg_end)

        if next_start - seg_end > EPS:
            dropped.append(
                Dropped(
                    start=_round(seg_end),
                    end=_round(next_start),
                    duration=_round(next_start - seg_end),
                    reason=f"пауза {pause:.3f} с длиннее --drop-pause ({params.drop_pause} с)",
                )
            )
        cursor = max(cursor, next_start)

    if duration - cursor > EPS:
        segments.append((cursor, duration, "end"))
    return segments, dropped


def _cut_segment(
    start: float,
    end: float,
    end_reason: str,
    number: int,
    cut_points: list[float],
    params: PlanParams,
) -> list[_Raw]:
    """Режет один отрезок непрерывного звука на чанки, жадно слева направо."""
    out: list[_Raw] = []
    cursor = start
    while True:
        rest = end - cursor
        if rest <= params.max_chunk + EPS:
            out.append(_Raw(cursor, end, end_reason, number))
            return out

        # Ищем самую позднюю тишину, которая даёт чанк не короче --min-chunk и
        # не длиннее --max-chunk: это ближайшая подходящая к границе max_chunk,
        # то есть чанков будет меньше, а каждый — ближе к рабочей длине.
        lo = cursor + max(params.min_chunk, EPS)
        hi = cursor + params.max_chunk
        position = bisect.bisect_right(cut_points, hi + EPS) - 1
        cut = cut_points[position] if position >= 0 else None

        if cut is not None and cut >= lo - EPS and cut > cursor + EPS and cut < end - EPS:
            out.append(_Raw(cursor, cut, "silence", number))
            cursor = cut
        else:
            # Сплошная речь без пауз: режем жёстко, иначе чанк перерастёт лимит
            # модели. Разрез по --max-chunk, а не по --hard-max-chunk: запас до
            # жёсткого предела нужен хвосту, который потом может прирасти сюда.
            out.append(_Raw(cursor, cursor + params.max_chunk, "hard_max", number))
            cursor = out[-1].end


def _merge_short_tails(raw: list[_Raw], params: PlanParams) -> list[_Raw]:
    """Присоединяет хвост короче --min-chunk к предыдущему чанку того же отрезка.

    Хвост возникает в конце каждого отрезка звука: остаток от деления на
    --max-chunk. Отдельным чанком в пару секунд модели отдавать невыгодно
    (контекста мало, накладные расходы те же), поэтому хвост прирастает к
    предыдущему чанку — но только если сумма остаётся в --hard-max-chunk, иначе
    хвост остаётся самостоятельным чанком.
    """
    merged: list[_Raw] = []
    for position, item in enumerate(raw):
        # Хвост отрезка — чанк, после которого номер отрезка меняется или список
        # кончается: только он может оказаться короче --min-chunk.
        is_last = position + 1 == len(raw) or raw[position + 1].segment != item.segment
        can_merge = (
            is_last
            and item.dur < params.min_chunk - EPS
            and merged
            and merged[-1].segment == item.segment
            and item.end - merged[-1].start <= params.hard_max_chunk + EPS
        )
        if can_merge:
            merged[-1] = _Raw(merged[-1].start, item.end, item.reason, item.segment)
        else:
            merged.append(item)
    return merged


def _sound_seconds(start: float, end: float, quiet: list[tuple[float, float]]) -> float:
    """Сколько в интервале звука, а не тишины."""
    silence = 0.0
    for s, e in quiet:
        overlap = min(end, e) - max(start, s)
        if overlap > 0:
            silence += overlap
    return max(0.0, end - start - silence)


def _fix_tiny_chunks(
    raw: list[_Raw], quiet: list[tuple[float, float]], params: PlanParams
) -> tuple[list[_Raw], list[Dropped]]:
    """Убирает чанки, в которых звука меньше MIN_CHUNK_FLOOR: присоединяет или выбрасывает.

    Такой огрызок появляется, например, между двумя длинными паузами, когда
    звука между ними меньше секунды. Присоединять его можно только к соседу по
    тому же отрезку: чанк не имеет права перешагнуть выброшенную паузу.

    Мерить здесь нужно именно звук, а не длину интервала: в неё входят --pad по
    краям и короткие паузы внутри, которые разрезом не стали. На прогоне
    2026-10-03 это вышло наружу — чанк 00:03:13.983–00:03:15.600 длиной 1.617 с
    содержал 0.971 с звука, и оба чекпойнта GigaAM вернули по нему пустой текст.
    Пустой сегмент в транскрипте — нарушение допуска (benchmark-plan.md,
    раздел 7), то есть отдавать такой огрызок модели нельзя.
    """
    out: list[_Raw] = []
    dropped: list[Dropped] = []
    for item in raw:
        sound = _sound_seconds(item.start, item.end, quiet)
        if sound >= MIN_CHUNK_FLOOR - EPS:
            out.append(item)
            continue
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.segment == item.segment
            and item.end - prev.start <= params.hard_max_chunk + EPS
        ):
            out[-1] = _Raw(prev.start, item.end, item.reason, prev.segment)
            continue
        dropped.append(
            Dropped(
                start=_round(item.start),
                end=_round(item.end),
                duration=_round(item.dur),
                reason=(
                    f"звука в чанке {sound:.3f} с — короче {MIN_CHUNK_FLOOR:.1f} с, "
                    f"и присоединить к предыдущему нельзя без выхода "
                    f"за --hard-max-chunk"
                ),
            )
        )
    return out, dropped


def check_plan(
    duration: float,
    chunks: list[Chunk],
    dropped: list[Dropped],
    params: PlanParams,
) -> list[str]:
    """Проверяет инварианты плана. Пустой список — план пригоден для нарезки.

    Это не декорация: ошибка планировщика проявилась бы уже внутри gigaam
    (ValueError на длинном чанке) или дырой в расшифровке, и разбираться в ней
    было бы на порядок дороже, чем увидеть здесь.
    """
    problems: list[str] = []
    previous_end = 0.0
    for chunk in chunks:
        where = f"чанк {chunk.index}"
        if chunk.end <= chunk.start:
            problems.append(f"{where}: end ({chunk.end}) не больше start ({chunk.start})")
        if chunk.start < -EPS or chunk.end > duration + EPS:
            problems.append(
                f"{where}: границы [{chunk.start}, {chunk.end}] вне записи [0, {duration}]"
            )
        if chunk.start < previous_end - EPS:
            problems.append(
                f"{where}: начинается ({chunk.start}) раньше конца предыдущего ({previous_end})"
            )
        if chunk.duration > params.hard_max_chunk + 1e-3:
            problems.append(
                f"{where}: длительность {chunk.duration} с больше "
                f"--hard-max-chunk ({params.hard_max_chunk} с)"
            )
        if chunk.duration < MIN_CHUNK_FLOOR - 1e-3:
            problems.append(
                f"{where}: длительность {chunk.duration} с меньше {MIN_CHUNK_FLOOR} с"
            )
        if abs(chunk.duration - (chunk.end - chunk.start)) > 1e-3:
            problems.append(f"{where}: duration не совпадает с end - start")
        if chunk.file != f"chunks/chunk_{chunk.index:03d}.wav":
            problems.append(f"{where}: имя файла не соответствует индексу: {chunk.file}")
        previous_end = chunk.end

    for index, item in enumerate(chunks):
        if item.index != index:
            problems.append(f"нумерация чанков прервана на позиции {index}: index={item.index}")

    for item in dropped:
        if item.end < item.start - EPS:
            problems.append(f"выброшенный участок [{item.start}, {item.end}] вывернут")
        if item.start < -EPS or item.end > duration + EPS:
            problems.append(
                f"выброшенный участок [{item.start}, {item.end}] вне записи [0, {duration}]"
            )
        for chunk in chunks:
            overlap = min(chunk.end, item.end) - max(chunk.start, item.start)
            if overlap > 1e-3:
                problems.append(
                    f"чанк {chunk.index} [{chunk.start}, {chunk.end}] пересекает "
                    f"выброшенный участок [{item.start}, {item.end}]"
                )

    covered = sum(c.duration for c in chunks) + sum(d.duration for d in dropped)
    if abs(covered - duration) > 0.01 + 0.002 * (len(chunks) + len(dropped)):
        problems.append(
            f"сумма чанков и выброшенных участков ({covered:.3f} с) "
            f"не совпадает с длительностью записи ({duration:.3f} с)"
        )
    return problems


# --------------------------------------------------------- ffmpeg/ffprobe ---


SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def parse_silencedetect(stderr: str, duration: float) -> list[tuple[float, float]]:
    """Разбирает вывод silencedetect в список интервалов тишины. Чистая функция.

    Фильтр печатает в stderr пары строк:
      [silencedetect @ ...] silence_start: 12.345
      [silencedetect @ ...] silence_end: 15.678 | silence_duration: 3.333
    Если тишина тянется до конца файла, silence_end не печатается вовсе —
    незакрытый silence_start трактуем как тишину до конца записи, иначе последняя
    пауза записи осталась бы невидимой для планировщика.
    """
    silences: list[tuple[float, float]] = []
    pending: float | None = None
    for line in stderr.splitlines():
        if "silence_start" in line:
            match = SILENCE_START_RE.search(line)
            if match:
                pending = float(match.group(1))
        if "silence_end" in line:
            match = SILENCE_END_RE.search(line)
            if match and pending is not None:
                silences.append((pending, float(match.group(1))))
                pending = None
    if pending is not None and duration - pending > EPS:
        silences.append((pending, duration))
    return silences


def _tail(text: str, lines: int = 5) -> str:
    """Последние строки вывода: в сообщении об ошибке полезен только хвост лога."""
    rows = [row for row in (text or "").splitlines() if row.strip()]
    return "\n".join(rows[-lines:])


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess[str]:
    """Запускает внешнюю команду и превращает любой сбой в понятную ошибку."""
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise ToolError(f"{what}: не удалось запустить {cmd[0]} ({exc})") from exc
    if proc.returncode != 0:
        detail = _tail(proc.stderr)
        raise ToolError(
            f"{what}: {cmd[0]} вернул код {proc.returncode}"
            + (f"\n{detail}" if detail else "")
        )
    return proc


def require_tools() -> None:
    for tool in ("ffprobe", "ffmpeg"):
        if shutil.which(tool) is None:
            raise ToolError(f"не найден {tool} (установите: brew install ffmpeg)")


def probe_duration(path: Path) -> float:
    """Длительность файла в секундах по ffprobe."""
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            "-i",
            str(path),
        ],
        "определение длительности",
    )
    raw = proc.stdout.strip().splitlines()
    value = raw[0].strip() if raw else ""
    try:
        duration = float(value)
    except ValueError:
        raise ToolError(f"не удалось определить длительность записи: {path}") from None
    if duration <= 0:
        raise ToolError(f"длительность записи нулевая или отрицательная: {path}")
    return duration


def detect_silences(path: Path, params: PlanParams, duration: float) -> list[tuple[float, float]]:
    """Находит тишину одним проходом ffmpeg silencedetect.

    Один проход на весь файл, а не вызов на каждый чанк: декодирование — самая
    дорогая часть, и платить за неё дважды незачем. Уровень логов -v info
    обязателен: silencedetect пишет результаты именно в info, при -v error
    вывод оказался бы пустым, и файл выглядел бы как сплошная речь.
    """
    proc = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            f"silencedetect=noise={params.noise}:d={params.min_silence}",
            "-f",
            "null",
            "-",
        ],
        "поиск тишины",
    )
    return parse_silencedetect(proc.stderr, duration)


def cut_chunk(source: Path, chunk: Chunk, target: Path) -> None:
    """Вырезает один чанк в 16 кГц моно pcm_s16le.

    -ss перед -i — быстрый поиск: для PCM WAV он точен до сэмпла, а декодировать
    файл с начала ради каждого чанка значило бы умножить время нарезки на их
    количество. Частота и число каналов задаются явно: на вход модели должен
    попасть ровно тот формат, в котором она работает, даже если источник внезапно
    окажется другим.
    """
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-ss",
            f"{chunk.start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{chunk.duration:.3f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(target),
        ],
        f"нарезка чанка {chunk.index}",
    )


# ------------------------------------------------------------------ CLI ---


USAGE = """Использование: python tools/chunker.py [опции]

Строит план нарезки 16 кГц моно WAV на чанки под GigaAM: находит тишину через
ffmpeg silencedetect, режет по центрам тишины, выбрасывает длинные паузы и
следит, чтобы ни один чанк не превысил лимит модели. План пишется в plan.json,
чанки — в подкаталог chunks/ рядом с планом.

Опции:
  --input <путь>        входной 16 кГц моно WAV (обязателен, кроме --selftest)
  --outdir <каталог>    куда положить план и чанки (по умолчанию data/transcripts)
  --plan <путь>         путь к файлу плана (по умолчанию <outdir>/plan.json)
  --dry-run             только посчитать план, чанки не нарезать
  --noise <уровень>     порог тишины для silencedetect (по умолчанию -35dB)
  --min-silence <сек>   минимальная длина тишины (по умолчанию 0.4)
  --max-chunk <сек>     рабочая максимальная длина чанка (по умолчанию 22.0)
  --hard-max-chunk <сек> абсолютный предел длины чанка (по умолчанию 24.0)
  --min-chunk <сек>     минимальная длина чанка при разрезе (по умолчанию 8.0)
  --drop-pause <сек>    паузы длиннее выбрасываются (по умолчанию 2.0)
  --pad <сек>           сколько тишины оставить по краям паузы (по умолчанию 0.1)
  --selftest            прогнать встроенные проверки планировщика и выйти
  -h, --help            показать эту справку

Примеры:
  python tools/chunker.py --input "data/samples/фрагмент_16k_mono.wav" --dry-run
  python tools/chunker.py --input data/samples/a.wav --outdir data/transcripts
  python tools/chunker.py --selftest
"""


@dataclass
class Options:
    input: Path | None = None
    outdir: Path = Path("data/transcripts")
    plan: Path | None = None
    dry_run: bool = False
    selftest: bool = False
    help: bool = False
    params: PlanParams = field(default_factory=PlanParams)


def _need_value(argv: list[str], index: int, name: str) -> str:
    if index + 1 >= len(argv):
        raise UsageError(f"{name} требует значение")
    return argv[index + 1]


def _need_float(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise UsageError(f"{name} ожидает число в секундах, получено: {raw}") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise UsageError(f"{name} ожидает конечное число, получено: {raw}")
    return value


def parse_args(argv: list[str]) -> Options:
    """Разбирает аргументы вручную — ради сообщений на русском, как в scripts/*.sh."""
    opts = Options()
    # Значения параметров планирования собираются в словарь и отдаются в
    # PlanParams целиком: так в одном месте и значения по умолчанию, и порядок
    # ключей в plan.json, и проверка согласованности.
    values: dict[str, object] = {}

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
        if arg == "--dry-run":
            opts.dry_run = True
            index += 1
            continue
        if arg == "--input":
            opts.input = Path(_need_value(argv, index, arg))
            index += 2
            continue
        if arg == "--outdir":
            opts.outdir = Path(_need_value(argv, index, arg))
            index += 2
            continue
        if arg == "--plan":
            opts.plan = Path(_need_value(argv, index, arg))
            index += 2
            continue
        if arg == "--noise":
            values["noise"] = _need_value(argv, index, arg)
            index += 2
            continue
        if arg in (
            "--min-silence",
            "--max-chunk",
            "--hard-max-chunk",
            "--min-chunk",
            "--drop-pause",
            "--pad",
        ):
            raw = _need_value(argv, index, arg)
            values[arg.lstrip("-").replace("-", "_")] = _need_float(arg, raw)
            index += 2
            continue
        raise UsageError(f"неизвестная опция: {arg} (см. --help)")

    opts.params = PlanParams(**values)  # type: ignore[arg-type]
    opts.params.validate()
    return opts


# ------------------------------------------------------------- сценарий ---


def build_plan_document(
    source: Path,
    duration: float,
    silences: list[tuple[float, float]],
    params: PlanParams,
    chunks: list[Chunk],
    dropped: list[Dropped],
) -> dict[str, object]:
    """Собирает plan.json. Порядок ключей фиксирован: это часть формата."""
    return {
        "schema": SCHEMA,
        "source": str(source),
        "source_duration": _round(duration),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "params": params.as_dict(),
        "silences": [{"start": _round(s), "end": _round(e)} for s, e in silences],
        "chunks": [
            {
                "index": c.index,
                "start": c.start,
                "end": c.end,
                "duration": c.duration,
                "file": c.file,
                "cut_reason": c.cut_reason,
            }
            for c in chunks
        ],
        "dropped": [
            {
                "start": d.start,
                "end": d.end,
                "duration": d.duration,
                "reason": d.reason,
            }
            for d in dropped
        ],
        "coverage_seconds": _round(sum(c.duration for c in chunks)),
        "dropped_seconds": _round(sum(d.duration for d in dropped)),
    }


def print_summary(
    plan_path: Path,
    document: dict[str, object],
    chunks: list[Chunk],
    dropped: list[Dropped],
    duration: float,
    dry_run: bool,
) -> None:
    """Печатает сводку по плану: цифры берутся из плана, а не считаются заново."""
    coverage = float(document["coverage_seconds"])  # type: ignore[arg-type]
    dropped_seconds = float(document["dropped_seconds"])  # type: ignore[arg-type]
    silence_count = len(document["silences"])  # type: ignore[arg-type]
    durations = [c.duration for c in chunks]
    print("================ ПЛАН НАРЕЗКИ ================")
    print(f"Источник:           {document['source']}")
    print(f"Длительность:       {duration:.3f} с")
    print(f"Интервалов тишины:  {silence_count}")
    print(f"Чанков:             {len(chunks)}")
    print(
        f"Покрытие:           {coverage:.3f} с"
        + (f" ({coverage / duration * 100:.1f} % записи)" if duration > 0 else "")
    )
    print(f"Выброшено пауз:     {len(dropped)} ({dropped_seconds:.3f} с)")
    if durations:
        print(
            f"Длина чанка:        мин {min(durations):.3f} с, макс {max(durations):.3f} с"
        )
    else:
        print("Длина чанка:        чанков нет")
    print(f"План:               {plan_path}")
    if dry_run:
        print("Чанки не нарезаны:  указан --dry-run")
    else:
        print(f"Нарезано файлов:    {len(chunks)} в {plan_path.parent / 'chunks'}")
    print("================ ГОТОВО ================")


def run(opts: Options) -> int:
    if opts.input is None:
        raise UsageError("не указан --input (путь к 16 кГц моно WAV); см. --help")
    source = opts.input
    if not source.exists():
        raise ToolError(f"файл не найден: {source}")
    if not source.is_file():
        raise ToolError(f"это не файл: {source}")

    require_tools()
    duration = probe_duration(source)
    silences = detect_silences(source, opts.params, duration)
    silences = normalize_silences(duration, silences)

    chunks, dropped = plan_chunks(duration, silences, opts.params)
    problems = check_plan(duration, chunks, dropped, opts.params)
    if problems:
        raise ToolError(
            "план нарушает собственные инварианты, нарезка отменена:\n  "
            + "\n  ".join(problems)
        )

    plan_path = opts.plan if opts.plan is not None else opts.outdir / "plan.json"
    try:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"не удалось создать каталог {plan_path.parent}: {exc}") from exc

    document = build_plan_document(source, duration, silences, opts.params, chunks, dropped)
    try:
        plan_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise ToolError(f"не удалось записать план {plan_path}: {exc}") from exc

    if not opts.dry_run:
        chunks_dir = plan_path.parent / "chunks"
        try:
            chunks_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(f"не удалось создать каталог {chunks_dir}: {exc}") from exc
        for chunk in chunks:
            cut_chunk(source, chunk, plan_path.parent / chunk.file)

    print_summary(plan_path, document, chunks, dropped, duration, opts.dry_run)
    return 0


# ------------------------------------------------------------ селф-тест ---


def _selftest_params(**overrides) -> PlanParams:
    base = PlanParams().as_dict()
    base.update(overrides)
    return PlanParams(**base)  # type: ignore[arg-type]


def _invariants(
    name: str,
    duration: float,
    chunks: list[Chunk],
    dropped: list[Dropped],
    params: PlanParams,
) -> list[str]:
    return [f"{name}: {problem}" for problem in check_plan(duration, chunks, dropped, params)]


def selftest() -> int:
    """Проверяет планировщик на синтетических данных: без ffmpeg и без аудио.

    Проверять именно так можно потому, что plan_chunks() — чистая функция: ей
    всё равно, пришёл список тишины от ffmpeg или из этого файла. В .venv нет
    pytest, поэтому проверки живут здесь и запускаются флагом --selftest.
    """
    failures: list[str] = []
    cases_done = 0

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    # --- 1. Пауза 3 с выбрасывается и даёт разрыв между чанками ---------------
    params = _selftest_params()
    chunks, dropped = plan_chunks(60.0, [(20.0, 23.0)], params)
    failures += _invariants("пауза 3 с", 60.0, chunks, dropped, params)
    check(len(dropped) == 1, f"пауза 3 с: ожидался один выброс, получено {len(dropped)}")
    if dropped:
        check(
            abs(dropped[0].start - 20.1) < 1e-6 and abs(dropped[0].end - 22.9) < 1e-6,
            f"пауза 3 с: выброшен участок [{dropped[0].start}, {dropped[0].end}], "
            "ожидался [20.1, 22.9]",
        )
    check(
        any(abs(c.end - 20.1) < 1e-6 for c in chunks),
        "пауза 3 с: ни один чанк не заканчивается на silence_start + pad (20.1)",
    )
    check(
        any(abs(c.start - 22.9) < 1e-6 for c in chunks),
        "пауза 3 с: ни один чанк не начинается на silence_end - pad (22.9)",
    )
    # Сравнение именно с выброшенным участком [20.1, 22.9], а не с самой тишиной
    # [20.0, 23.0]: заходить в тишину на pad чанк обязан по замыслу, а вот
    # выброшенного участка он касаться не имеет права.
    check(
        not any(c.start < 22.9 - 1e-9 and c.end > 20.1 + 1e-9 for c in chunks),
        "пауза 3 с: чанк пересекает выброшенную паузу",
    )
    cases_done += 1

    # --- 2. Сплошная речь: режется только жёстко по --max-chunk ---------------
    params = _selftest_params()
    chunks, dropped = plan_chunks(100.0, [], params)
    failures += _invariants("100 с без тишины", 100.0, chunks, dropped, params)
    check(not dropped, "100 с без тишины: выбрасывать нечего, а dropped не пуст")
    check(
        all(c.cut_reason in ("hard_max", "end") for c in chunks),
        "100 с без тишины: появился разрез по тишине, которой нет: "
        + ", ".join(c.cut_reason for c in chunks),
    )
    check(
        sum(1 for c in chunks if c.cut_reason == "hard_max") == 4,
        f"100 с без тишины: ожидалось 4 жёстких разреза, чанков {len(chunks)}",
    )
    check(
        all(abs(c.duration - 22.0) < 1e-6 for c in chunks if c.cut_reason == "hard_max"),
        "100 с без тишины: жёсткий разрез не по --max-chunk",
    )
    check(
        abs(sum(c.duration for c in chunks) - 100.0) < 1e-6,
        "100 с без тишины: покрытие не равно длительности",
    )
    cases_done += 1

    # --- 3. Короткие тишины каждые 10 с: разрезы по их центрам ----------------
    params = _selftest_params()
    silences = [(float(t), float(t) + 0.4) for t in range(10, 100, 10)]
    centers = {round(s + 0.2, 3) for s, _ in silences}
    chunks, dropped = plan_chunks(100.0, silences, params)
    failures += _invariants("тишины каждые 10 с", 100.0, chunks, dropped, params)
    check(not dropped, "тишины каждые 10 с: короткие паузы выбрасывать нельзя")
    check(
        all(c.cut_reason == "silence" for c in chunks[:-1]),
        "тишины каждые 10 с: не все разрезы сделаны по тишине",
    )
    check(
        all(c.end in centers for c in chunks[:-1]),
        "тишины каждые 10 с: разрез не попал в центр тишины: "
        + ", ".join(str(c.end) for c in chunks[:-1]),
    )
    check(
        chunks[-1].cut_reason == "end" and abs(chunks[-1].end - 100.0) < 1e-6,
        "тишины каждые 10 с: последний чанк не доходит до конца записи",
    )
    cases_done += 1

    # --- 4. Хвост короче --min-chunk прирастает к предыдущему чанку -----------
    params = _selftest_params()
    chunks, dropped = plan_chunks(23.0, [(18.0, 18.4)], params)
    failures += _invariants("короткий хвост", 23.0, chunks, dropped, params)
    check(
        len(chunks) == 1 and abs(chunks[0].duration - 23.0) < 1e-6,
        "короткий хвост: хвост 4.8 с не присоединился к предыдущему чанку, "
        f"чанков {len(chunks)}",
    )
    check(not dropped, "короткий хвост: ничего выбрасывать не требовалось")
    # А если присоединение выводит за --hard-max-chunk, хвост остаётся сам собой.
    chunks, dropped = plan_chunks(27.0, [(21.0, 21.4)], params)
    failures += _invariants("короткий хвост у предела", 27.0, chunks, dropped, params)
    check(
        len(chunks) == 2,
        "короткий хвост у предела: хвост присоединили в обход --hard-max-chunk, "
        f"чанков {len(chunks)}",
    )
    cases_done += 1

    # --- 5. Огрызок звука между двумя паузами выбрасывается -------------------
    params = _selftest_params()
    chunks, dropped = plan_chunks(30.0, [(10.0, 13.0), (13.5, 17.0)], params)
    failures += _invariants("огрызок между паузами", 30.0, chunks, dropped, params)
    check(
        len(chunks) == 2,
        f"огрызок между паузами: ожидалось 2 чанка, получено {len(chunks)}",
    )
    check(
        any("короче" in d.reason for d in dropped),
        "огрызок между паузами: в dropped нет записи о чанке короче минимума",
    )
    check(
        all(c.duration >= MIN_CHUNK_FLOOR for c in chunks),
        "огрызок между паузами: в плане остался чанк короче минимума",
    )
    cases_done += 1

    # --- 5a. Огрызок длиннее секунды, но звука в нём меньше секунды ------------
    # Регрессия прогона 2026-10-03: в длину интервала входят --pad по краям и
    # короткая пауза внутри, поэтому мерить минимум её нельзя. Здесь остров между
    # двумя выброшенными паузами длиной 2.2 с, а звука в нём 0.6 с — модель по
    # такому чанку возвращает пустой текст, а пустой сегмент не проходит допуск
    # (benchmark-plan.md, раздел 7).
    params = _selftest_params()
    chunks, dropped = plan_chunks(30.0, [(5.0, 9.0), (9.5, 10.9), (11.0, 15.0)], params)
    failures += _invariants("огрызок по звуку", 30.0, chunks, dropped, params)
    check(
        len(chunks) == 2,
        "огрызок по звуку: ожидалось 2 чанка, получено "
        + f"{len(chunks)}: " + ", ".join(f"{c.start}–{c.end}" for c in chunks),
    )
    check(
        any("звука в чанке" in d.reason for d in dropped),
        "огрызок по звуку: остров 8.9–11.1 (2.2 с длиной, 0.6 с звука) остался в плане",
    )
    cases_done += 1

    # --- 6. Тишина до конца записи: хвост-пауза выбрасывается -----------------
    params = _selftest_params()
    chunks, dropped = plan_chunks(40.0, [(0.0, 4.0), (36.0, 40.0)], params)
    failures += _invariants("паузы по краям", 40.0, chunks, dropped, params)
    check(
        len(chunks) == 2 and abs(chunks[0].start - 3.9) < 1e-6,
        f"паузы по краям: первый чанк должен начинаться на 3.9, чанков {len(chunks)}",
    )
    if chunks:
        check(
            abs(chunks[-1].end - 36.1) < 1e-6,
            f"паузы по краям: звук должен кончаться на 36.1, а кончается на {chunks[-1].end}",
        )
    check(
        len(dropped) == 2
        and abs(dropped[0].start) < 1e-6
        and abs(dropped[-1].end - 40.0) < 1e-6,
        "паузы по краям: выброшенные участки не упираются в границы записи",
    )
    cases_done += 1

    # --- 7. Разбор вывода silencedetect, включая незакрытый silence_start -----
    stderr = (
        "[silencedetect @ 0x7f8] silence_start: 1.5\n"
        "[silencedetect @ 0x7f8] silence_end: 2.75 | silence_duration: 1.25\n"
        "size=N/A time=00:00:10.00 bitrate=N/A speed=120x\n"
        "[silencedetect @ 0x7f8] silence_start: 8.25\n"
    )
    parsed = parse_silencedetect(stderr, 10.0)
    check(
        parsed == [(1.5, 2.75), (8.25, 10.0)],
        f"разбор silencedetect: получено {parsed}, ожидалось [(1.5, 2.75), (8.25, 10.0)]",
    )
    check(
        parse_silencedetect("", 10.0) == [],
        "разбор silencedetect: пустой вывод должен давать пустой список",
    )
    cases_done += 1

    # --- 8. Нормализация тишины: порядок, пересечения, выход за границы -------
    normalized = normalize_silences(10.0, [(5.0, 6.0), (-1.0, 0.5), (5.5, 7.0), (9.5, 12.0)])
    check(
        normalized == [(0.0, 0.5), (5.0, 7.0), (9.5, 10.0)],
        f"нормализация тишины: получено {normalized}",
    )
    # Пауза, разбитая на две соприкасающиеся, должна слиться и попасть под --drop-pause.
    params = _selftest_params()
    chunks, dropped = plan_chunks(60.0, [(20.0, 21.5), (21.5, 23.0)], params)
    failures += _invariants("слипшиеся паузы", 60.0, chunks, dropped, params)
    check(
        len(dropped) == 1 and dropped[0].duration > 2.0,
        "слипшиеся паузы: две половины одной паузы не слились в один выброс",
    )
    cases_done += 1

    # --- 9. Нестандартные параметры: план остаётся корректным -----------------
    for noise_params in (
        _selftest_params(max_chunk=5.0, hard_max_chunk=5.0, min_chunk=1.0, drop_pause=0.3),
        _selftest_params(min_chunk=22.0, pad=0.5, drop_pause=1.0),
        _selftest_params(max_chunk=24.0, min_chunk=24.0),
    ):
        silences = [(float(t) + 0.3, float(t) + 1.9) for t in range(3, 90, 7)]
        chunks, dropped = plan_chunks(95.0, silences, noise_params)
        failures += _invariants(f"параметры {noise_params}", 95.0, chunks, dropped, noise_params)
    cases_done += 1

    print(f"{PROG}: селф-тест, проверенных случаев: {cases_done}")
    if failures:
        print(f"{PROG}: селф-тест ПРОВАЛЕН, нарушений: {len(failures)}", file=sys.stderr)
        for problem in failures:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{PROG}: селф-тест пройден")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        opts = parse_args(args)
        if opts.help:
            print(USAGE, end="")
            return 0
        if opts.selftest:
            return selftest()
        return run(opts)
    except ToolError as exc:
        return die(str(exc))
    except KeyboardInterrupt:
        return die("прервано пользователем")


if __name__ == "__main__":
    sys.exit(main())
