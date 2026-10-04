#!/usr/bin/env python3
"""speakers.py — разбиение расшифровки на фразы и привязка говорящих к словам.

Сходятся три источника, каждый из которых знает только своё:

  transcript/1   — пословные таймкоды (tools/transcribe.py). Паузы между словами
                   считаются только здесь: в punctuated/1 их нет;
  punctuated/1   — предложения с пунктуацией, заглавными и абзацами
                   (tools/punctuate.py). Текст фраз берётся только отсюда;
  diarization/1  — интервалы говорящих (tools/diarize.py). Имена говорящих
                   берутся только отсюда.

На выходе — файл схемы speakers/1: расшифровка, разбитая на фразы, у каждой
время начала и конца, текст с пунктуацией и имя говорящего («Говорящий N»).

Границы фразы ставятся ровно по трём правилам, и верхнего предела длины нет:

  paragraph — предложение начинает новый абзац punctuated/1;
  speaker   — у слова другой говорящий, чем у предыдущего слова;
  pause     — между концом предыдущего слова и началом текущего не меньше
              --pause секунд (по умолчанию 1 с, решение владельца).

Правила проверяются в этом порядке: граница абзаца сильнее смены говорящего, та
сильнее паузы — одной фразе всё равно досталась бы одна причина, а порядок делает
её предсказуемой. Однословная фраза остаётся отдельной фразой: склеивать её с
соседями значило бы приписать слово чужому говорящему или чужому абзацу.

Разрез проходит по слову, а не по предложению: смена говорящего и пауза случаются
и в середине предложения («А что ты думаешь — про это?»). Поэтому поле sentences
у фразы — это диапазон задетых ею предложений, и при разрезе внутри предложения
его номер встречается у двух соседних фраз. Текст при этом не теряется и не
дублируется: фразы идут подряд по словам и вместе дают исходный текст слово в
слово. Исключение — невыровненное предложение (см. ниже): внутри него разрезов
не делается.

Как слова punctuated/1 сопоставляются со словами транскрипта: позиционно, по
нормализованной форме (нижний регистр, ё→е, только буквы и цифры — то же правило
и та же функция, что в tools/punctuate.py). punctuate.py сам сверяет слова с
транскриптом и не имеет права их менять, поэтому k-е слово предложения — это k-е
слово транскрипта. Но полагаться на это слепо нельзя: откатившееся окно, правка
файла руками или склейка файлов от разных прогонов ломают соответствие. Если
нормализованные последовательности разошлись, предложение целиком становится
одной фразой со своими собственными start/end — выдуманных по чужим словам
таймкодов внутри него не появится, случай считается в stats.misaligned_sentences,
печатается предупреждение, и код возврата становится 1.

Ролей инструмент не знает: ни «Преподавателя», ни «Студента» здесь нет, только
имена говорящих из диаризации. Роли назначает tools/roles.py, markdown собирает
tools/render_doc.py — состав полей speakers/1 для них контракт.

Зависимости: стандартная библиотека плюс три соседних модуля (outputs для разбора
транскрипта, punctuate для нормализации слов, render_doc для разбора punctuated/1
— это единственный в проекте разборщик этой схемы, и дублировать его значило бы
развести требования к одному и тому же файлу). Ни pyannote, ни claude, ни ffmpeg,
ни веса моделей здесь не нужны. Проверки живут в самом файле (--selftest): pytest
в окружении проекта нет.

Использование:
  python tools/speakers.py --transcript <t.json> --punctuated <p.json> \\
      --diarization <d.json> --out <файл.json> [--pause 1.0]
  python tools/speakers.py --selftest
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tempfile
from bisect import bisect_right
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

# outputs.py, punctuate.py и render_doc.py лежат рядом, а tools/ — не пакет: при
# запуске «python tools/speakers.py» каталог скрипта уже первый в sys.path, но при
# импорте модуля откуда-то ещё — нет. Поэтому путь добавляется явно.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import outputs  # noqa: E402 — импорт возможен только после правки sys.path выше
import punctuate  # noqa: E402 — то же самое
import render_doc  # noqa: E402 — то же самое

PROG = "speakers.py"

# Схема файла на выходе. Это контракт с tools/roles.py, tools/render_doc.py и
# tools/summarize.py: набор полей менять нельзя.
SCHEMA = "speakers/1"

# Схемы входов — константы соседних инструментов, чтобы требования не разъехались.
TRANSCRIPT_SCHEMA = outputs.SCHEMA
PUNCTUATED_SCHEMA = render_doc.SCHEMA
DIARIZATION_SCHEMA = "diarization/1"

# Порог паузы по умолчанию. Решение владельца: «давай порог сделаем больше, хотя
# бы 1 секунду» — в устной речи пауза короче уже не граница фразы, а дыхание.
DEFAULT_PAUSE = 1.0

# Насколько далеко от слова разрешено искать интервал говорящего, когда слово не
# перекрывается ни с одним. Полсекунды — это типичный зазор между репликой и
# границей интервала диаризации; дальше начинается угадывание.
NEAR_SECONDS = 0.5

# Сколько знаков после запятой остаётся у секунд и долей в файле. Транскрипт
# приходит с точностью до миллисекунды, выдумывать точнее нечего.
ROUND_TO = 3

# Причины границ фразы. Первые три — те самые три правила, start — начало записи.
REASON_PARAGRAPH = "paragraph"
REASON_SPEAKER = "speaker"
REASON_PAUSE = "pause"
REASON_START = "start"
SPLIT_REASONS = (REASON_PARAGRAPH, REASON_SPEAKER, REASON_PAUSE)

# Шаблон имени говорящего — такой же, как в diarization/1. Нужен, когда в turns
# пришёл ярлык, которого нет в списке speakers: имя для него приходится выдать.
SPEAKER_NAME = "Говорящий %d"

# Предупреждения соседних разборщиков должны выглядеть так, будто их написал
# работающий инструмент, а не библиотека, из которой он их позвал. Приём взят у
# tools/punctuate.py (там то же сделано с outputs).
outputs.PROG = PROG
render_doc.PROG = PROG


class UsageError(Exception):
    """Ошибка в аргументах командной строки."""


class InputError(Exception):
    """Вход непригоден: нет файла, не та схема, битый JSON, нет предложений."""


def die(message: str, code: int = 2) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main().

    По умолчанию 2: непригодный вход и ошибки аргументов означают «файл не
    собран», а код 1 здесь занят другим смыслом — файл записан, но в нём есть
    фразы без говорящего или невыровненные предложения.
    """
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return code


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но пользователь должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


HELP = """Использование:
  python tools/speakers.py --transcript <файл.json> --punctuated <файл.json> \\
      --diarization <файл.json> --out <файл.json> [опции]
  python tools/speakers.py --selftest

Соединяет три источника — пословные таймкоды транскрипта (transcript/1),
предложения с пунктуацией и абзацами (punctuated/1) и интервалы говорящих
(diarization/1) — и пишет файл схемы speakers/1: расшифровку, разбитую на фразы.
У каждой фразы время начала и конца, текст с пунктуацией и имя говорящего
(«Говорящий N»).

Границы фразы ставятся ровно по трём правилам: граница абзаца punctuated/1,
смена говорящего и пауза между соседними словами не меньше --pause. Верхнего
предела длины фразы нет; однословная фраза остаётся отдельной фразой. Ролей
(«Преподаватель», «Студент») инструмент не знает — это дело tools/roles.py.

Опции:
  --transcript <файл>    JSON схемы transcript/1 от tools/transcribe.py
                         (обязателен; из него берутся пословные таймкоды)
  --punctuated <файл>    JSON схемы punctuated/1 от tools/punctuate.py
                         (обязателен; из него берутся текст, предложения, абзацы)
  --diarization <файл>   JSON схемы diarization/1 от tools/diarize.py
                         (обязателен; из него берутся интервалы говорящих)
  --out <файл>           куда записать результат (обязателен)
  --pause <секунды>      порог паузы между соседними словами; пауза ровно в порог
                         уже разрывает фразу; по умолчанию %(pause)s
  --selftest             прогнать встроенные проверки и выйти
  -h, --help             показать эту справку

Каталог для --out создаётся при необходимости. Файл пишется в UTF-8 без
эскейпов, с отступом 2; секунды и доли округлены до %(round)d знаков.

Коды возврата:
  0  файл записан: все фразы получили говорящего, все предложения выровнялись
  1  файл записан, но есть фразы без говорящего или невыровненные предложения
  2  вход непригоден (нет файла, не та схема, битый JSON, нет предложений) —
     --out не создаётся, существующий файл не перетирается

Примеры:
  python tools/speakers.py --transcript data/output/лекция/transcript.json \\
      --punctuated data/output/лекция/punctuated.json \\
      --diarization data/output/лекция/diarization.json \\
      --out data/output/лекция/speakers.json
  python tools/speakers.py --transcript t.json --punctuated p.json \\
      --diarization d.json --out s.json --pause 1.5
""" % {"pause": DEFAULT_PAUSE, "round": ROUND_TO}


# --------------------------------------------------------------- структуры ---


@dataclass
class Speaker:
    """Говорящий диаризации: имя для человека и ярлык модели."""

    name: str
    label: str


@dataclass
class Turn:
    """Интервал речи одного говорящего в секундах от начала записи."""

    index: int
    start: float
    end: float
    speaker: Speaker


@dataclass
class Diarization:
    """То, что инструменту нужно от diarization/1: говорящие и их интервалы."""

    path: str
    speakers: list[Speaker]
    turns: list[Turn]


@dataclass
class Word:
    """Слово транскрипта: текст, границы в секундах и говорящий после привязки."""

    raw: str
    norm: str
    start: float
    end: float
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SentenceUnit:
    """Предложение punctuated/1 после сопоставления со словами транскрипта.

    pieces — по одному куску текста на слово: само слово и прилипшие к нему
    токены без букв и цифр (одинокое тире, кавычки). Склейка кусков через пробел
    даёт исходный текст предложения, а разрез по словам не теряет пунктуацию.
    """

    index: int
    paragraph: int
    start: float
    end: float
    text: str
    pieces: list[str]
    words: list[Word]
    aligned: bool
    fallback: bool

    @property
    def atomic(self) -> bool:
        """Предложение, внутри которого разрезов не делается.

        Либо слова не выровнялись (тогда времени у них нет), либо слов в нём
        нет вовсе (предложение из одного тире). И там и там единственные
        честные таймкоды — собственные start/end предложения.
        """
        return not self.aligned or not self.words


@dataclass
class Phrase:
    """Фраза итогового файла — то, что станет одной строкой документа."""

    reason: str
    paragraph: int
    first_sentence: int
    last_sentence: int
    start: float
    end: float
    pieces: list[str] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    fallback: bool = False
    speaker: str | None = None
    atomic: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.pieces)


@dataclass
class Report:
    """Чем закончилась сборка: это же уходит в stats, в отчёт и в код возврата."""

    phrases: int = 0
    sentences: int = 0
    paragraphs: int = 0
    unattributed: int = 0
    misaligned: int = 0
    splits: dict[str, int] = field(default_factory=lambda: {key: 0 for key in SPLIT_REASONS})
    speakers: list[dict] = field(default_factory=list)

    @property
    def flawed(self) -> bool:
        """Файл записан, но прогон вышел неполным."""
        return bool(self.unattributed or self.misaligned)


# ------------------------------------------------------------ чтение входа ---


def _typename(value: object) -> str:
    """Имя типа по-русски — для сообщений об ошибках."""
    return render_doc._typename(value)


def _read_json(path: str, what: str) -> object:
    """Читает JSON-файл. Любая беда на входе — InputError, не трейсбек."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise InputError("файл %s не найден: %s" % (what, path)) from None
    except IsADirectoryError:
        raise InputError("вместо файла %s указан каталог: %s" % (what, path)) from None
    except PermissionError:
        raise InputError("файл %s недоступен для чтения: %s" % (what, path)) from None
    except UnicodeDecodeError:
        raise InputError("файл %s не в кодировке UTF-8: %s" % (what, path)) from None
    except json.JSONDecodeError as exc:
        raise InputError(
            "файл %s не разбирается как JSON: %s (строка %d, символ %d: %s)"
            % (what, path, exc.lineno, exc.colno, exc.msg)
        ) from None
    except OSError as exc:
        raise InputError(
            "не удалось прочитать файл %s: %s (%s)" % (what, path, exc.strerror or exc)
        ) from None


def load_transcript(path: str) -> tuple[list[Word], float | None]:
    """Читает транскрипт и выкладывает его слова в один плоский список.

    Структуру проверяет tools/outputs.parse_transcript, а слова сегмента берёт
    tools/punctuate.segment_words — тот же код, которым пользовался punctuate.py,
    когда расставлял таймкоды предложениям. Значит и времена, и состав слов, и
    нормализация здесь ровно те же, с какими сверялась пунктуация: у сегмента без
    пословных таймкодов слова получают границы самого сегмента, слова без букв и
    цифр выпадают.
    """
    data = _read_json(path, "транскрипта")
    try:
        transcript = outputs.parse_transcript(data, source=path)
    except outputs.TranscriptError as exc:
        raise InputError(str(exc)) from None

    words: list[Word] = []
    for segment in transcript.segments:
        for word in punctuate.segment_words(segment):
            words.append(Word(word.raw, word.norm, word.start, word.end))
    if not words:
        # Это не повод падать: ниже каждое предложение окажется невыровненным,
        # фразы соберутся по предложениям, и код возврата будет 1.
        warn("в транскрипте нет ни одного слова — паузы и говорящие посчитаны не будут")
    return words, transcript.source_duration


def load_punctuated(path: str) -> render_doc.Punctuated:
    """Читает punctuated/1 разборщиком tools/render_doc.py.

    Своего разбора здесь нет сознательно: render_doc.py — единственный в проекте
    читатель этой схемы, и два разных набора требований к одному файлу
    разъехались бы при первой же правке punctuate.py.
    """
    try:
        return render_doc.load_punctuated(path)
    except render_doc.DocumentError as exc:
        raise InputError(str(exc)) from None


def _as_seconds(holder: dict, key: str, where: str) -> float:
    """Достаёт из словаря обязательное число секунд, иначе — понятная ошибка."""
    if key not in holder:
        raise InputError("%s: нет обязательного поля %s" % (where, key))
    value = holder[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(
            "%s: поле %s должно быть числом секунд, получено: %s"
            % (where, key, _typename(value))
        )
    if not math.isfinite(float(value)):
        raise InputError("%s: поле %s не является конечным числом" % (where, key))
    if float(value) < 0:
        raise InputError(
            "%s: поле %s меньше нуля (%s) — время считается от начала записи"
            % (where, key, value)
        )
    return float(value)


def _parse_speakers(data: dict) -> tuple[list[Speaker], dict[str, Speaker]]:
    """Разбирает список говорящих и строит поиск по имени и по ярлыку.

    Список не обязателен: diarization/1 его пишет всегда, но turns сами по себе
    достаточны — имена тогда выдаются по порядку появления ярлыков.
    """
    raw_speakers = data.get("speakers")
    if raw_speakers is None:
        warn("в диаризации нет списка speakers — имена говорящих выданы по ярлыкам turns")
        return [], {}
    if not isinstance(raw_speakers, list):
        raise InputError(
            "поле speakers должно быть списком, получено: %s" % _typename(raw_speakers)
        )

    speakers: list[Speaker] = []
    index: dict[str, Speaker] = {}
    for position, raw in enumerate(raw_speakers):
        where = "говорящий №%d" % (position + 1)
        if not isinstance(raw, dict):
            raise InputError(
                "%s должен быть объектом JSON, получено: %s" % (where, _typename(raw))
            )
        name = raw.get("name")
        label = raw.get("label")
        if not isinstance(name, str) or not name.strip():
            raise InputError("%s: поле name должно быть непустой строкой" % where)
        if not isinstance(label, str):
            label = ""
        speaker = Speaker(name=name.strip(), label=label.strip())
        speakers.append(speaker)
        # Имя сильнее ярлыка: в turns встречается и то и другое, а имя — то, что
        # уйдёт в документ, и подменять его ярлыком-тёзкой нельзя.
        index.setdefault(speaker.label, speaker)
        index[speaker.name] = speaker
    index.pop("", None)
    return speakers, index


def _resolve_speaker(
    raw: str, index: dict[str, Speaker], speakers: list[Speaker], unknown: set[str]
) -> Speaker:
    """Строка speaker из turns -> говорящий. Неизвестной выдаётся своё имя.

    В turns лежит либо имя («Говорящий 1»), либо ярлык модели («SPEAKER_00») —
    план схемы допускает и то и другое, поэтому принимается и то и другое. Если
    строка не нашлась ни среди имён, ни среди ярлыков, говорящий добавляется с
    первым свободным номером: потерять его интервалы хуже, чем выдать имя.
    """
    found = index.get(raw)
    if found is not None:
        return found

    if raw not in unknown:
        unknown.add(raw)
        warn(
            "в turns говорящий %r, которого нет в списке speakers — "
            "ему выдано собственное имя" % raw
        )
    taken = {speaker.name for speaker in speakers}
    number = len(speakers) + 1
    while SPEAKER_NAME % number in taken:
        number += 1
    speaker = Speaker(name=SPEAKER_NAME % number, label=raw)
    speakers.append(speaker)
    index[raw] = speaker
    return speaker


def parse_diarization(data: object, path: str = "<память>") -> Diarization:
    """Проверяет структуру JSON схемы diarization/1 и приводит её к Diarization.

    Проверяется только то, без чего привязка невозможна: схема, список turns, их
    границы и говорящий. Перекрытие интервалов ошибкой не считается — оно
    разрешается правилом «максимум перекрытия по длительности».
    """
    if not isinstance(data, dict):
        raise InputError(
            "на верхнем уровне диаризации ожидается объект JSON, получено: %s"
            % _typename(data)
        )
    if "schema" not in data:
        raise InputError("в диаризации нет поля schema (ожидается %r)" % DIARIZATION_SCHEMA)
    if data["schema"] != DIARIZATION_SCHEMA:
        raise InputError(
            "не та схема диаризации: ожидается %r, в файле %r"
            % (DIARIZATION_SCHEMA, data["schema"])
        )

    speakers, index = _parse_speakers(data)

    if "turns" not in data:
        raise InputError("в диаризации нет поля turns")
    raw_turns = data["turns"]
    if not isinstance(raw_turns, list):
        raise InputError(
            "поле turns должно быть списком, получено: %s" % _typename(raw_turns)
        )
    if not raw_turns:
        # Пустой список — это запись, в которой диаризация никого не нашла. Файл
        # собрать можно, но все фразы останутся без говорящего и код будет 1.
        warn("в диаризации нет ни одного интервала — фразы останутся без говорящего")

    unknown: set[str] = set()
    turns: list[Turn] = []
    for position, raw in enumerate(raw_turns):
        where = "интервал №%d" % (position + 1)
        if not isinstance(raw, dict):
            raise InputError(
                "%s должен быть объектом JSON, получено: %s" % (where, _typename(raw))
            )
        number = raw.get("index")
        if isinstance(number, int) and not isinstance(number, bool):
            where = "интервал index=%d" % number
        else:
            number = position

        start = _as_seconds(raw, "start", where)
        end = _as_seconds(raw, "end", where)
        name = raw.get("speaker")
        if not isinstance(name, str) or not name.strip():
            raise InputError("%s: поле speaker должно быть непустой строкой" % where)

        turns.append(
            Turn(
                index=number,
                start=start,
                end=max(start, end),
                speaker=_resolve_speaker(name.strip(), index, speakers, unknown),
            )
        )

    # Интервалы в файле отсортированы по (start, end), но полагаться на это нельзя:
    # поиск перекрытия ищет кандидатов двоичным поиском по start.
    turns.sort(key=lambda turn: (turn.start, turn.end))
    return Diarization(path=path, speakers=speakers, turns=turns)


def load_diarization(path: str) -> Diarization:
    """Читает diarization.json. Любая беда на входе — InputError, не трейсбек."""
    return parse_diarization(_read_json(path, "диаризации"), path=path)


# ------------------------------------------------- привязка говорящих ---


class Attribution:
    """Поиск говорящего по интервалу времени.

    Правило: берётся тот говорящий, чей интервал перекрывает слово по
    длительности сильнее всех. Если перекрытия нет ни с одним (слово попало в
    молчание между интервалами), берётся ближайший интервал в пределах
    NEAR_SECONDS. Если и такого нет, ответ None — выше по стеку слово наследует
    говорящего предыдущего слова.

    При равном перекрытии и равном зазоре выигрывает интервал, который начался
    раньше: выбор должен быть один и тот же при каждом прогоне.
    """

    def __init__(self, turns: list[Turn], near: float = NEAR_SECONDS) -> None:
        self.turns = turns
        self.near = near
        self.starts = [turn.start for turn in turns]
        # Самый длинный интервал задаёт, насколько далеко влево от слова имеет
        # смысл идти: интервал, начавшийся раньше, до слова уже не дотянется.
        self.longest = max((turn.end - turn.start for turn in turns), default=0.0)

    def speaker(self, start: float, end: float) -> str | None:
        """Имя говорящего для интервала [start, end] или None."""
        if not self.turns:
            return None
        end = max(start, end)
        limit = start - self.near - self.longest
        high = bisect_right(self.starts, end + self.near)
        low = high
        while low > 0 and self.turns[low - 1].start >= limit:
            low -= 1

        best: str | None = None
        best_overlap = 0.0
        nearest: str | None = None
        nearest_gap: float | None = None
        for turn in self.turns[low:high]:
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = turn.speaker.name
            gap = max(turn.start - end, start - turn.end, 0.0)
            if gap <= self.near and (nearest_gap is None or gap < nearest_gap):
                nearest_gap = gap
                nearest = turn.speaker.name
        return best if best is not None else nearest


def attribute_words(words: list[Word], attribution: Attribution) -> None:
    """Проставляет говорящего каждому слову, наследуя его при неудаче.

    Наследуется именно предыдущее слово, а не следующее: наследование вперёд
    задним числом переписало бы уже отданного говорящего, а слова в начале
    записи, до первого интервала, честнее оставить без говорящего.
    """
    previous: str | None = None
    for word in words:
        name = attribution.speaker(word.start, word.end)
        if name is None:
            name = previous
        word.speaker = name
        if name is not None:
            previous = name


def dominant_speaker(words: list[Word]) -> str | None:
    """Говорящий фразы: у кого больше суммарная длительность её слов.

    При равной длительности выигрывает тот, чьё слово встретилось раньше, —
    по той же причине, по которой детерминирован выбор интервала.
    """
    totals: dict[str, float] = {}
    order: list[str] = []
    for word in words:
        if word.speaker is None:
            continue
        if word.speaker not in totals:
            totals[word.speaker] = 0.0
            order.append(word.speaker)
        totals[word.speaker] += word.duration
    if not totals:
        return None
    return max(order, key=lambda name: (totals[name], -order.index(name)))


# ------------------------------------------- сопоставление слов и текста ---


def split_pieces(text: str) -> tuple[list[str], list[str]]:
    """Текст предложения -> (куски по одному на слово, нормализованные формы).

    Токен, от которого после нормализации ничего не осталось (одинокое тире,
    кавычки), в сверке не участвует — его нет и в словах транскрипта. Но из
    текста он выпасть не должен, поэтому прилипает к следующему слову, а в конце
    предложения — к предыдущему.
    """
    pieces: list[str] = []
    norms: list[str] = []
    pending: list[str] = []
    for token in text.split():
        norm = punctuate.normalize_word(token)
        if not norm:
            pending.append(token)
            continue
        pieces.append(" ".join(pending + [token]))
        norms.append(norm)
        pending = []
    if pending:
        tail = " ".join(pending)
        if pieces:
            pieces[-1] = "%s %s" % (pieces[-1], tail)
        else:
            # Предложение без единого слова: кусок есть, слова нет. Такое
            # предложение целиком становится одной фразой.
            pieces.append(tail)
    return pieces, norms


def align_sentences(
    punctuated: render_doc.Punctuated, words: list[Word]
) -> tuple[list[list[SentenceUnit]], int]:
    """Сопоставляет слова предложений со словами транскрипта позиционно.

    Возвращает предложения, разложенные по абзацам, и число невыровненных.
    Указатель по словам транскрипта двигается на длину предложения даже тогда,
    когда предложение не выровнялось: сопоставление позиционное, и подгонять его
    поиском по тексту — значит выдумывать соответствие, которого в файлах нет.
    """
    units: list[list[SentenceUnit]] = []
    misaligned = 0
    position = 0

    for paragraph_number, paragraph in enumerate(punctuated.paragraphs):
        block: list[SentenceUnit] = []
        for sentence in paragraph.sentences:
            pieces, norms = split_pieces(sentence.text)
            taken = words[position:position + len(norms)]
            position += len(norms)

            aligned = len(taken) == len(norms) and all(
                word.norm == norm for word, norm in zip(taken, norms)
            )
            if norms and not aligned:
                misaligned += 1
                warn(
                    "предложение %d не сошлось со словами транскрипта (%s) — "
                    "оно целиком стало одной фразой со своими таймкодами"
                    % (sentence.index, _mismatch_detail(taken, norms, pieces))
                )

            block.append(
                SentenceUnit(
                    index=sentence.index,
                    paragraph=paragraph_number,
                    start=sentence.start,
                    end=max(sentence.start, sentence.end),
                    text=sentence.text,
                    pieces=pieces,
                    words=list(taken) if aligned else [],
                    aligned=aligned,
                    fallback=sentence.fallback,
                )
            )
        if block:
            units.append(block)

    left = len(words) - position
    if left > 0:
        warn(
            "%d слов транскрипта не вошли ни в одно предложение — "
            "punctuated собран не от этого транскрипта" % left
        )
    return units, misaligned


def _mismatch_detail(taken: list[Word], norms: list[str], pieces: list[str]) -> str:
    """Первое расхождение словами: «позиция 3: «созвон» вместо «разговор»»."""
    for position, (word, norm) in enumerate(zip(taken, norms), start=1):
        if word.norm != norm:
            return "позиция %d: %r вместо %r" % (position, pieces[position - 1], word.raw)
    return "слов в предложении %d, а в транскрипте осталось %d" % (len(norms), len(taken))


# -------------------------------------------------------- разбивка на фразы ---


def split_reason(
    *,
    paragraph_start: bool,
    previous_speaker: str | None,
    speaker: str | None,
    previous_end: float | None,
    start: float,
    pause: float,
) -> str | None:
    """Чем новая фраза отделена от предыдущей; None — граница не нужна.

    Порядок проверок задаёт и приоритет причин: абзац сильнее смены говорящего,
    та сильнее паузы. Неизвестный говорящий (None) сменой не считается: слово без
    атрибуции бывает только в начале записи, до первого интервала диаризации, и
    рвать по нему фразу незачем.
    """
    if previous_end is None:
        return REASON_START
    if paragraph_start:
        return REASON_PARAGRAPH
    if previous_speaker is not None and speaker is not None and previous_speaker != speaker:
        return REASON_SPEAKER
    if start - previous_end >= pause:
        return REASON_PAUSE
    return None


def build_phrases(
    units: list[list[SentenceUnit]], attribution: Attribution, pause: float
) -> list[Phrase]:
    """Собирает фразы из предложений по трём правилам.

    Фраза растёт по словам, пока ни одно правило не сработало: верхнего предела
    длины нет. Однословная фраза не склеивается с соседями — она закрылась
    потому, что сработало правило, и игнорировать его значило бы отдать слово
    чужому говорящему или чужому абзацу.
    """
    phrases: list[Phrase] = []
    current: Phrase | None = None
    previous_end: float | None = None
    previous_speaker: str | None = None

    def reason_for(paragraph_start: bool, speaker: str | None, start: float) -> str:
        """Причина границы там, где фраза обязана начаться заново."""
        found = split_reason(
            paragraph_start=paragraph_start,
            previous_speaker=previous_speaker,
            speaker=speaker,
            previous_end=previous_end,
            start=start,
            pause=pause,
        )
        # Невыровненное предложение закрывает фразу само, без правил. Причина
        # всё равно нужна, и из четырёх возможных подходит pause: соседние фразы
        # разделены именно временем — предложением со своими таймкодами.
        return found or REASON_PAUSE

    for block in units:
        for position, unit in enumerate(block):
            paragraph_start = position == 0

            if unit.atomic:
                speaker = attribution.speaker(unit.start, unit.end) or previous_speaker
                phrases.append(
                    Phrase(
                        reason=reason_for(paragraph_start, speaker, unit.start),
                        paragraph=unit.paragraph,
                        first_sentence=unit.index,
                        last_sentence=unit.index,
                        start=unit.start,
                        end=unit.end,
                        pieces=[unit.text],
                        fallback=unit.fallback,
                        speaker=speaker,
                        atomic=True,
                    )
                )
                current = None
                previous_end = unit.end
                if speaker is not None:
                    previous_speaker = speaker
                continue

            for number, word in enumerate(unit.words):
                at_paragraph = paragraph_start and number == 0
                reason = (
                    reason_for(at_paragraph, word.speaker, word.start)
                    if current is None
                    else split_reason(
                        paragraph_start=at_paragraph,
                        previous_speaker=previous_speaker,
                        speaker=word.speaker,
                        previous_end=previous_end,
                        start=word.start,
                        pause=pause,
                    )
                )
                if current is None or reason is not None:
                    current = Phrase(
                        reason=reason or REASON_START,
                        paragraph=unit.paragraph,
                        first_sentence=unit.index,
                        last_sentence=unit.index,
                        start=word.start,
                        end=word.end,
                    )
                    phrases.append(current)

                current.pieces.append(unit.pieces[number])
                current.words.append(word)
                current.last_sentence = unit.index
                current.end = max(current.end, word.end)
                current.fallback = current.fallback or unit.fallback

                previous_end = word.end
                if word.speaker is not None:
                    previous_speaker = word.speaker

    for phrase in phrases:
        if not phrase.atomic:
            phrase.speaker = dominant_speaker(phrase.words)
    return phrases


def clamp_phrases(phrases: list[Phrase], duration: float | None) -> None:
    """Загоняет таймкоды фраз внутрь записи и не даёт им убывать.

    Источник таких правок один — немонотонный транскрипт (у GigaAM такое бывает
    на стыке чанков). Молча отдать документ, в котором таймкоды прыгают назад,
    нельзя, поэтому фраза подтягивается к предыдущей, а про число правок
    говорится предупреждением.
    """
    limit = duration if duration and duration > 0 else None
    previous = 0.0
    shifted = 0
    for phrase in phrases:
        start = max(0.0, phrase.start if limit is None else min(phrase.start, limit))
        end = max(0.0, phrase.end if limit is None else min(phrase.end, limit))
        if start < previous:
            start = previous
            shifted += 1
        end = max(end, start)
        phrase.start, phrase.end = start, end
        previous = start
    if shifted:
        warn(
            "таймкоды %d фраз подтянуты вперёд: в транскрипте слова идут не по "
            "возрастанию времени" % shifted
        )


# ------------------------------------------------------------ сборка файла ---


def build_speakers(
    phrases: list[Phrase], diarization: Diarization
) -> list[dict]:
    """Состав говорящих по фактической атрибуции, а не по списку диаризации.

    В файл идут только те, кому что-то досталось: говорящий, чьи интервалы не
    задели ни одного слова, в расшифровке не участвует. speech_seconds — сумма
    длительностей его слов; у невыровненного предложения слов нет, и за него
    берётся длительность фразы целиком. share — доля от всей приписанной речи.
    """
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    order: list[str] = []

    def touch(name: str) -> None:
        if name not in seconds:
            seconds[name] = 0.0
            counts[name] = 0
            order.append(name)

    for phrase in phrases:
        if phrase.speaker is not None:
            touch(phrase.speaker)
            counts[phrase.speaker] += 1
        if phrase.words:
            for word in phrase.words:
                if word.speaker is not None:
                    touch(word.speaker)
                    seconds[word.speaker] += word.duration
        elif phrase.speaker is not None:
            seconds[phrase.speaker] += max(0.0, phrase.end - phrase.start)

    total = sum(seconds.values())
    labels = {speaker.name: speaker.label for speaker in diarization.speakers}
    # Порядок — как в диаризации: имена «Говорящий 1», «Говорящий 2» должны идти
    # по номерам, а не по тому, кто первым заговорил. Имена, которых в списке
    # диаризации нет, дописываются в конце по порядку появления.
    names = [speaker.name for speaker in diarization.speakers if speaker.name in seconds]
    names += [name for name in order if name not in labels]

    return [
        {
            "name": name,
            "label": labels.get(name, ""),
            "phrases": counts[name],
            "speech_seconds": round(seconds[name], ROUND_TO),
            "share": round(seconds[name] / total, ROUND_TO) if total > 0 else 0.0,
        }
        for name in names
    ]


def build_payload(
    phrases: list[Phrase],
    *,
    punctuated: render_doc.Punctuated,
    diarization: Diarization,
    transcript_path: str,
    punctuated_path: str,
    pause: float,
    misaligned: int,
) -> tuple[dict, Report]:
    """Собирает объект схемы speakers/1 и отчёт по нему."""
    report = Report(
        phrases=len(phrases),
        sentences=len(punctuated.sentences),
        paragraphs=len(punctuated.paragraphs),
        misaligned=misaligned,
    )

    phrases_out: list[dict] = []
    for number, phrase in enumerate(phrases):
        if phrase.reason in report.splits:
            report.splits[phrase.reason] += 1
        if phrase.speaker is None:
            report.unattributed += 1
        phrases_out.append(
            {
                "index": number,
                "start": round(phrase.start, ROUND_TO),
                "end": round(phrase.end, ROUND_TO),
                "text": phrase.text,
                "speaker": phrase.speaker,
                "sentences": [phrase.first_sentence, phrase.last_sentence],
                "paragraph": phrase.paragraph,
                "fallback": phrase.fallback,
                "reason": phrase.reason,
            }
        )

    report.speakers = build_speakers(phrases, diarization)

    payload = {
        "schema": SCHEMA,
        "transcript": transcript_path,
        "punctuated": punctuated_path,
        "diarization": diarization.path,
        "pause": round(pause, ROUND_TO),
        "speakers": report.speakers,
        "phrases": phrases_out,
        "stats": {
            "phrases": report.phrases,
            "sentences": report.sentences,
            "paragraphs": report.paragraphs,
            "speakers": len(report.speakers),
            "unattributed_phrases": report.unattributed,
            "misaligned_sentences": report.misaligned,
            "splits": {key: report.splits[key] for key in SPLIT_REASONS},
        },
    }
    return payload, report


def write_payload(path: str, payload: dict) -> None:
    """Пишет speakers.json: UTF-8 без эскейпов, отступ 2, перевод строки в конце."""
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


def print_report(path: str, report: Report, pause: float) -> None:
    """Краткий отчёт в stdout — он же попадает в лог прогона."""
    sys.stdout.write("================ ГОВОРЯЩИЕ ================\n")
    sys.stdout.write("Фраз:              %d\n" % report.phrases)
    sys.stdout.write("Предложений:       %d\n" % report.sentences)
    sys.stdout.write("Абзацев:           %d\n" % report.paragraphs)
    sys.stdout.write("Говорящих:         %d\n" % len(report.speakers))
    for speaker in report.speakers:
        sys.stdout.write(
            "  * %s%s: фраз %d, речи %.3f с (%.1f %%)\n"
            % (
                speaker["name"],
                " (%s)" % speaker["label"] if speaker["label"] else "",
                speaker["phrases"],
                speaker["speech_seconds"],
                speaker["share"] * 100.0,
            )
        )
    sys.stdout.write("Без говорящего:    %d\n" % report.unattributed)
    sys.stdout.write("Не выровнено:      %d\n" % report.misaligned)
    sys.stdout.write(
        "Разрезы:           абзац %d, говорящий %d, пауза %d\n"
        % (
            report.splits[REASON_PARAGRAPH],
            report.splits[REASON_SPEAKER],
            report.splits[REASON_PAUSE],
        )
    )
    sys.stdout.write("Порог паузы:       %s с\n" % pause)
    sys.stdout.write("Файл:              %s\n" % path)


# --------------------------------------------------------- разбор аргументов ---

_FLAGS_VALUE = {
    "--transcript": "transcript",
    "--punctuated": "punctuated",
    "--diarization": "diarization",
    "--out": "out",
    "--pause": "pause",
}


def _as_float(flag: str, value: str) -> float:
    """Значение флага -> неотрицательное конечное число секунд."""
    try:
        number = float(value)
    except ValueError:
        raise UsageError("%s требует число секунд, получено: %r" % (flag, value)) from None
    if not math.isfinite(number):
        raise UsageError("%s требует конечное число секунд, получено: %r" % (flag, value))
    if number < 0:
        raise UsageError("%s не может быть отрицательным, получено: %r" % (flag, value))
    return number


def parse_args(argv: list[str]) -> dict:
    """Разбирает аргументы так же, как это делают остальные инструменты проекта.

    Своим разбором, а не argparse: тот печатает usage и свои ошибки по-английски,
    а весь остальной вывод инструментов проекта — на русском.
    """
    opts: dict = {
        "transcript": None,
        "punctuated": None,
        "diarization": None,
        "out": None,
        "pause": DEFAULT_PAUSE,
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
            if token == "--pause":
                opts["pause"] = _as_float(token, value)
            else:
                opts[_FLAGS_VALUE[token]] = value
            continue

        if token.startswith("-"):
            raise UsageError("неизвестная опция: %s (см. --help)" % token)
        raise UsageError("лишний аргумент: %s (см. --help)" % token)

    return opts


def check_args(opts: dict) -> None:
    """Проверяет набор аргументов целиком — то, что нельзя проверить по одному."""
    for flag, key in (
        ("--transcript", "transcript"),
        ("--punctuated", "punctuated"),
        ("--diarization", "diarization"),
        ("--out", "out"),
    ):
        if not opts[key]:
            raise UsageError("не указан %s (см. --help)" % flag)

    out = os.path.abspath(opts["out"])
    # Перетереть собственный вход — это потерять прогон целиком: ни транскрипт,
    # ни пунктуацию, ни диаризацию заново не собрать без часов работы.
    for flag, key in (
        ("--transcript", "transcript"),
        ("--punctuated", "punctuated"),
        ("--diarization", "diarization"),
    ):
        if out == os.path.abspath(opts[key]):
            raise UsageError(
                "--out указывает на тот же файл, что и %s: %s" % (flag, opts["out"])
            )


# ---------------------------------------------------------------- селф-тест ---

# Фикстура: абзацы -> предложения -> (текст с пунктуацией, слова с временами).
# Из неё собираются все три входных файла, поэтому последовательности слов в них
# согласованы по построению, как их согласует настоящий прогон punctuate.py.
#
# Что в неё заложено по критериям приёмки:
#   - пауза 1.2 с внутри абзаца (4.8 -> 6.0) — разрез по правилу pause;
#   - смена говорящего в середине предложения (9.9 -> 10.5) — разрез по speaker;
#   - граница абзаца при паузе 0.5 с (11.0 -> 11.5) — разрез по paragraph;
#   - однословное предложение «Хорошо.» между двумя паузами — фраза из одного слова;
#   - «вернёмся» в пунктуации против «вернемся» в транскрипте — проверка ё→е;
#   - слова в молчании между интервалами (22.0–24.0) — наследование говорящего;
#   - граница абзаца и пауза одновременно (20.0 -> 22.0) — приоритет paragraph.
_FIXTURE: list[list[tuple[str, list[tuple[str, float, float]]]]] = [
    [
        (
            "Итак, начинаем наш созвон.",
            [("итак", 0.5, 1.0), ("начинаем", 1.1, 1.6), ("наш", 1.7, 2.0), ("созвон", 2.1, 2.6)],
        ),
        (
            "Сегодня обсуждаем пунктуацию.",
            [("сегодня", 2.8, 3.2), ("обсуждаем", 3.3, 3.9), ("пунктуацию", 4.0, 4.8)],
        ),
        (
            "Потом вернёмся к таймкодам.",
            [("потом", 6.0, 6.4), ("вернемся", 6.5, 7.1), ("к", 7.2, 7.3), ("таймкодам", 7.4, 8.2)],
        ),
        (
            "А что ты думаешь про это?",
            [
                ("а", 9.0, 9.1),
                ("что", 9.2, 9.4),
                ("ты", 9.5, 9.6),
                ("думаешь", 9.7, 9.9),
                ("про", 10.5, 10.7),
                ("это", 10.8, 11.0),
            ],
        ),
    ],
    [
        (
            "Ну давайте начнём.",
            [("ну", 11.5, 12.0), ("давайте", 12.1, 12.6), ("начнем", 12.7, 13.2)],
        ),
        ("Да.", [("да", 13.3, 13.5)]),
        ("Хорошо.", [("хорошо", 16.0, 16.5)]),
        (
            "Теперь по порядку.",
            [("теперь", 19.0, 19.4), ("по", 19.5, 19.6), ("порядку", 19.7, 20.0)],
        ),
    ],
    [
        (
            "Это пауза в диаризации.",
            [("это", 22.0, 22.4), ("пауза", 22.5, 23.0), ("в", 23.1, 23.2), ("диаризации", 23.3, 24.0)],
        ),
        (
            "И снова первый.",
            [("и", 26.0, 26.2), ("снова", 26.3, 26.8), ("первый", 26.9, 27.5)],
        ),
    ],
]

# Как предложения фикстуры разложены по сегментам транскрипта (чанкам нарезки).
_FIXTURE_SEGMENTS = ((0, 1, 2), (3, 4, 5), (6, 7), (8, 9))
_FIXTURE_DURATION = 31.0

# Интервалы диаризации фикстуры: два говорящих, между вторым и третьим
# интервалом молчание 20.0–25.0 — там и живёт наследование говорящего.
_FIXTURE_TURNS = (
    (0.0, 10.0, "SPEAKER_00"),
    (10.0, 20.0, "SPEAKER_01"),
    (25.0, 30.0, "SPEAKER_00"),
)
_FIXTURE_NAMES = {"SPEAKER_00": "Говорящий 1", "SPEAKER_01": "Говорящий 2"}

# Что должно получиться из фикстуры: (причина, говорящий, предложения, текст).
_EXPECTED = (
    (REASON_START, "Говорящий 1", [0, 1], "Итак, начинаем наш созвон. Сегодня обсуждаем пунктуацию."),
    (REASON_PAUSE, "Говорящий 1", [2, 3], "Потом вернёмся к таймкодам. А что ты думаешь"),
    (REASON_SPEAKER, "Говорящий 2", [3, 3], "про это?"),
    (REASON_PARAGRAPH, "Говорящий 2", [4, 5], "Ну давайте начнём. Да."),
    (REASON_PAUSE, "Говорящий 2", [6, 6], "Хорошо."),
    (REASON_PAUSE, "Говорящий 2", [7, 7], "Теперь по порядку."),
    (REASON_PARAGRAPH, "Говорящий 2", [8, 8], "Это пауза в диаризации."),
    (REASON_SPEAKER, "Говорящий 1", [9, 9], "И снова первый."),
)

EXPECTED_KEYS = [
    "schema",
    "transcript",
    "punctuated",
    "diarization",
    "pause",
    "speakers",
    "phrases",
    "stats",
]
PHRASE_KEYS = [
    "index",
    "start",
    "end",
    "text",
    "speaker",
    "sentences",
    "paragraph",
    "fallback",
    "reason",
]
SPEAKER_KEYS = ["name", "label", "phrases", "speech_seconds", "share"]
STATS_KEYS = [
    "phrases",
    "sentences",
    "paragraphs",
    "speakers",
    "unattributed_phrases",
    "misaligned_sentences",
    "splits",
]


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fixture_sentences() -> list[tuple[int, int, str, list[tuple[str, float, float]]]]:
    """Плоский список предложений фикстуры: (номер, абзац, текст, слова)."""
    flat = []
    for paragraph_number, paragraph in enumerate(_FIXTURE):
        for text, words in paragraph:
            flat.append((len(flat), paragraph_number, text, words))
    return flat


def _fixture_transcript(*, without_words: tuple[int, ...] = ()) -> dict:
    """Транскрипт схемы transcript/1 из фикстуры.

    without_words — номера сегментов, у которых пословные таймкоды убраны: так
    выглядит короткий или пустой чанк, и слова такого сегмента получают его
    границы. Последовательность слов при этом сохраняется (она есть в text).
    """
    flat = _fixture_sentences()
    segments = []
    for number, sentence_numbers in enumerate(_FIXTURE_SEGMENTS):
        words = [word for index in sentence_numbers for word in flat[index][3]]
        segment = {
            "index": number,
            "start": round(words[0][1], ROUND_TO),
            "end": round(words[-1][2], ROUND_TO),
            "text": " ".join(word[0] for word in words),
        }
        if number not in without_words:
            segment["words"] = [
                {"text": raw, "start": round(start, ROUND_TO), "end": round(end, ROUND_TO)}
                for raw, start, end in words
            ]
        segments.append(segment)
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "source": "data/output/созвон/созвон.webm",
        "source_duration": _FIXTURE_DURATION,
        "model": "v3_rnnt",
        "segments": segments,
    }


def _fixture_punctuated(*, texts: dict[int, str] | None = None, fallback: tuple[int, ...] = ()) -> dict:
    """Файл схемы punctuated/1 из фикстуры; texts подменяет текст предложений."""
    texts = texts or {}
    flat = _fixture_sentences()
    sentences = []
    paragraphs: list[dict] = []
    for index, paragraph_number, text, words in flat:
        if not paragraphs or paragraphs[-1]["window"] != paragraph_number:
            paragraphs.append(
                {"index": len(paragraphs), "window": paragraph_number, "sentences": [index, index]}
            )
        paragraphs[-1]["sentences"][1] = index
        sentences.append(
            {
                "index": index,
                "start": round(words[0][1], ROUND_TO),
                "end": round(words[-1][2], ROUND_TO),
                "text": texts.get(index, text),
                "window": paragraph_number,
                "fallback": index in fallback,
            }
        )
    return {
        "schema": PUNCTUATED_SCHEMA,
        "source": "data/output/созвон/созвон.webm",
        "source_duration": _FIXTURE_DURATION,
        "transcript": "data/output/созвон/transcript.json",
        "model": "v3_rnnt",
        "windows": [
            {"index": number, "segments": [number, number], "status": "ok", "reason": ""}
            for number in range(len(_FIXTURE))
        ],
        "paragraphs": paragraphs,
        "sentences": sentences,
        "text": "\n\n".join(
            " ".join(item["text"] for item in sentences if item["window"] == number)
            for number in range(len(_FIXTURE))
        ),
        "stats": {"sentences": len(sentences), "paragraphs": len(paragraphs)},
    }


def _fixture_diarization(
    *,
    turns: tuple[tuple[float, float, str], ...] = _FIXTURE_TURNS,
    by_label: bool = False,
    speakers: bool = True,
) -> dict:
    """Файл схемы diarization/1 из фикстуры.

    by_label — в turns стоят ярлыки модели, а не имена: план схемы допускает и
    то и другое, и инструмент обязан понимать оба написания.
    """
    labels = sorted({label for _, _, label in turns})
    payload: dict = {
        "schema": DIARIZATION_SCHEMA,
        "source": "data/output/созвон/audio.wav",
        "source_duration": _FIXTURE_DURATION,
        "model": "pyannote/speaker-diarization-3.1",
        "device": "cpu",
        "turns": [
            {
                "index": number,
                "start": start,
                "end": end,
                "speaker": label if by_label else _FIXTURE_NAMES[label],
            }
            for number, (start, end, label) in enumerate(turns)
        ],
        "stats": {"turns": len(turns), "speakers": len(labels)},
    }
    if speakers:
        payload["speakers"] = [
            {
                "name": _FIXTURE_NAMES[label],
                "label": label,
                "turns": sum(1 for _, _, item in turns if item == label),
                "speech_seconds": round(
                    sum(end - start for start, end, item in turns if item == label), ROUND_TO
                ),
                "share": 0.5,
            }
            for label in labels
        ]
        # Ключи в diarization/1 идут в своём порядке; speakers перед turns.
        payload = {
            key: payload[key]
            for key in ("schema", "source", "source_duration", "model", "device", "speakers", "turns", "stats")
        }
    return payload


def _write_json(path: str, payload: object) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _read_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Сквозной прогон main() с перехватом вывода."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _inputs(workdir: str, **kwargs: object) -> list[str]:
    """Пишет три входных файла фикстуры и отдаёт аргументы для main()."""
    transcript = _write_json(
        os.path.join(workdir, "transcript.json"),
        kwargs.get("transcript") or _fixture_transcript(),
    )
    punctuated = _write_json(
        os.path.join(workdir, "punctuated.json"),
        kwargs.get("punctuated") or _fixture_punctuated(),
    )
    diarization = _write_json(
        os.path.join(workdir, "diarization.json"),
        kwargs.get("diarization") or _fixture_diarization(),
    )
    return [
        "--transcript", transcript,
        "--punctuated", punctuated,
        "--diarization", diarization,
    ]


def _rounded(value: object) -> bool:
    """Число округлено до ROUND_TO знаков (или это вообще не число)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return abs(float(value) - round(float(value), ROUND_TO)) < 1e-12


def _validate_payload(payload: dict, punctuated: dict, *, duration: float | None) -> None:
    """Проверяет обещания схемы speakers/1, не зная содержания фикстуры."""
    _check(list(payload) == EXPECTED_KEYS, "поля файла не те: %r" % list(payload))
    _check(payload["schema"] == SCHEMA, "не та схема: %r" % payload["schema"])
    _check(_rounded(payload["pause"]), "pause округлён не до %d знаков" % ROUND_TO)

    phrases = payload["phrases"]
    _check(isinstance(phrases, list) and phrases, "список фраз пуст")
    stats = payload["stats"]
    _check(list(stats) == STATS_KEYS, "поля stats не те: %r" % list(stats))
    _check(list(stats["splits"]) == list(SPLIT_REASONS), "splits не по трём правилам: %r" % stats["splits"])

    previous_start = -1.0
    unattributed = 0
    for number, phrase in enumerate(phrases):
        where = "фраза %d" % number
        _check(list(phrase) == PHRASE_KEYS, "%s: поля не те: %r" % (where, list(phrase)))
        _check(phrase["index"] == number, "%s: index не по порядку" % where)
        _check(_rounded(phrase["start"]) and _rounded(phrase["end"]), "%s: таймкод не округлён" % where)
        _check(phrase["start"] >= previous_start, "%s: таймкод пошёл назад" % where)
        _check(phrase["end"] >= phrase["start"], "%s: конец раньше начала" % where)
        _check(phrase["start"] >= 0.0, "%s: таймкод меньше нуля" % where)
        if duration is not None:
            _check(phrase["end"] <= duration + 1e-9, "%s: таймкод вышел за конец записи" % where)
        previous_start = phrase["start"]

        _check(phrase["text"].strip() == phrase["text"] and phrase["text"], "%s: текст пуст или с пробелами" % where)
        _check(phrase["reason"] in SPLIT_REASONS + (REASON_START,), "%s: причина %r" % (where, phrase["reason"]))
        _check((phrase["reason"] == REASON_START) == (number == 0), "%s: start не у первой фразы" % where)
        _check(isinstance(phrase["fallback"], bool), "%s: fallback не логическое значение" % where)
        first, last = phrase["sentences"]
        _check(isinstance(first, int) and isinstance(last, int) and first <= last, "%s: границы предложений %r" % (where, phrase["sentences"]))
        if phrase["speaker"] is None:
            unattributed += 1
        else:
            _check(isinstance(phrase["speaker"], str) and phrase["speaker"], "%s: говорящий не строка" % where)

    # Фразы идут подряд по предложениям: следующая начинается там же, где
    # закончилась предыдущая (разрез внутри предложения), или на следующем.
    _check(phrases[0]["sentences"][0] == punctuated["sentences"][0]["index"], "первая фраза начинается не с первого предложения")
    _check(phrases[-1]["sentences"][1] == punctuated["sentences"][-1]["index"], "последняя фраза кончается не последним предложением")
    for previous, phrase in zip(phrases, phrases[1:]):
        step = phrase["sentences"][0] - previous["sentences"][1]
        _check(step in (0, 1), "между фразами %d и %d пропуск или наложение предложений" % (previous["index"], phrase["index"]))
        if step == 0:
            _check(
                previous["sentences"][1] == phrase["sentences"][0] == previous["sentences"][1],
                "разрез внутри предложения оформлен не тем диапазоном",
            )

    # Текст фраз — это текст предложений слово в слово, без пропусков и повторов.
    _check(
        " ".join(phrase["text"] for phrase in phrases)
        == " ".join(item["text"] for item in punctuated["sentences"]),
        "текст фраз не совпал с текстом предложений",
    )

    _check(stats["phrases"] == len(phrases), "stats.phrases врёт")
    _check(stats["sentences"] == len(punctuated["sentences"]), "stats.sentences врёт")
    _check(stats["paragraphs"] == len(punctuated["paragraphs"]), "stats.paragraphs врёт")
    _check(stats["unattributed_phrases"] == unattributed, "stats.unattributed_phrases врёт")
    _check(
        sum(stats["splits"].values()) + 1 == stats["phrases"],
        "разрезов %d при %d фразах" % (sum(stats["splits"].values()), stats["phrases"]),
    )

    names = [phrase["speaker"] for phrase in phrases if phrase["speaker"] is not None]
    _check(stats["speakers"] == len(payload["speakers"]), "stats.speakers врёт")
    _check(len(payload["speakers"]) == len(set(names)), "в составе говорящих не те, кто получил фразы")
    total = 0.0
    for speaker in payload["speakers"]:
        _check(list(speaker) == SPEAKER_KEYS, "поля говорящего не те: %r" % list(speaker))
        _check(speaker["name"] in set(names), "говорящий %r не получил ни одной фразы" % speaker["name"])
        _check(speaker["phrases"] == names.count(speaker["name"]), "у %r не то число фраз" % speaker["name"])
        _check(_rounded(speaker["speech_seconds"]) and _rounded(speaker["share"]), "числа говорящего не округлены")
        total += speaker["share"]
    if payload["speakers"]:
        _check(abs(total - 1.0) < 0.01, "доли говорящих в сумме дают %s" % total)


def _test_attribution() -> None:
    """Правило привязки слова к говорящему: перекрытие, близость, наследование."""
    speakers = [Speaker("Говорящий 1", "SPEAKER_00"), Speaker("Говорящий 2", "SPEAKER_01")]
    turns = [
        Turn(0, 0.0, 5.0, speakers[0]),
        Turn(1, 4.0, 10.0, speakers[1]),   # перекрывается с первым
        Turn(2, 20.0, 21.0, speakers[0]),
    ]
    attribution = Attribution(turns)

    # Максимум перекрытия: слово 4.2–4.9 целиком в обоих, 4.5–6.0 — больше во втором.
    _check(attribution.speaker(1.0, 2.0) == "Говорящий 1", "слово внутри первого интервала")
    _check(attribution.speaker(4.5, 6.0) == "Говорящий 2", "перекрытие больше у второго интервала")
    _check(attribution.speaker(3.0, 4.5) == "Говорящий 1", "перекрытие больше у первого интервала")
    # Равное перекрытие — выигрывает начавшийся раньше, и ответ не меняется.
    _check(attribution.speaker(4.0, 5.0) == "Говорящий 1", "равное перекрытие решено не в пользу раннего")
    # Слово нулевой длительности внутри интервала: перекрытия нет, близость есть.
    _check(attribution.speaker(7.0, 7.0) == "Говорящий 2", "слово нулевой длины внутри интервала")
    # Молчание: в пределах 0,5 с берётся ближайший, дальше — никто.
    _check(attribution.speaker(10.3, 10.4) == "Говорящий 2", "ближайший интервал в пределах 0,5 с")
    _check(attribution.speaker(19.6, 19.8) == "Говорящий 1", "ближайший интервал справа")
    _check(attribution.speaker(14.0, 15.0) is None, "слово в молчании получило говорящего")
    _check(Attribution([]).speaker(1.0, 2.0) is None, "без интервалов нашёлся говорящий")

    # Наследование: слово без атрибуции берёт говорящего предыдущего слова, а в
    # начале записи остаётся без него.
    words = [
        Word("раз", "раз", 14.0, 14.5),
        Word("два", "два", 1.0, 2.0),
        Word("три", "три", 15.0, 15.5),
    ]
    attribute_words(words, attribution)
    _check(words[0].speaker is None, "первое слово унаследовало несуществующего говорящего")
    _check(words[1].speaker == "Говорящий 1", "второе слово привязано неверно")
    _check(words[2].speaker == "Говорящий 1", "третье слово не унаследовало говорящего")

    # Говорящий фразы — у кого больше суммарная длительность её слов.
    phrase_words = [
        Word("а", "а", 0.0, 0.2, "Говорящий 1"),
        Word("б", "б", 0.2, 1.4, "Говорящий 2"),
        Word("в", "в", 1.4, 1.5, None),
    ]
    _check(dominant_speaker(phrase_words) == "Говорящий 2", "говорящий фразы не по длительности")
    _check(dominant_speaker([Word("а", "а", 0.0, 0.1)]) is None, "фраза без атрибуции получила говорящего")


def _test_successful_run() -> None:
    """Обычный прогон: три правила, однословная фраза, код 0."""
    with tempfile.TemporaryDirectory(prefix="speakers-selftest-") as workdir:
        argv = _inputs(workdir)
        out = os.path.join(workdir, "итог", "speakers.json")
        code, report, stderr = _run(argv + ["--out", out])
        _check(code == 0, "обычный прогон вернул код %d (stderr: %s)" % (code, stderr.strip()))
        _check(not stderr.strip(), "при чистом входе есть вывод в stderr: %s" % stderr.strip())
        _check(os.path.isfile(out), "файл не записан: %s" % out)

        payload = _read_payload(out)
        punctuated = _fixture_punctuated()
        _validate_payload(payload, punctuated, duration=_FIXTURE_DURATION)

        phrases = payload["phrases"]
        _check(len(phrases) == len(_EXPECTED), "фраз %d, ожидалось %d" % (len(phrases), len(_EXPECTED)))
        for number, (reason, speaker, sentences, text) in enumerate(_EXPECTED):
            phrase = phrases[number]
            _check(phrase["reason"] == reason, "фраза %d: причина %r, ожидалась %r" % (number, phrase["reason"], reason))
            _check(phrase["speaker"] == speaker, "фраза %d: говорящий %r, ожидался %r" % (number, phrase["speaker"], speaker))
            _check(phrase["sentences"] == sentences, "фраза %d: предложения %r, ожидались %r" % (number, phrase["sentences"], sentences))
            _check(phrase["text"] == text, "фраза %d: текст %r, ожидался %r" % (number, phrase["text"], text))
            _check(phrase["fallback"] is False, "фраза %d помечена откатом" % number)

        # Таймкоды фраз — это времена слов транскрипта, а не предложений.
        _check(phrases[1]["start"] == 6.0 and phrases[1]["end"] == 9.9, "фраза 1: таймкоды %r" % [phrases[1]["start"], phrases[1]["end"]])
        _check(phrases[2]["start"] == 10.5 and phrases[2]["end"] == 11.0, "фраза 2: таймкоды %r" % [phrases[2]["start"], phrases[2]["end"]])
        # Однословная фраза осталась отдельной фразой.
        _check(phrases[4]["text"] == "Хорошо." and len(phrases[4]["text"].split()) == 1, "однословная фраза склеилась")

        stats = payload["stats"]
        _check(stats["splits"] == {"paragraph": 2, "speaker": 2, "pause": 3}, "разрезы по правилам: %r" % stats["splits"])
        _check(stats["misaligned_sentences"] == 0, "предложения не выровнялись на чистой фикстуре")
        _check(stats["unattributed_phrases"] == 0, "есть фразы без говорящего")
        _check(stats["sentences"] == 10 and stats["paragraphs"] == 3, "не то число предложений или абзацев: %r" % stats)
        _check(payload["pause"] == DEFAULT_PAUSE, "порог паузы в файле: %r" % payload["pause"])

        # Состав говорящих: порядок как в диаризации, доли считаются по словам.
        names = [speaker["name"] for speaker in payload["speakers"]]
        _check(names == ["Говорящий 1", "Говорящий 2"], "порядок говорящих: %r" % names)
        labels = [speaker["label"] for speaker in payload["speakers"]]
        _check(labels == ["SPEAKER_00", "SPEAKER_01"], "ярлыки говорящих: %r" % labels)
        _check([speaker["phrases"] for speaker in payload["speakers"]] == [3, 5], "фразы по говорящим: %r" % payload["speakers"])
        first = payload["speakers"][0]["speech_seconds"]
        expected_first = sum(
            end - start
            for index, (_, _, _, words) in enumerate(_fixture_sentences())
            for raw, start, end in words
            if (index in (0, 1, 2, 9)) or (index == 3 and raw in ("а", "что", "ты", "думаешь"))
        )
        _check(abs(first - expected_first) < 1e-6, "речь первого говорящего %s, ожидалось %s" % (first, expected_first))

        # Пути входов сохранены как есть: по ним файл потом найдут соседние тулы.
        _check(payload["transcript"] == argv[1], "в файле не тот путь к транскрипту")
        _check(payload["punctuated"] == argv[3], "в файле не тот путь к пунктуации")
        _check(payload["diarization"] == argv[5], "в файле не тот путь к диаризации")

        # Отчёт в stdout — по-русски и со всеми числами.
        for needle in ("Фраз:              8", "Предложений:       10", "Абзацев:           3", "Говорящих:         2", "Без говорящего:    0", "абзац 2, говорящий 2, пауза 3", "Говорящий 1 (SPEAKER_00)"):
            _check(needle in report, "в отчёте нет строки %r" % needle)
        _check(out in report, "в отчёте нет пути к файлу")

        # Ярлыки вместо имён в turns дают тот же результат.
        other = os.path.join(workdir, "by-label.json")
        argv2 = _inputs(
            os.path.join(workdir, ""),
            diarization=_fixture_diarization(by_label=True),
        )
        code, _, stderr = _run(argv2 + ["--out", other])
        _check(code == 0, "прогон с ярлыками в turns вернул код %d (%s)" % (code, stderr.strip()))
        by_label = _read_payload(other)
        _check(by_label["phrases"] == payload["phrases"], "ярлыки в turns дали другие фразы")


def _test_pause_threshold() -> None:
    """Порог паузы: граница ставится при пауза >= --pause, и ни при чём ином."""
    with tempfile.TemporaryDirectory(prefix="speakers-selftest-") as workdir:
        argv = _inputs(workdir)

        # Самая маленькая пауза внутри абзаца — 0.1 с, самая большая — 2.5 с.
        counts = {}
        for pause in ("0.05", "1.0", "1.2", "1.21", "2.5", "100"):
            out = os.path.join(workdir, "pause-%s.json" % pause)
            code, _, stderr = _run(argv + ["--out", out, "--pause", pause])
            _check(code == 0, "порог %s вернул код %d (%s)" % (pause, code, stderr.strip()))
            payload = _read_payload(out)
            _validate_payload(payload, _fixture_punctuated(), duration=_FIXTURE_DURATION)
            counts[pause] = payload["stats"]
            _check(payload["pause"] == float(pause), "порог в файле %r при --pause %s" % (payload["pause"], pause))

        # Пауза ровно в порог уже разрывает фразу: 1.2 с рвёт при 1.2 и не рвёт
        # при 1.21.
        _check(counts["1.2"]["splits"]["pause"] == 3, "при пороге 1.2 разрезов по паузе %d" % counts["1.2"]["splits"]["pause"])
        _check(counts["1.21"]["splits"]["pause"] == 2, "при пороге 1.21 разрезов по паузе %d" % counts["1.21"]["splits"]["pause"])
        # Крошечный порог рвёт по любой паузе, огромный — ни по одной.
        _check(counts["0.05"]["splits"]["pause"] > counts["1.0"]["splits"]["pause"], "мелкий порог не добавил разрезов")
        _check(counts["100"]["splits"]["pause"] == 0, "при пороге 100 с остались разрезы по паузе")
        # Правила абзаца и говорящего от порога не зависят.
        for pause, stats in counts.items():
            _check(stats["splits"]["paragraph"] == 2, "порог %s изменил разрезы по абзацам" % pause)
            _check(stats["splits"]["speaker"] == 2, "порог %s изменил разрезы по говорящим" % pause)


def _test_misaligned_sentence() -> None:
    """Слова разошлись: предложение целиком одной фразой, код 1, предупреждение."""
    with tempfile.TemporaryDirectory(prefix="speakers-selftest-") as workdir:
        # Подменено одно слово в середине предложения 3.
        changed = _fixture_punctuated(texts={3: "А что ты думаешь про него?"})
        argv = _inputs(workdir, punctuated=changed)
        out = os.path.join(workdir, "changed.json")
        code, report, stderr = _run(argv + ["--out", out])
        _check(code == 1, "невыровненное предложение вернуло код %d, ожидался 1" % code)
        _check("предложение 3" in stderr and "не сошлось" in stderr, "нет предупреждения о расхождении: %s" % stderr)

        payload = _read_payload(out)
        _validate_payload(payload, changed, duration=_FIXTURE_DURATION)
        _check(payload["stats"]["misaligned_sentences"] == 1, "невыровненных предложений %d" % payload["stats"]["misaligned_sentences"])

        # Внутри предложения 3 разрезов нет: оно целиком одна фраза со своими
        # таймкодами, хотя говорящий внутри него менялся.
        own = [phrase for phrase in payload["phrases"] if phrase["sentences"] == [3, 3]]
        _check(len(own) == 1, "предложение 3 разрезано на %d фраз" % len(own))
        _check(own[0]["text"] == "А что ты думаешь про него?", "текст невыровненного предложения изменён: %r" % own[0]["text"])
        _check(own[0]["start"] == 9.0 and own[0]["end"] == 11.0, "таймкоды взяты не у предложения: %r" % own[0])
        _check(own[0]["speaker"] == "Говорящий 1", "говорящий невыровненного предложения: %r" % own[0]["speaker"])
        _check("Не выровнено:      1" in report, "в отчёте нет числа невыровненных предложений")

        # Слов в предложении стало больше, чем осталось в транскрипте.
        longer = _fixture_punctuated(texts={9: "И снова первый, точно первый."})
        code, _, stderr = _run(
            _inputs(workdir, punctuated=longer) + ["--out", os.path.join(workdir, "longer.json")]
        )
        _check(code == 1, "лишнее слово в последнем предложении вернуло код %d" % code)
        _check("предложение 9" in stderr, "про предложение 9 ничего не сказано: %s" % stderr)

        # Слов стало меньше: предложения сошлись, но часть транскрипта осталась.
        shorter = _fixture_punctuated(texts={9: "И снова."})
        code, _, stderr = _run(
            _inputs(workdir, punctuated=shorter) + ["--out", os.path.join(workdir, "shorter.json")]
        )
        _check("не вошли ни в одно предложение" in stderr, "про лишние слова транскрипта не сказано: %s" % stderr)
        _check(code == 0, "лишние слова транскрипта дали код %d" % code)


def _test_unattributed() -> None:
    """Фразы без говорящего: код 1, speaker null, состав говорящих не врёт."""
    with tempfile.TemporaryDirectory(prefix="speakers-selftest-") as workdir:
        # Диаризация никого не нашла: все фразы без говорящего.
        empty = _fixture_diarization(turns=())
        empty["speakers"] = []
        argv = _inputs(workdir, diarization=empty)
        out = os.path.join(workdir, "empty.json")
        code, report, stderr = _run(argv + ["--out", out])
        _check(code == 1, "пустая диаризация вернула код %d, ожидался 1" % code)
        _check("ни одного интервала" in stderr, "про пустую диаризацию не предупредили: %s" % stderr)

        payload = _read_payload(out)
        _validate_payload(payload, _fixture_punctuated(), duration=_FIXTURE_DURATION)
        _check(payload["speakers"] == [], "без интервалов нашлись говорящие: %r" % payload["speakers"])
        _check(all(phrase["speaker"] is None for phrase in payload["phrases"]), "без интервалов кому-то достались фразы")
        _check(payload["stats"]["unattributed_phrases"] == payload["stats"]["phrases"], "не все фразы без говорящего")
        _check(payload["stats"]["splits"]["speaker"] == 0, "без говорящих появились разрезы по смене говорящего")
        _check("Без говорящего:    %d" % payload["stats"]["phrases"] in report, "в отчёте не то число фраз без говорящего")

        # Диаризация началась поздно: первые слова без говорящего, остальные с ним.
        late = _fixture_diarization(turns=((15.0, 30.0, "SPEAKER_01"),))
        code, _, _ = _run(
            _inputs(workdir, diarization=late) + ["--out", os.path.join(workdir, "late.json")]
        )
        _check(code == 1, "поздняя диаризация вернула код %d" % code)
        payload = _read_payload(os.path.join(workdir, "late.json"))
        _validate_payload(payload, _fixture_punctuated(), duration=_FIXTURE_DURATION)
        kinds = {phrase["speaker"] is None for phrase in payload["phrases"]}
        _check(kinds == {True, False}, "ожидались фразы и с говорящим, и без: %r" % kinds)
        _check(payload["phrases"][0]["speaker"] is None, "первая фраза получила говорящего задним числом")
        _check(payload["speakers"][0]["name"] == "Говорящий 2", "в составе не тот говорящий: %r" % payload["speakers"])


def _test_degraded_input() -> None:
    """Вход неполный, но работать можно: сегмент без слов, чужой ярлык, откат."""
    with tempfile.TemporaryDirectory(prefix="speakers-selftest-") as workdir:
        # У сегмента 1 нет пословных таймкодов: его слова получают границы
        # сегмента, пауз внутри них нет, и фраз становится меньше.
        argv = _inputs(workdir, transcript=_fixture_transcript(without_words=(1,)))
        out = os.path.join(workdir, "no-words.json")
        code, _, stderr = _run(argv + ["--out", out])
        payload = _read_payload(out)
        _validate_payload(payload, _fixture_punctuated(), duration=_FIXTURE_DURATION)
        _check(code == 0, "сегмент без пословных таймкодов дал код %d (%s)" % (code, stderr.strip()))
        _check(payload["stats"]["misaligned_sentences"] == 0, "слова сегмента без таймкодов не сошлись")

        # В turns ярлык, которого нет в списке speakers: имя ему выдаётся, и
        # интервалы не теряются.
        strange = _fixture_diarization(by_label=True)
        strange["speakers"] = [strange["speakers"][0]]
        code, _, stderr = _run(
            _inputs(workdir, diarization=strange) + ["--out", os.path.join(workdir, "strange.json")]
        )
        _check(code == 0, "чужой ярлык в turns дал код %d" % code)
        _check("которого нет в списке speakers" in stderr, "про чужой ярлык не предупредили: %s" % stderr)
        payload = _read_payload(os.path.join(workdir, "strange.json"))
        names = [speaker["name"] for speaker in payload["speakers"]]
        _check(names == ["Говорящий 1", "Говорящий 2"], "чужому ярлыку выдано не то имя: %r" % names)
        _check(payload["speakers"][1]["label"] == "SPEAKER_01", "ярлык потерялся: %r" % payload["speakers"][1])

        # Списка speakers нет вовсе: имена выдаются по ярлыкам turns.
        without = _fixture_diarization(by_label=True, speakers=False)
        code, _, stderr = _run(
            _inputs(workdir, diarization=without) + ["--out", os.path.join(workdir, "without.json")]
        )
        _check(code == 0, "диаризация без списка speakers дала код %d" % code)
        _check("нет списка speakers" in stderr, "про отсутствие списка speakers не предупредили")
        payload = _read_payload(os.path.join(workdir, "without.json"))
        _check(
            [speaker["name"] for speaker in payload["speakers"]] == ["Говорящий 1", "Говорящий 2"],
            "имена по ярлыкам: %r" % payload["speakers"],
        )

        # Предложения откатившегося окна помечены fallback — пометка доезжает
        # до фраз, которые их задели.
        marked = _fixture_punctuated(fallback=(6, 7))
        code, _, _ = _run(
            _inputs(workdir, punctuated=marked) + ["--out", os.path.join(workdir, "fallback.json")]
        )
        _check(code == 0, "пометка fallback изменила код возврата: %d" % code)
        payload = _read_payload(os.path.join(workdir, "fallback.json"))
        flagged = {tuple(phrase["sentences"]) for phrase in payload["phrases"] if phrase["fallback"]}
        _check(flagged == {(6, 6), (7, 7)}, "пометка fallback разъехалась: %r" % flagged)


def _test_broken_input() -> None:
    """Непригодный вход: код 2, понятное сообщение, --out не создаётся."""
    cases = [
        ("не объект", []),
        ("нет schema", {"turns": []}),
        ("чужая схема", {"schema": "diarization/2", "turns": []}),
        ("схема транскрипта", {"schema": TRANSCRIPT_SCHEMA, "turns": []}),
        ("нет turns", {"schema": DIARIZATION_SCHEMA, "speakers": []}),
        ("turns не список", {"schema": DIARIZATION_SCHEMA, "turns": {}}),
        ("speakers не список", {"schema": DIARIZATION_SCHEMA, "speakers": 5, "turns": []}),
        ("говорящий не объект", {"schema": DIARIZATION_SCHEMA, "speakers": [1], "turns": []}),
        ("говорящий без имени", {"schema": DIARIZATION_SCHEMA, "speakers": [{"label": "SPEAKER_00"}], "turns": []}),
        ("интервал не объект", {"schema": DIARIZATION_SCHEMA, "turns": ["0-5"]}),
        ("интервал без start", {"schema": DIARIZATION_SCHEMA, "turns": [{"end": 5.0, "speaker": "Говорящий 1"}]}),
        ("start не число", {"schema": DIARIZATION_SCHEMA, "turns": [{"start": "ноль", "end": 5.0, "speaker": "Говорящий 1"}]}),
        ("start меньше нуля", {"schema": DIARIZATION_SCHEMA, "turns": [{"start": -1.0, "end": 5.0, "speaker": "Говорящий 1"}]}),
        ("интервал без говорящего", {"schema": DIARIZATION_SCHEMA, "turns": [{"start": 0.0, "end": 5.0}]}),
    ]
    for name, payload in cases:
        try:
            with redirect_stderr(io.StringIO()):
                parse_diarization(payload)
        except InputError as exc:
            _check(str(exc).strip() != "", "случай %r: пустое сообщение об ошибке" % name)
        else:
            raise AssertionError("случай %r прошёл проверку структуры" % name)

    with tempfile.TemporaryDirectory(prefix="speakers-selftest-") as workdir:
        good = _inputs(workdir)
        transcript, punctuated, diarization = good[1], good[3], good[5]
        out = os.path.join(workdir, "не должен появиться", "speakers.json")

        broken = os.path.join(workdir, "broken.json")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write('{"schema": "diarization/1", "turns": [')
        missing = os.path.join(workdir, "нет.json")
        empty_sentences = _write_json(
            os.path.join(workdir, "empty.json"),
            dict(_fixture_punctuated(), sentences=[], paragraphs=[]),
        )
        alien = _write_json(os.path.join(workdir, "alien.json"), _fixture_diarization())

        cases = [
            (["--punctuated", punctuated, "--diarization", diarization, "--out", out], "без --transcript"),
            (["--transcript", transcript, "--diarization", diarization, "--out", out], "без --punctuated"),
            (["--transcript", transcript, "--punctuated", punctuated, "--out", out], "без --diarization"),
            (good + [], "без --out"),
            (good + ["--out", out, "--pause", "нет"], "--pause не число"),
            (good + ["--out", out, "--pause", "-1"], "--pause отрицательный"),
            (good + ["--out", out, "--pause", "nan"], "--pause не конечный"),
            (good + ["--out", out, "--pause"], "--pause без значения"),
            (good + ["--out", transcript], "--out поверх транскрипта"),
            (good + ["--out", punctuated], "--out поверх пунктуации"),
            (good + ["--out", diarization], "--out поверх диаризации"),
            (["--transcript", missing, "--punctuated", punctuated, "--diarization", diarization, "--out", out], "нет транскрипта"),
            (["--transcript", transcript, "--punctuated", missing, "--diarization", diarization, "--out", out], "нет пунктуации"),
            (["--transcript", transcript, "--punctuated", punctuated, "--diarization", missing, "--out", out], "нет диаризации"),
            (["--transcript", transcript, "--punctuated", punctuated, "--diarization", broken, "--out", out], "битый JSON диаризации"),
            (["--transcript", transcript, "--punctuated", empty_sentences, "--diarization", diarization, "--out", out], "пустой список предложений"),
            (["--transcript", alien, "--punctuated", punctuated, "--diarization", diarization, "--out", out], "чужая схема транскрипта"),
            (["--transcript", transcript, "--punctuated", alien, "--diarization", diarization, "--out", out], "чужая схема пунктуации"),
            (["--transcript", transcript, "--punctuated", punctuated, "--diarization", punctuated, "--out", out], "чужая схема диаризации"),
            (["--transcript", workdir, "--punctuated", punctuated, "--diarization", diarization, "--out", out], "каталог вместо транскрипта"),
            (good + ["--out", out, "--кривая-опция"], "неизвестная опция"),
            (good + ["--out", out, "лишнее"], "лишний аргумент"),
            (["--selftest=1"], "--selftest со значением"),
        ]
        for argv, what in cases:
            code, _, stderr = _run(argv)
            _check(code == 2, "случай %r завершился кодом %d, ожидался 2" % (what, code))
            _check(stderr.startswith(PROG + ":") or (PROG + ":") in stderr, "случай %r ничего не написал в stderr" % what)
            _check(not os.path.exists(out), "случай %r оставил файл --out на диске" % what)

        # Существующий файл при непригодном входе не перетирается.
        kept = os.path.join(workdir, "kept.json")
        with open(kept, "w", encoding="utf-8") as handle:
            handle.write("старое содержимое\n")
        code, _, _ = _run(
            ["--transcript", transcript, "--punctuated", punctuated, "--diarization", broken, "--out", kept]
        )
        _check(code == 2, "перетирание: код %d вместо 2" % code)
        with open(kept, "r", encoding="utf-8") as handle:
            _check(handle.read() == "старое содержимое\n", "старый файл перетёрт при ошибке входа")


def _test_help() -> None:
    """Справка — на русском, со всеми флагами, код 0."""
    for flag in ("-h", "--help"):
        code, out, _ = _run([flag])
        _check(code == 0, "%s вернул код %d" % (flag, code))
        for needle in (
            "--transcript",
            "--punctuated",
            "--diarization",
            "--out",
            "--pause",
            "--selftest",
            "Использование",
            "Коды возврата",
        ):
            _check(needle in out, "в справке (%s) нет %r" % (flag, needle))
        _check(re.search(r"[а-яА-Я]", out) is not None, "справка не на русском")


def selftest() -> int:
    """Прогоняет встроенные проверки. Код возврата 0 — все прошли."""
    tests = (
        ("привязка говорящих к словам", _test_attribution),
        ("обычный прогон и три правила", _test_successful_run),
        ("порог паузы", _test_pause_threshold),
        ("невыровненные предложения", _test_misaligned_sentence),
        ("фразы без говорящего", _test_unattributed),
        ("неполный вход", _test_degraded_input),
        ("непригодный вход и коды возврата", _test_broken_input),
        ("справка", _test_help),
    )
    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ speakers.py ================\n")
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
        words, duration = load_transcript(opts["transcript"])
        punctuated = load_punctuated(opts["punctuated"])
        diarization = load_diarization(opts["diarization"])

        attribution = Attribution(diarization.turns)
        attribute_words(words, attribution)
        units, misaligned = align_sentences(punctuated, words)
        phrases = build_phrases(units, attribution, opts["pause"])
        clamp_phrases(phrases, duration)
        if not phrases:
            # Предложения есть (иначе разбор бы не прошёл), а фраз нет — значит
            # в них нет ни слов, ни текста. Собирать нечего.
            raise InputError("из предложений не получилось ни одной фразы")

        payload, report = build_payload(
            phrases,
            punctuated=punctuated,
            diarization=diarization,
            transcript_path=opts["transcript"],
            punctuated_path=opts["punctuated"],
            pause=opts["pause"],
            misaligned=misaligned,
        )
        write_payload(opts["out"], payload)
    except (UsageError, InputError) as exc:
        return die(str(exc))

    print_report(opts["out"], report, opts["pause"])
    # Код 1 — файл записан, но прогон неполный: сборочный скрипт обязан это
    # заметить и сказать человеку.
    return 1 if report.flawed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
