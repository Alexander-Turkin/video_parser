#!/usr/bin/env python3
"""render_doc.py — сборка итогового markdown из саммари и предложений с таймкодами.

На вход — файл схемы punctuated/1 (его пишет tools/punctuate.py: предложения с
таймкодами, абзацы, окна пунктуации) и, необязательно, готовый markdown-фрагмент
саммари от tools/summarize.py. На выход — один .md, в котором ровно два раздела:

  ## Саммари      — фрагмент вставляется как есть, без переформатирования;
  ## Расшифровка  — по строке на предложение, вида «[ЧЧ:ММ:СС] Текст.».

Если переданы ещё два файла — speakers/1 (фразы с говорящими от tools/speakers.py)
и, необязательно, roles/1 (соответствие «Говорящий N → роль» от tools/roles.py), —
расшифровка собирается не по предложениям, а по фразам, и каждая строка получает
ярлык говорящего: «[00:12:34] Преподаватель: фраза». Без --speakers ничего не
меняется: документ выходит ровно таким же, как и до появления этих флагов.

Диаризация — шаг необязательный и ненадёжный, поэтому её неудача документ не
роняет: негодный speakers/1 или roles/1 даёт предупреждение в stderr, расшифровку
предложениями без ярлыков, видимую пометку в документе и код возврата 1. Тем же
кончается --speakers-note «<причина>» — им сборочный скрипт говорит, почему
ярлыков не будет, когда до диаризации дело вовсе не дошло.

Технического блока в документе нет сознательно: владелец решил, что модель, ×RT,
SHA-256 входа, версии и длительность в итоговый файл не идут — они и так лежат в
промежуточных JSON рядом с документом.

Почему таймкоды форматируются своей функцией, а не format_timestamp() из
tools/outputs.py: там формат SRT (HH:MM:SS,mmm с округлением до миллисекунд), а
здесь нужен «ЧЧ:ММ:СС» с отбрасыванием дробной части — такая строка читается
человеком и не притворяется точнее, чем есть. Файл outputs.py не правим: он занят
рендером .txt и .srt.

Почему строки предложений без двух пробелов на конце (жёсткий перенос markdown):
критерий требует строку ровно вида «[ЧЧ:ММ:СС] текст», а невидимые пробелы в
конце ломают и её, и обратный разбор файла построчно. Пометка об откате, наоборот,
отделена от своего абзаца пустой строкой: без неё markdown по правилу ленивого
продолжения затянул бы первую строку абзаца внутрь цитаты «>».

Зависимостей нет: только стандартная библиотека. Ни claude, ни torch, ни ffmpeg
здесь не нужны — инструмент только собирает текст из готовых файлов. Проверки
живут в самом файле (--selftest): в окружении проекта нет pytest.

Использование:
  python tools/render_doc.py --punctuated <файл.json> --out <файл.md> [--summary <файл.md>]
  python tools/render_doc.py --punctuated <файл.json> --speakers <файл.json> \
      --roles <файл.json> --out <файл.md>
  python tools/render_doc.py --selftest
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

PROG = "render_doc.py"
SCHEMA = "punctuated/1"

# Необязательные входы: фразы с говорящими и соответствие «говорящий -> роль».
SPEAKERS_SCHEMA = "speakers/1"
ROLES_SCHEMA = "roles/1"

# Заголовки разделов итогового документа. Больше в нём ничего нет.
TITLE_SUMMARY = "## Саммари"
TITLE_BODY = "## Расшифровка"

# Заголовок, под который уходит пометка об отсутствующем саммари: тот же, что
# первым разделом ставит tools/summarize.py, — чтобы документ выглядел одинаково
# и с саммари, и без него.
SUMMARY_FIRST_SECTION = "### Изложение"

HELP = """Использование:
  python tools/render_doc.py --punctuated <файл.json> --out <файл.md> [опции]
  python tools/render_doc.py --selftest

Собирает итоговый документ прогона: один markdown-файл с заголовком, разделом
«## Саммари» (готовый фрагмент от tools/summarize.py вставляется как есть) и
разделом «## Расшифровка», где каждое предложение — отдельная строка вида
«[ЧЧ:ММ:СС] Текст предложения.». Абзацы разделены пустой строкой. Окна, в которых
пунктуация не восстановилась, помечены видимой строкой перед своим блоком.
Ничего, кроме этих двух разделов, в документе нет.

С --speakers расшифровка собирается не по предложениям, а по фразам говорящих, и
каждая строка получает ярлык: «[ЧЧ:ММ:СС] Преподаватель: текст фразы.». Ярлык —
это роль из файла --roles, а без него — имя говорящего («Говорящий 1»); у фразы,
которой диаризация говорящего не дала, строка остаётся прежней, без ярлыка.

Опции:
  --punctuated <файл>   JSON схемы punctuated/1 от tools/punctuate.py (обязателен)
  --out <файл>          куда записать итоговый .md (обязателен)
  --summary <файл>      markdown-фрагмент саммари от tools/summarize.py;
                        если не указан, не существует или пуст — вместо саммари
                        в документ идёт пометка «не получено»
  --speakers <файл>     JSON схемы speakers/1 от tools/speakers.py: фразы с
                        говорящими. С ним расшифровка идёт фразами с ярлыками,
                        без него — предложениями, как и раньше
  --roles <файл>        JSON схемы roles/1 от tools/roles.py: соответствие
                        «Говорящий N → роль». Без него ярлыком служит имя
                        говорящего; без --speakers файл не используется
  --speakers-note <текст>
                        причина, по которой ярлыков говорящих не будет (например,
                        «диаризация не запускалась»). Ставит видимую пометку под
                        заголовком «## Расшифровка» и даёт код возврата 1
  --title <строка>      заголовок документа; по умолчанию — имя исходной записи
                        без расширения из поля source файла punctuated
  --selftest            прогнать встроенные проверки и выйти
  -h, --help            показать эту справку

Каталог для --out создаётся при необходимости, файл пишется в UTF-8.

Негодный speakers/1 или roles/1 (чужая схема, битый JSON, фразы не покрывают
предложения) документ не роняет: причина уходит в stderr, расшифровка собирается
предложениями без ярлыков, в документе появляется видимая пометка, код 1.

Коды возврата:
  0  документ записан, пометок в нём нет
  1  документ записан, но в нём есть пометки (нет саммари, окна в откате или
     не проставлены ярлыки говорящих)
  2  вход непригоден (нет файла, не та схема, нет предложений) — --out не создаётся

Примеры:
  python tools/render_doc.py --punctuated data/output/лекция/punctuated.json \\
      --summary data/output/лекция/summary.md --out "data/output/лекция-2026-10-03.md"
  python tools/render_doc.py --punctuated p.json --speakers speakers.json \\
      --roles roles.json --summary summary.md --out doc.md
  python tools/render_doc.py --punctuated p.json --out doc.md \\
      --speakers-note "диаризация не запускалась (--no-diarize)"
"""


class UsageError(Exception):
    """Ошибка в аргументах командной строки."""


class DocumentError(Exception):
    """Вход не читается, не той схемы или в нём нет нужных полей."""


class SpeakersError(Exception):
    """Негодный speakers/1 или roles/1 — ярлыков не будет, но документ будет.

    Отдельный тип, а не наследник DocumentError, именно поэтому: DocumentError в
    main() означает «документ не собран, код 2», а беда с говорящими такого права
    не имеет. Расшифровка — главное содержимое прогона, и терять её из-за
    необязательного шага нельзя.
    """


def die(message: str, code: int = 2) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main().

    По умолчанию 2: все ошибки входа и аргументов означают «документ не собран»,
    а код 1 в этом инструменте занят другим смыслом — документ записан, но с
    пометками.
    """
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return code


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но пользователь должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


# ----------------------------------------------------------------- таймкоды ---


def format_timecode(seconds: float) -> str:
    """Секунды от начала записи -> «ЧЧ:ММ:СС».

    Дробная часть отбрасывается, а не округляется: таймкод в документе — это
    метка «ищи здесь», и она не должна убегать вперёд от слова, с которого
    предложение начинается. Часы, минуты и секунды всегда двузначные; запись
    длиннее 100 часов даст три знака в часах, и это лучше, чем обрезанный час.
    """
    value = float(seconds)
    if not math.isfinite(value):
        raise ValueError("таймкод не является конечным числом: %r" % (seconds,))
    if value < 0:
        raise ValueError("таймкод не может быть отрицательным: %r" % (seconds,))
    total = int(value)  # отбрасывание дробной части, а не округление
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return "%02d:%02d:%02d" % (hours, minutes, secs)


# ------------------------------------------------------------ чтение входа ---


@dataclass
class Sentence:
    """Предложение с таймкодами начала и конца в секундах от начала записи."""

    index: int
    start: float
    end: float
    text: str
    window: int | None
    fallback: bool


@dataclass
class Paragraph:
    """Абзац: номер окна пунктуации и вошедшие в него предложения."""

    index: int
    window: int | None
    sentences: list[Sentence] = field(default_factory=list)


@dataclass
class Window:
    """Окно пунктуации: диапазон сегментов, итог сверки слов и причина отката."""

    index: int
    first_segment: int | None
    last_segment: int | None
    status: str
    reason: str

    @property
    def fallback(self) -> bool:
        return self.status == "fallback"


@dataclass
class Punctuated:
    """То, что инструменту нужно от punctuated.json.

    Остальные поля схемы (transcript, model, source_duration, text, stats) здесь
    не нужны: в документ они не попадают по решению владельца, а привязываться к
    ним ради отчёта значило бы ломаться от любой правки соседнего инструмента.
    """

    path: str
    source: str
    sentences: list[Sentence]
    paragraphs: list[Paragraph]
    windows: dict[int, Window]


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


def _as_number(holder: dict, key: str, where: str) -> float:
    """Достаёт из словаря обязательное число, иначе — понятная ошибка."""
    if key not in holder:
        raise DocumentError("%s: нет обязательного поля %s" % (where, key))
    value = holder[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DocumentError(
            "%s: поле %s должно быть числом секунд, получено: %s"
            % (where, key, _typename(value))
        )
    if not math.isfinite(float(value)):
        raise DocumentError("%s: поле %s не является конечным числом" % (where, key))
    return float(value)


def _as_text(holder: dict, key: str, where: str) -> str:
    """Достаёт из словаря обязательную строку, иначе — понятная ошибка."""
    if key not in holder:
        raise DocumentError("%s: нет обязательного поля %s" % (where, key))
    value = holder[key]
    if not isinstance(value, str):
        raise DocumentError(
            "%s: поле %s должно быть строкой, получено: %s"
            % (where, key, _typename(value))
        )
    return value


def _as_index(value: object) -> int | None:
    """Целое без логических значений — или None, если это не номер."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_windows(data: dict) -> dict[int, Window]:
    """Разбирает список окон в отображение «номер окна -> окно».

    Окна нужны ровно для одного: понять, где пунктуация не восстановилась, и
    выписать перед таким блоком пометку с диапазоном сегментов и причиной.
    Отсутствие списка фатальным не считается — тогда откаты определяются по
    флагу fallback у самих предложений, но без диапазона сегментов.
    """
    raw_windows = data.get("windows")
    if raw_windows is None:
        warn("в файле нет списка windows — пометки об откате будут без номеров сегментов")
        return {}
    if not isinstance(raw_windows, list):
        raise DocumentError(
            "поле windows должно быть списком, получено: %s" % _typename(raw_windows)
        )

    windows: dict[int, Window] = {}
    for position, raw in enumerate(raw_windows):
        where = "окно №%d" % (position + 1)
        if not isinstance(raw, dict):
            raise DocumentError(
                "%s должно быть объектом JSON, получено: %s" % (where, _typename(raw))
            )
        index = _as_index(raw.get("index"))
        if index is None:
            index = position

        first_segment = last_segment = None
        segments = raw.get("segments")
        if isinstance(segments, list) and len(segments) == 2:
            first_segment = _as_index(segments[0])
            last_segment = _as_index(segments[1])

        status = raw.get("status")
        if not isinstance(status, str):
            status = "ok"
        reason = raw.get("reason")
        if not isinstance(reason, str):
            reason = ""

        windows[index] = Window(
            index=index,
            first_segment=first_segment,
            last_segment=last_segment,
            status=status,
            reason=reason.strip(),
        )
    return windows


def _parse_sentences(data: dict) -> list[Sentence]:
    """Разбирает предложения и упорядочивает их по полю index.

    Порядок именно по index, а не по порядку в файле: index — это сквозная
    нумерация из punctuate.py, и абзацы ссылаются на предложения по ней.
    """
    if "sentences" not in data:
        raise DocumentError("в файле нет поля sentences")
    raw_sentences = data["sentences"]
    if not isinstance(raw_sentences, list):
        raise DocumentError(
            "поле sentences должно быть списком, получено: %s" % _typename(raw_sentences)
        )
    if not raw_sentences:
        raise DocumentError("список sentences пуст — расшифровку собирать не из чего")

    sentences: list[Sentence] = []
    seen: set[int] = set()
    for position, raw in enumerate(raw_sentences):
        where = "предложение №%d" % (position + 1)
        if not isinstance(raw, dict):
            raise DocumentError(
                "%s должно быть объектом JSON, получено: %s" % (where, _typename(raw))
            )
        index = _as_index(raw.get("index"))
        if index is None:
            index = position
        else:
            where = "предложение index=%d" % index
        if index in seen:
            raise DocumentError("%s: номер index повторяется" % where)
        seen.add(index)

        start = _as_number(raw, "start", where)
        end = _as_number(raw, "end", where)
        if start < 0:
            raise DocumentError(
                "%s: start меньше нуля (%s) — таймкод считается от начала записи"
                % (where, start)
            )
        text = " ".join(_as_text(raw, "text", where).split())
        if not text:
            # Пустое предложение в документе дало бы строку из одного таймкода.
            # Это не повод падать: ниже проверяется, что осталось хоть что-то.
            warn("%s: пустой текст — предложение в документ не попадёт" % where)
            continue

        sentences.append(
            Sentence(
                index=index,
                start=start,
                end=max(start, end),
                text=text,
                window=_as_index(raw.get("window")),
                fallback=bool(raw.get("fallback")),
            )
        )

    if not sentences:
        raise DocumentError("во всех предложениях пустой текст — писать нечего")
    sentences.sort(key=lambda item: item.index)
    return sentences


def _parse_paragraphs(data: dict, sentences: list[Sentence]) -> list[Paragraph]:
    """Собирает абзацы: каждый — диапазон [первый, последний] по index предложений.

    Предложение, не попавшее ни в один абзац, не теряется: такие уходят в
    дополнительный абзац в конце с предупреждением в stderr. Потерять строку
    расшифровки из-за рассогласованного файла — хуже, чем отдать лишний абзац.
    """
    by_index = {sentence.index: sentence for sentence in sentences}

    raw_paragraphs = data.get("paragraphs")
    if raw_paragraphs is not None and not isinstance(raw_paragraphs, list):
        raise DocumentError(
            "поле paragraphs должно быть списком, получено: %s" % _typename(raw_paragraphs)
        )
    if not raw_paragraphs:
        warn("в файле нет абзацев — вся расшифровка пойдёт одним абзацем")
        return [
            Paragraph(index=0, window=sentences[0].window, sentences=list(sentences))
        ]

    paragraphs: list[Paragraph] = []
    used: set[int] = set()
    for position, raw in enumerate(raw_paragraphs):
        where = "абзац №%d" % (position + 1)
        if not isinstance(raw, dict):
            raise DocumentError(
                "%s должен быть объектом JSON, получено: %s" % (where, _typename(raw))
            )
        index = _as_index(raw.get("index"))
        if index is None:
            index = position

        bounds = raw.get("sentences")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise DocumentError(
                "%s: поле sentences должно быть списком из двух номеров предложений"
                % where
            )
        first = _as_index(bounds[0])
        last = _as_index(bounds[1])
        if first is None or last is None:
            raise DocumentError("%s: границы абзаца должны быть целыми числами" % where)
        if last < first:
            raise DocumentError(
                "%s: конец диапазона предложений раньше начала (%d и %d)"
                % (where, first, last)
            )

        picked = [
            by_index[number]
            for number in range(first, last + 1)
            if number in by_index and number not in used
        ]
        used.update(sentence.index for sentence in picked)
        if not picked:
            # Абзац без предложений (все пустые или уже израсходованы) —
            # просто не попадает в документ, пустых блоков в нём нет.
            continue
        # Номер окна берётся у абзаца, а при его отсутствии — у первого
        # предложения. Проверка именно на None: окно с номером 0 существует.
        window = _as_index(raw.get("window"))
        if window is None:
            window = picked[0].window
        paragraphs.append(Paragraph(index=index, window=window, sentences=picked))

    paragraphs.sort(key=lambda item: item.index)

    orphans = [sentence for sentence in sentences if sentence.index not in used]
    if orphans:
        warn(
            "%d предложений не попали ни в один абзац — дописаны в конец расшифровки"
            % len(orphans)
        )
        next_index = (paragraphs[-1].index + 1) if paragraphs else 0
        paragraphs.append(
            Paragraph(index=next_index, window=orphans[0].window, sentences=orphans)
        )
    return paragraphs


def parse_punctuated(data: object, path: str = "<память>") -> Punctuated:
    """Проверяет структуру JSON схемы punctuated/1 и приводит её к Punctuated.

    Проверяется только то, без чего документ не собрать: схема, предложения с
    текстом и таймкодами, согласованность абзацев. Качество самой пунктуации и
    годность таймкодов — не дело этого инструмента.
    """
    if not isinstance(data, dict):
        raise DocumentError(
            "на верхнем уровне ожидается объект JSON, получено: %s" % _typename(data)
        )

    if "schema" not in data:
        raise DocumentError("в файле нет поля schema (ожидается %r)" % SCHEMA)
    if data["schema"] != SCHEMA:
        raise DocumentError(
            "не та схема: ожидается %r, в файле %r" % (SCHEMA, data["schema"])
        )

    source = data.get("source")
    if not isinstance(source, str):
        source = ""

    sentences = _parse_sentences(data)
    paragraphs = _parse_paragraphs(data, sentences)
    windows = _parse_windows(data)
    return Punctuated(
        path=path,
        source=source.strip(),
        sentences=sentences,
        paragraphs=paragraphs,
        windows=windows,
    )


def load_punctuated(path: str) -> Punctuated:
    """Читает punctuated.json. Любая беда на входе — DocumentError, не трейсбек."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise DocumentError("файл punctuated не найден: %s" % path) from None
    except IsADirectoryError:
        raise DocumentError("вместо файла punctuated указан каталог: %s" % path) from None
    except PermissionError:
        raise DocumentError("файл punctuated недоступен для чтения: %s" % path) from None
    except UnicodeDecodeError:
        raise DocumentError("файл punctuated не в кодировке UTF-8: %s" % path) from None
    except json.JSONDecodeError as exc:
        raise DocumentError(
            "файл punctuated не разбирается как JSON: %s (строка %d, символ %d: %s)"
            % (path, exc.lineno, exc.colno, exc.msg)
        ) from None
    except OSError as exc:
        raise DocumentError(
            "не удалось прочитать файл punctuated: %s (%s)" % (path, exc.strerror or exc)
        ) from None
    return parse_punctuated(data, path=path)


# ------------------------------------------------- говорящие и их ярлыки ---


@dataclass
class Phrase:
    """Фраза одного говорящего из speakers/1.

    sentences — границы [первое, последнее] по сквозной нумерации предложений
    punctuated/1: по ним фраза привязывается к окну пунктуации, а значит, и к
    пометке об откате. speaker — имя от диаризации («Говорящий 1») или None,
    если кусок записи ей не достался.
    """

    index: int
    start: float
    end: float
    text: str
    speaker: str | None
    first_sentence: int
    last_sentence: int
    paragraph: int | None


@dataclass
class Speakers:
    """То, что инструменту нужно от speakers.json: фразы по порядку и имена."""

    path: str
    names: list[str]
    phrases: list[Phrase]


def _speaker_field(holder: dict, key: str, where: str, reader) -> object:
    """Читает поле чужой схемы теми же проверками, что и поля punctuated/1.

    Разница только в типе исключения: беда в speakers/1 и roles/1 не должна
    приводить к коду 2, поэтому DocumentError переводится в SpeakersError.
    """
    try:
        return reader(holder, key, where)
    except DocumentError as exc:
        raise SpeakersError(str(exc)) from None


def _check_foreign_schema(data: object, schema: str) -> dict:
    """Общая для speakers/1 и roles/1 проверка верхнего уровня."""
    if not isinstance(data, dict):
        raise SpeakersError(
            "на верхнем уровне ожидается объект JSON, получено: %s" % _typename(data)
        )
    if "schema" not in data:
        raise SpeakersError("в файле нет поля schema (ожидается %r)" % schema)
    if data["schema"] != schema:
        raise SpeakersError(
            "не та схема: ожидается %r, в файле %r" % (schema, data["schema"])
        )
    return data


def _parse_speaker_names(data: dict) -> list[str]:
    """Имена говорящих из поля speakers. Для рендера необязательны.

    Ярлык строки берётся из самой фразы, поэтому список нужен ровно для одного:
    заметить рассогласование файла (во фразах есть тот, кого нет в списке) и
    сказать об этом человеку, не ломая документ.
    """
    entries = data.get("speakers")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise SpeakersError(
            "поле speakers должно быть списком, получено: %s" % _typename(entries)
        )
    names: list[str] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SpeakersError(
                "говорящий №%d должен быть объектом JSON, получено: %s"
                % (position + 1, _typename(entry))
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SpeakersError(
                "говорящий №%d: поле name должно быть непустой строкой" % (position + 1)
            )
        names.append(" ".join(name.split()))
    return names


def _parse_phrase_speaker(raw: object, where: str) -> str | None:
    """Имя говорящего фразы: строка, либо None — «диаризация его не дала»."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise SpeakersError(
            "%s: поле speaker должно быть строкой или null, получено: %s"
            % (where, _typename(raw))
        )
    name = " ".join(raw.split())
    return name or None


def _parse_phrase_bounds(raw: object, where: str) -> tuple[int, int]:
    """Границы фразы по предложениям punctuated/1 — [первое, последнее]."""
    if not isinstance(raw, list) or len(raw) != 2:
        raise SpeakersError(
            "%s: поле sentences должно быть списком из двух номеров предложений" % where
        )
    first = _as_index(raw[0])
    last = _as_index(raw[1])
    if first is None or last is None:
        raise SpeakersError("%s: границы фразы должны быть целыми числами" % where)
    if last < first:
        raise SpeakersError(
            "%s: конец диапазона предложений раньше начала (%d и %d)"
            % (where, first, last)
        )
    return first, last


def parse_speakers(data: object, path: str = "<память>") -> Speakers:
    """Проверяет структуру JSON схемы speakers/1 и приводит её к Speakers.

    Проверяется то, без чего ярлыки не поставить: схема, фразы с текстом,
    временем, именем говорящего и границами по предложениям. Качество самой
    диаризации — не дело этого инструмента.
    """
    data = _check_foreign_schema(data, SPEAKERS_SCHEMA)
    names = _parse_speaker_names(data)

    if "phrases" not in data:
        raise SpeakersError("в файле нет поля phrases")
    raw_phrases = data["phrases"]
    if not isinstance(raw_phrases, list):
        raise SpeakersError(
            "поле phrases должно быть списком, получено: %s" % _typename(raw_phrases)
        )
    if not raw_phrases:
        raise SpeakersError("список phrases пуст — расшифровку по фразам не собрать")

    phrases: list[Phrase] = []
    for position, raw in enumerate(raw_phrases):
        where = "фраза №%d" % (position + 1)
        if not isinstance(raw, dict):
            raise SpeakersError(
                "%s должна быть объектом JSON, получено: %s" % (where, _typename(raw))
            )
        index = _as_index(raw.get("index"))
        if index is None:
            index = position
        else:
            where = "фраза index=%d" % index

        start = float(_speaker_field(raw, "start", where, _as_number))
        if start < 0:
            raise SpeakersError(
                "%s: start меньше нуля (%s) — таймкод считается от начала записи"
                % (where, start)
            )
        end = float(_speaker_field(raw, "end", where, _as_number)) if "end" in raw else start
        text = " ".join(str(_speaker_field(raw, "text", where, _as_text)).split())
        if not text:
            # Пустую фразу нельзя просто выбросить, как пустое предложение: за
            # ней закреплены предложения punctuated/1, и их текст тогда пропал
            # бы из документа совсем. Такой вход негоден целиком.
            raise SpeakersError("%s: пустой текст — предложения фразы потерялись бы" % where)

        first, last = _parse_phrase_bounds(raw.get("sentences"), where)
        phrases.append(
            Phrase(
                index=index,
                start=start,
                end=max(start, end),
                text=text,
                speaker=_parse_phrase_speaker(raw.get("speaker"), where),
                first_sentence=first,
                last_sentence=last,
                paragraph=_as_index(raw.get("paragraph")),
            )
        )

    if names:
        unknown = sorted(
            {
                phrase.speaker
                for phrase in phrases
                if phrase.speaker is not None and phrase.speaker not in names
            }
        )
        if unknown:
            warn(
                "в phrases есть говорящие, которых нет в списке speakers: %s"
                % ", ".join(unknown)
            )
    return Speakers(path=path, names=names, phrases=phrases)


def _load_foreign_json(path: str, what: str) -> object:
    """Читает необязательный входной файл. Любая беда — SpeakersError."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SpeakersError("файл %s не найден: %s" % (what, path)) from None
    except IsADirectoryError:
        raise SpeakersError("вместо файла %s указан каталог: %s" % (what, path)) from None
    except PermissionError:
        raise SpeakersError("файл %s недоступен для чтения: %s" % (what, path)) from None
    except UnicodeDecodeError:
        raise SpeakersError("файл %s не в кодировке UTF-8: %s" % (what, path)) from None
    except json.JSONDecodeError as exc:
        raise SpeakersError(
            "файл %s не разбирается как JSON: %s (строка %d, символ %d: %s)"
            % (what, path, exc.lineno, exc.colno, exc.msg)
        ) from None
    except OSError as exc:
        raise SpeakersError(
            "не удалось прочитать файл %s: %s (%s)" % (what, path, exc.strerror or exc)
        ) from None


def load_speakers(path: str) -> Speakers:
    """Читает speakers.json схемы speakers/1.

    Имя функции зафиксировано планом: tools/summarize.py импортирует её отсюда,
    чтобы разбор speakers/1 жил в проекте в одном месте.
    """
    return parse_speakers(_load_foreign_json(path, SPEAKERS_SCHEMA), path=path)


def parse_roles(data: object, path: str = "<память>") -> dict[str, str]:
    """Проверяет структуру JSON схемы roles/1 и отдаёт «имя говорящего -> роль».

    Статус файла (ok, fallback, skip) здесь не проверяется: при fallback роли
    равны именам, при skip список пуст, и в обоих случаях ярлыки получаются
    правильные сами собой. Решение «звать ли claude ещё раз» принимает не этот
    инструмент.
    """
    data = _check_foreign_schema(data, ROLES_SCHEMA)

    if "roles" not in data:
        raise SpeakersError("в файле нет поля roles")
    entries = data["roles"]
    if not isinstance(entries, list):
        raise SpeakersError(
            "поле roles должно быть списком, получено: %s" % _typename(entries)
        )

    roles: dict[str, str] = {}
    for position, entry in enumerate(entries):
        where = "роль №%d" % (position + 1)
        if not isinstance(entry, dict):
            raise SpeakersError(
                "%s должна быть объектом JSON, получено: %s" % (where, _typename(entry))
            )
        speaker = entry.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise SpeakersError("%s: поле speaker должно быть непустой строкой" % where)
        role = entry.get("role")
        if not isinstance(role, str) or not role.strip():
            raise SpeakersError("%s: поле role должно быть непустой строкой" % where)
        speaker = " ".join(speaker.split())
        if speaker in roles:
            # Две роли у одного говорящего — непонятно, какую печатать в строке.
            raise SpeakersError("%s: говорящий %s встречается дважды" % (where, speaker))
        roles[speaker] = " ".join(role.split())
    return roles


def load_roles(path: str) -> dict[str, str]:
    """Читает roles.json схемы roles/1. Имя функции зафиксировано планом."""
    return parse_roles(_load_foreign_json(path, ROLES_SCHEMA), path=path)


def speaker_label(phrase: object, roles: dict[str, str] | None) -> str | None:
    """Ярлык строки расшифровки: роль, иначе имя говорящего, иначе None.

    None означает «ярлыка нет»: у фразы, которой диаризация говорящего не дала,
    строка остаётся прежней — «[ЧЧ:ММ:СС] текст», без двоеточия и пустого места
    перед ним.

    Фраза принимается и объектом Phrase, и словарём прямо из JSON: читателям
    speakers/1 (сейчас — tools/summarize.py) удобно звать функцию и так, и так.
    """
    raw = phrase.get("speaker") if isinstance(phrase, dict) else getattr(phrase, "speaker", None)
    if not isinstance(raw, str):
        return None
    name = " ".join(raw.split())
    if not name:
        return None
    if roles:
        role = roles.get(name)
        if role:
            return role
    return name


def read_summary(path: str | None) -> tuple[str, str | None]:
    """Читает фрагмент саммари. Возвращает (текст, что сказать в stderr).

    Пустой текст означает «саммари нет»: решение о пометке в документе
    принимается по нему, а второй элемент — только предупреждение для человека
    и None, если говорить не о чем.

    Фрагмент не переформатируется и не проверяется на внутреннюю структуру: его
    пишет tools/summarize.py, и три раздела уровня «###» — его забота. Здесь
    снимаются только пустые строки по краям, чтобы в документе не появилось
    двух пустых строк подряд.

    Нечитаемый фрагмент (нет прав, не UTF-8) приравнивается к отсутствующему:
    расшифровку из-за него терять нельзя, а пометка в документе и код возврата 1
    про проблему скажут.
    """
    if path is None:
        return "", "файл с саммари не передан (--summary)"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except FileNotFoundError:
        # Про отсутствующий файл в stderr не пишем: это ожидаемый исход прогона,
        # в котором claude не ответил, и он уже виден по пометке и коду 1.
        return "", None
    except IsADirectoryError:
        return "", "вместо файла саммари указан каталог: %s" % path
    except PermissionError:
        return "", "файл саммари недоступен для чтения: %s" % path
    except UnicodeDecodeError:
        return "", "файл саммари не в кодировке UTF-8: %s" % path
    except OSError as exc:
        return "", "не удалось прочитать файл саммари: %s (%s)" % (path, exc.strerror or exc)

    text = raw.strip("\n").rstrip()
    if not text.strip():
        return "", None
    return text, None


# ----------------------------------------------------------------- рендер ---


@dataclass
class Stats:
    """Чем закончилась сборка — это же уходит в отчёт и в код возврата."""

    lines: int = 0
    paragraphs: int = 0
    fallback_windows: list[int] = field(default_factory=list)
    summary_missing: bool = False
    speakers_note: str | None = None
    by_phrases: bool = False

    @property
    def marks(self) -> int:
        """Сколько пометок видно в документе."""
        return (
            len(self.fallback_windows)
            + (1 if self.summary_missing else 0)
            + (1 if self.speakers_note else 0)
        )


def default_title(source: str, out: str) -> str:
    """Заголовок по умолчанию — имя исходной записи без расширения.

    Берётся из поля source файла punctuated: там лежит путь к записи, с которой
    начинался прогон. Если его нет, остаётся имя самого документа — это хуже
    (в нём уже есть дата прогона), но лучше пустого заголовка.
    """
    name = os.path.basename(source.rstrip("/\\")) if source else ""
    if not name:
        warn("в файле punctuated нет source — заголовок взят из имени --out")
        name = os.path.basename(out)
    title = os.path.splitext(name)[0].strip()
    return title or "Расшифровка записи"


def render_fallback_mark(window: Window | None, index: int | None) -> str:
    """Строка-пометка перед блоком окна, где пунктуация не восстановилась.

    Markdown-цитата «>» выбрана потому, что пометка обязана быть видна в
    отрендеренном документе, а не прятаться в комментарии: читатель должен
    понимать, что ниже идёт дословный текст модели без пунктуации.
    """
    number = window.index if window is not None else index
    where = "окно %s" % ("—" if number is None else number)
    if window is not None and window.first_segment is not None and window.last_segment is not None:
        where += ", сегменты %d–%d" % (window.first_segment, window.last_segment)
    else:
        where += ", сегменты не указаны"
    reason = (window.reason if window is not None else "") or "причина не указана"
    reason = reason.rstrip().rstrip(".")  # точку ставит сам шаблон пометки
    return (
        "> ⚠️ Пунктуация не восстановлена (%s): %s. "
        "Ниже — дословный текст модели." % (where, reason)
    )


def render_speakers_mark(reason: str) -> str:
    """Строка-пометка под заголовком расшифровки: ярлыков говорящих не будет.

    Та же markdown-цитата, что и у пометки об откате пунктуации, и по той же
    причине: читатель должен видеть, что строки идут без имён не потому, что
    говорил один человек.
    """
    text = (reason or "").strip().rstrip(".") or "причина не указана"
    return "> ⚠️ Ярлыки говорящих не проставлены: %s." % text


def render_sentence(sentence: Sentence) -> str:
    """Предложение -> строка «[ЧЧ:ММ:СС] текст»."""
    return "[%s] %s" % (format_timecode(sentence.start), sentence.text)


def render_phrase(phrase: Phrase, roles: dict[str, str] | None) -> str:
    """Фраза -> строка «[ЧЧ:ММ:СС] Ярлык: текст», а без говорящего — без ярлыка."""
    label = speaker_label(phrase, roles)
    if label is None:
        return "[%s] %s" % (format_timecode(phrase.start), phrase.text)
    return "[%s] %s: %s" % (format_timecode(phrase.start), label, phrase.text)


@dataclass
class Block:
    """Абзац расшифровки: готовые строки и окна пунктуации, из которых он собран.

    Окон у абзаца обычно одно, но список, а не число: фразы speakers/1 нарезаны
    по речи, а не по окнам пунктуации, и абзац из них может задеть границу окон.
    Пометка об откате тогда встанет перед всем абзацем — раньше, чем начинается
    откатившееся окно, но зато не потеряется.
    """

    windows: list[int | None]
    fallback_hint: bool
    lines: list[str]


def _is_fallback(window_index: int | None, windows: dict[int, Window], hint: bool) -> bool:
    """Нужна ли пометка об откате для этого окна.

    Истина окна — поле status в windows. Флаг fallback у предложений работает
    страховкой: если списка окон в файле нет, пометку всё равно надо поставить.
    """
    window = windows.get(window_index) if window_index is not None else None
    if window is not None:
        return window.fallback
    return hint


def build_sentence_blocks(data: Punctuated) -> list[Block]:
    """Расшифровка предложениями — то, как документ собирался до говорящих."""
    blocks: list[Block] = []
    for paragraph in data.paragraphs:
        if not paragraph.sentences:
            continue
        blocks.append(
            Block(
                windows=[paragraph.window],
                fallback_hint=any(sentence.fallback for sentence in paragraph.sentences),
                lines=[render_sentence(sentence) for sentence in paragraph.sentences],
            )
        )
    return blocks


def build_phrase_blocks(
    data: Punctuated, speakers: Speakers, roles: dict[str, str] | None
) -> list[Block]:
    """Расшифровка фразами с ярлыками. Негодный speakers/1 — SpeakersError.

    Фразы обязаны идти по порядку, не перекрываться и покрывать все предложения
    punctuated/1: иначе часть расшифровки просто исчезла бы из документа, а это
    хуже, чем документ без ярлыков. Границы абзацев берутся из поля paragraph
    фраз, окно пунктуации — у предложений, которые фраза собой закрывает.
    """
    by_index = {sentence.index: sentence for sentence in data.sentences}
    covered: set[int] = set()
    previous_last: int | None = None
    missing_paragraph = False

    blocks: list[Block] = []
    current_key: object = object()  # заведомо не равно ни одному номеру абзаца
    for phrase in speakers.phrases:
        if previous_last is not None and phrase.first_sentence <= previous_last:
            raise SpeakersError(
                "фраза index=%d: предложения %d–%d перекрывают предыдущую фразу "
                "или идут не по порядку"
                % (phrase.index, phrase.first_sentence, phrase.last_sentence)
            )
        previous_last = phrase.last_sentence

        sentences = [
            by_index[number]
            for number in range(phrase.first_sentence, phrase.last_sentence + 1)
            if number in by_index
        ]
        covered.update(sentence.index for sentence in sentences)
        window = sentences[0].window if sentences else None
        hint = any(sentence.fallback for sentence in sentences)

        if phrase.paragraph is None:
            missing_paragraph = True
        key = phrase.paragraph
        if not blocks or key != current_key:
            current_key = key
            blocks.append(Block(windows=[], fallback_hint=False, lines=[]))
        block = blocks[-1]
        if window not in block.windows:
            block.windows.append(window)
        block.fallback_hint = block.fallback_hint or hint
        block.lines.append(render_phrase(phrase, roles))

    if missing_paragraph:
        warn("у части фраз нет поля paragraph — такие фразы идут одним абзацем")

    lost = sorted(number for number in by_index if number not in covered)
    if lost:
        raise SpeakersError(
            "фразы не покрывают %d предложений punctuated/1 (например, index=%d)"
            % (len(lost), lost[0])
        )
    if not blocks:
        raise SpeakersError("из фраз не собралось ни одного абзаца расшифровки")
    return blocks


def render_document(
    data: Punctuated,
    title: str,
    summary: str,
    summary_path: str | None,
    body: list[Block],
    speakers_note: str | None = None,
) -> tuple[str, Stats]:
    """Собирает текст итогового документа и статистику по нему.

    Документ — это список блоков, склеенных одной пустой строкой: заголовок,
    «## Саммари», фрагмент саммари, «## Расшифровка», пометка об отсутствии
    ярлыков (если она нужна) и дальше по два блока на абзац (пометка, если окно
    в откате, и сами строки расшифровки).
    """
    stats = Stats(speakers_note=speakers_note)
    blocks: list[str] = ["# %s" % title.strip(), TITLE_SUMMARY]

    if summary.strip():
        blocks.append(summary)
    else:
        stats.summary_missing = True
        if summary_path is None:
            missing = "Саммари не получено: файл с саммари не передан (--summary)."
        else:
            missing = (
                "Саммари не получено: файл %s отсутствует или пуст." % summary_path
            )
        blocks.append("%s\n\n%s" % (SUMMARY_FIRST_SECTION, missing))

    blocks.append(TITLE_BODY)
    if speakers_note:
        blocks.append(render_speakers_mark(speakers_note))

    marked: set[int | None] = set()
    for block in body:
        if not block.lines:
            continue
        for number in block.windows:
            if not _is_fallback(number, data.windows, block.fallback_hint):
                continue
            if number in marked:
                # Одно окно — одна пометка, даже если абзацев в нём несколько.
                continue
            marked.add(number)
            window = data.windows.get(number) if number is not None else None
            blocks.append(render_fallback_mark(window, number))
            stats.fallback_windows.append(
                window.index if window is not None else (number or 0)
            )
        blocks.append("\n".join(block.lines))
        stats.paragraphs += 1
        stats.lines += len(block.lines)

    return "\n\n".join(blocks) + "\n", stats


def build_body(data: Punctuated, opts: dict) -> tuple[list[Block], str | None, bool]:
    """Решает, из чего собирать расшифровку: из фраз с ярлыками или из предложений.

    Возвращает блоки расшифровки, причину отсутствия ярлыков (её увидит читатель
    документа) и признак «собрано фразами» — он нужен только отчёту в stdout.

    Правило одно: расшифровка выходит в любом случае. Негодный speakers/1 или
    roles/1 стоит ярлыков, но не строк, поэтому его беда становится пометкой в
    документе и кодом возврата 1, а не ошибкой прогона.
    """
    note = opts["speakers_note"].strip() if opts["speakers_note"] else None
    if opts["roles"] and not opts["speakers"]:
        warn("--roles без --speakers не используется: ярлыки берутся из фраз speakers/1")
    if note and opts["speakers"]:
        # Явная причина от вызывающего сильнее файла: раз он говорит, что
        # ярлыков не будет, ставить их в половину строк — врать читателю.
        warn(
            "--speakers-note задан вместе с --speakers: файл %s не читается, "
            "ярлыки не проставляются" % opts["speakers"]
        )

    if opts["speakers"] and not note:
        try:
            speakers = load_speakers(opts["speakers"])
            roles = load_roles(opts["roles"]) if opts["roles"] else None
            if roles and speakers.names and not any(name in roles for name in speakers.names):
                # Не беда файла, а рассогласование двух файлов: ярлыками станут
                # имена говорящих, и это лучше, чем документ без ярлыков вовсе.
                warn(
                    "ни одна роль из %s не совпала с именами говорящих — "
                    "ярлыками будут имена" % opts["roles"]
                )
            return build_phrase_blocks(data, speakers, roles), None, True
        except SpeakersError as exc:
            warn(str(exc))
            note = str(exc)

    return build_sentence_blocks(data), note, False


def write_text_file(path: str, content: str) -> None:
    """Пишет файл в UTF-8 с переводами строк LF.

    newline="\\n" задан явно: на Windows иначе получились бы CRLF, и «ровно одна
    пустая строка между абзацами» перестала бы быть правдой.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except OSError as exc:
        raise DocumentError(
            "не удалось записать файл %s (%s)" % (path, exc.strerror or exc)
        ) from None


# ------------------------------------------------------------------- отчёт ---


def print_report(path: str, size: int, stats: Stats) -> None:
    """Печатает итоги сборки — то, что потом уходит в отчёт о прогоне."""
    sys.stdout.write("================ ИТОГОВЫЙ ДОКУМЕНТ ================\n")
    sys.stdout.write("Файл:          %s\n" % path)
    sys.stdout.write("Размер:        %d байт\n" % size)
    if stats.by_phrases:
        sys.stdout.write("Фраз:          %d\n" % stats.lines)
    else:
        sys.stdout.write("Предложений:   %d\n" % stats.lines)
    sys.stdout.write("Абзацев:       %d\n" % stats.paragraphs)
    sys.stdout.write("Пометок:       %d\n" % stats.marks)
    if stats.summary_missing:
        sys.stdout.write("  * саммари не получено\n")
    if stats.speakers_note:
        sys.stdout.write("  * ярлыки говорящих не проставлены: %s\n" % stats.speakers_note)
    if stats.fallback_windows:
        sys.stdout.write(
            "  * пунктуация не восстановлена в окнах: %s\n"
            % ", ".join(str(number) for number in stats.fallback_windows)
        )


# --------------------------------------------------------- разбор аргументов ---

_FLAGS_VALUE = {
    "--punctuated": "punctuated",
    "--summary": "summary",
    "--speakers": "speakers",
    "--roles": "roles",
    "--speakers-note": "speakers_note",
    "--out": "out",
    "--title": "title",
}

# Флаги, значение которых — не путь, а произвольный текст: пустым для них
# считается и значение из одних пробелов.
_FLAGS_TEXT = ("--title", "--speakers-note")


def parse_args(argv: list[str]) -> dict:
    """Разбирает аргументы так же, как это делают остальные инструменты проекта.

    Своим разбором, а не argparse: тот печатает usage и свои ошибки по-английски,
    а весь остальной вывод инструментов проекта — на русском.
    """
    opts: dict = {
        "punctuated": None,
        "summary": None,
        "speakers": None,
        "roles": None,
        "speakers_note": None,
        "out": None,
        "title": None,
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
            # --title и --speakers-note принимают любую строку, остальные флаги
            # — путь; пустое значение в обоих случаях бессмысленно. У текстовых
            # флагов пустым считается и текст из одних пробелов: в документ он
            # уйдёт обрезанным.
            empty = not value.strip() if token in _FLAGS_TEXT else not value
            if empty:
                raise UsageError("%s требует непустое значение" % token)
            opts[_FLAGS_VALUE[token]] = value
            continue

        if token.startswith("-"):
            raise UsageError("неизвестная опция: %s (см. --help)" % token)
        raise UsageError("лишний аргумент: %s (см. --help)" % token)

    return opts


def check_args(opts: dict) -> None:
    """Проверяет набор аргументов целиком — то, что нельзя проверить по одному."""
    if not opts["punctuated"]:
        raise UsageError("не указан --punctuated (см. --help)")
    if not opts["out"]:
        raise UsageError("не указан --out (см. --help)")

    out = os.path.abspath(opts["out"])
    # Перетереть собственный вход — это потерять прогон целиком, а переписать
    # заново punctuated.json нечем: он стоит двух часов распознавания.
    if out == os.path.abspath(opts["punctuated"]):
        raise UsageError("--out указывает на тот же файл, что и --punctuated: %s" % opts["out"])
    for flag in ("--summary", "--speakers", "--roles"):
        value = opts[_FLAGS_VALUE[flag]]
        if value and out == os.path.abspath(value):
            raise UsageError(
                "--out указывает на тот же файл, что и %s: %s" % (flag, opts["out"])
            )


# ---------------------------------------------------------------- селф-тест ---

# Строка предложения: таймкод в квадратных скобках, пробел, непустой текст.
# Часы не ограничены двумя знаками сверху, но меньше двух быть не может.
SENTENCE_RE = re.compile(r"^\[(\d{2,}):([0-5]\d):([0-5]\d)\] (\S.*)$")

# Пометка об откате — ровно того вида, который требуют критерии приёмки.
MARK_RE = re.compile(
    r"^> ⚠️ Пунктуация не восстановлена \((?P<where>[^()]+)\): (?P<reason>.+)\. "
    r"Ниже — дословный текст модели\.$"
)

# Пометка об отсутствии ярлыков говорящих — она же стоит под «## Расшифровка».
NOTE_RE = re.compile(r"^> ⚠️ Ярлыки говорящих не проставлены: (?P<reason>.+)\.$")

# Разбор строки расшифровки на ярлык и текст. Двоеточие внутри самого текста в
# проверочных фразах не встречается, поэтому разделитель однозначен.
LABEL_RE = re.compile(r"^(?P<label>[^:]+): (?P<text>.+)$")

# Слова технического блока: ни одного из них в документе быть не должно.
TECH_WORDS = ("×RT", "SHA-256", "sha256", "source_duration", "v3_rnnt", "torch")


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass
class ParsedDoc:
    """Результат обратного разбора документа простым проходом по строкам."""

    title: str
    summary: list[str]
    marks: list[str]
    paragraphs: list[list[tuple[int, str]]]  # (секунды, текст) по строкам абзаца
    notes: list[str] = field(default_factory=list)  # пометки об отсутствии ярлыков


def parse_document(text: str) -> ParsedDoc:
    """Разбирает собственный документ построчно. Отступление от формата — AssertionError.

    Это и проверка структуры, и демонстрация того, что документ читается без
    markdown-парсера: сборочному скрипту и человеку хватает split("\\n").
    """
    _check(text, "документ пуст")
    _check(text.endswith("\n"), "документ не заканчивается переводом строки")
    _check("\r" not in text, "в документе есть возврат каретки")
    _check("\n\n\n" not in text, "в документе больше одной пустой строки подряд")

    lines = text.rstrip("\n").split("\n")
    _check(lines[0].startswith("# "), "первая строка не заголовок первого уровня: %r" % lines[0])
    title = lines[0][2:]
    _check(title.strip() == title and title, "заголовок пуст или с пробелами по краям: %r" % title)
    _check(len(lines) > 1 and lines[1] == "", "после заголовка нет пустой строки")

    _check(lines[2] == TITLE_SUMMARY, "третья строка не %r, а %r" % (TITLE_SUMMARY, lines[2]))
    _check(TITLE_BODY in lines, "в документе нет раздела %r" % TITLE_BODY)
    body_at = lines.index(TITLE_BODY)
    summary = [line for line in lines[3:body_at] if line != ""] or []

    headings = [line for line in lines if line.startswith("# ") or line.startswith("## ")]
    _check(
        headings == [lines[0], TITLE_SUMMARY, TITLE_BODY],
        "в документе есть лишние заголовки: %r" % headings,
    )

    marks: list[str] = []
    notes: list[str] = []
    paragraphs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for line in lines[body_at + 1:]:
        if line == "":
            if current:
                paragraphs.append(current)
                current = []
            continue
        if line.startswith(">"):
            _check(not current, "пометка внутри абзаца, а не перед ним: %r" % line)
            if NOTE_RE.match(line) is not None:
                _check(
                    not notes and not marks and not paragraphs,
                    "пометка о ярлыках стоит не сразу под заголовком расшифровки: %r" % line,
                )
                notes.append(line)
                continue
            _check(MARK_RE.match(line) is not None, "пометка не того вида: %r" % line)
            marks.append(line)
            continue
        match = SENTENCE_RE.match(line)
        _check(match is not None, "строка расшифровки не того вида: %r" % line)
        _check(line == line.rstrip(), "в конце строки расшифровки есть пробелы: %r" % line)
        hours, minutes, seconds = (int(match.group(number)) for number in (1, 2, 3))
        current.append((hours * 3600 + minutes * 60 + seconds, match.group(4)))
    if current:
        paragraphs.append(current)

    _check(paragraphs, "в разделе расшифровки нет ни одного абзаца")
    return ParsedDoc(
        title=title, summary=summary, marks=marks, paragraphs=paragraphs, notes=notes
    )


def _fixture_summary() -> str:
    """Фрагмент саммари — такой, какой пишет tools/summarize.py: три раздела «###»."""
    return (
        "### Изложение\n"
        "\n"
        "Обсудили сборку итогового документа и порядок разделов в нём.\n"
        "\n"
        "Договорились, что технический блок в документ не идёт.\n"
        "\n"
        "### Темы\n"
        "\n"
        "- [00:00:00] Вступление и план созвона\n"
        "- [01:02:03] Формат итогового markdown\n"
        "\n"
        "### Решения, задачи, договорённости\n"
        "\n"
        "- [01:02:03] Таймкод на каждое предложение\n"
    )


def _fixture_punctuated(*, fallback: bool = False) -> dict:
    """Файл схемы punctuated/1 для проверок: собирается в памяти, без claude.

    Два окна и три абзаца. Второе окно начинается на 3723-й секунде — это ровно
    01:02:03, то есть случай «таймкод больше часа» из критериев приёмки. При
    fallback=True второе окно объявлено откатившимся: в нём одна строка на
    сегмент, дословный текст модели без пунктуации.
    """
    sentences = [
        {"index": 0, "start": 0.0, "end": 4.25, "text": "Добрый день, начинаем созвон.", "window": 0, "fallback": False},
        {"index": 1, "start": 4.25, "end": 9.5, "text": "Сегодня разбираем формат итогового файла.", "window": 0, "fallback": False},
        {"index": 2, "start": 9.5, "end": 15.125, "text": "Первый вопрос — заголовок документа.", "window": 0, "fallback": False},
        {"index": 3, "start": 59.999, "end": 64.0, "text": "Второй абзац того же окна.", "window": 0, "fallback": False},
        {"index": 4, "start": 3723.9, "end": 3730.0, "text": "Через час вернулись к таймкодам.", "window": 1, "fallback": fallback},
        {"index": 5, "start": 3730.0, "end": 3736.5, "text": "На этом закончили.", "window": 1, "fallback": fallback},
    ]
    if fallback:
        # В откатившемся окне предложения — это сегменты: строчными, без точек.
        sentences[4]["text"] = "через час вернулись к таймкодам"
        sentences[5]["text"] = "на этом закончили"

    windows = [
        {"index": 0, "segments": [0, 99], "status": "ok", "reason": ""},
        {
            "index": 1,
            "segments": [100, 142],
            "status": "fallback" if fallback else "ok",
            "reason": "слова не совпали с транскриптом (позиция 17: «созвон» → «разговор»)"
            if fallback
            else "",
        },
    ]
    return {
        "schema": SCHEMA,
        "source": "data/output/созвон команды/созвон команды.webm",
        "source_duration": 3740.0,
        "transcript": "data/output/созвон команды/transcript.json",
        "model": "v3_rnnt",
        "windows": windows,
        "paragraphs": [
            {"index": 0, "window": 0, "sentences": [0, 2]},
            {"index": 1, "window": 0, "sentences": [3, 3]},
            {"index": 2, "window": 1, "sentences": [4, 5]},
        ],
        "sentences": sentences,
        "text": "\n\n".join(
            [
                " ".join(item["text"] for item in sentences[0:3]),
                sentences[3]["text"],
                " ".join(item["text"] for item in sentences[4:6]),
            ]
        ),
        "stats": {
            "windows": 2,
            "fallback_windows": 1 if fallback else 0,
            "paragraphs": 3,
            "sentences": len(sentences),
            "words": sum(len(item["text"].split()) for item in sentences),
        },
    }


def _fixture_speakers(*, fallback: bool = False) -> dict:
    """Файл схемы speakers/1 под тот же punctuated: фразы покрывают предложения.

    Пять фраз на шесть предложений, три абзаца — те же, что и в punctuated. У
    третьей фразы говорящего нет (диаризация не покрыла кусок): на ней
    проверяется строка без ярлыка.
    """
    sentences = _fixture_punctuated(fallback=fallback)["sentences"]
    plan = [
        ((0, 1), "Говорящий 1", 0),
        ((2, 2), "Говорящий 2", 0),
        ((3, 3), None, 1),
        ((4, 4), "Говорящий 1", 2),
        ((5, 5), "Говорящий 2", 2),
    ]
    phrases = []
    for number, ((first, last), speaker, paragraph) in enumerate(plan):
        covered = sentences[first : last + 1]
        phrases.append(
            {
                "index": number,
                "start": covered[0]["start"],
                "end": covered[-1]["end"],
                "text": " ".join(item["text"] for item in covered),
                "speaker": speaker,
                "sentences": [first, last],
                "paragraph": paragraph,
                "fallback": False,
                "reason": "speaker" if speaker else "no-speaker",
            }
        )
    names = ["Говорящий 1", "Говорящий 2"]
    return {
        "schema": SPEAKERS_SCHEMA,
        "transcript": "data/output/созвон команды/transcript.json",
        "punctuated": "data/output/созвон команды/punctuated.json",
        "diarization": "data/output/созвон команды/diarization.json",
        "pause": 1.0,
        "speakers": [
            {
                "name": name,
                "label": name,
                "phrases": len([item for item in phrases if item["speaker"] == name]),
                "speech_seconds": 100.0,
                "share": 0.5,
            }
            for name in names
        ],
        "phrases": phrases,
        "stats": {"speakers": len(names), "phrases": len(phrases)},
    }


def _fixture_roles() -> dict:
    """Файл схемы roles/1: «Говорящий N → роль», как его пишет tools/roles.py."""
    return {
        "schema": ROLES_SCHEMA,
        "speakers": "data/output/созвон команды/speakers.json",
        "status": "ok",
        "reason": "",
        "roles": [
            {"speaker": "Говорящий 1", "role": "Преподаватель", "share": 0.7, "source": "claude"},
            {"speaker": "Говорящий 2", "role": "Студент 1", "share": 0.3, "source": "claude"},
        ],
        "stats": {"speakers": 2, "from_claude": 2, "defaulted": 0},
    }


# Документ, который инструмент собирает из _fixture_punctuated() и
# _fixture_summary() без --speakers. Он выписан целиком, а не собран кодом,
# именно затем, чтобы правка рендера не могла молча изменить в нём ни байта:
# критерий требует, чтобы без --speakers вывод остался прежним.
PLAIN_DOCUMENT = """# созвон команды

## Саммари

### Изложение

Обсудили сборку итогового документа и порядок разделов в нём.

Договорились, что технический блок в документ не идёт.

### Темы

- [00:00:00] Вступление и план созвона
- [01:02:03] Формат итогового markdown

### Решения, задачи, договорённости

- [01:02:03] Таймкод на каждое предложение

## Расшифровка

[00:00:00] Добрый день, начинаем созвон.
[00:00:04] Сегодня разбираем формат итогового файла.
[00:00:09] Первый вопрос — заголовок документа.

[00:00:59] Второй абзац того же окна.

[01:02:03] Через час вернулись к таймкодам.
[01:02:10] На этом закончили.
"""


def _write_json(path: str, payload: object) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Сквозной прогон main() с перехватом вывода."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _test_timecodes() -> None:
    """Формат таймкода: двузначные части, отбрасывание дробей, больше часа."""
    cases = [
        (0.0, "00:00:00"),
        (0.999, "00:00:00"),
        (9.5, "00:00:09"),
        (59.999, "00:00:59"),  # округления нет: 59.999 это ещё 59-я секунда
        (60.0, "00:01:00"),
        (3599.0, "00:59:59"),
        (3600.0, "01:00:00"),
        (3723.0, "01:02:03"),
        (3723.9, "01:02:03"),
        (9323.0, "02:35:23"),
    ]
    for seconds, expected in cases:
        got = format_timecode(seconds)
        _check(got == expected, "%s -> %s, ожидалось %s" % (seconds, got, expected))
        _check(
            re.fullmatch(r"\d{2,}:\d{2}:\d{2}", got) is not None,
            "таймкод не в формате ЧЧ:ММ:СС: %s" % got,
        )
    # Больше 99 часов: часы растут, а не обрезаются.
    _check(format_timecode(360_000.0) == "100:00:00", "100 часов -> %s" % format_timecode(360_000.0))
    for bad in (-0.001, float("nan"), float("inf")):
        try:
            format_timecode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("формат принял недопустимое значение %r" % bad)


def _test_plain_document() -> None:
    """Обычный документ: структура, порядок предложений, саммари как есть."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        punctuated = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        summary = os.path.join(workdir, "summary.md")
        write_text_file(summary, _fixture_summary())
        # Кириллица и пробелы в пути и каталог, которого ещё нет.
        out = os.path.join(workdir, "итог прогона", "созвон команды-2026-10-03.md")

        code, report, stderr = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out]
        )
        _check(code == 0, "обычный документ вернул код %d (stderr: %s)" % (code, stderr.strip()))
        _check(os.path.isfile(out), "документ не записан: %s" % out)

        with open(out, "r", encoding="utf-8") as handle:
            text = handle.read()
        doc = parse_document(text)

        _check(text == PLAIN_DOCUMENT, "документ собрался не байт в байт:\n%s" % text)
        _check(doc.title == "созвон команды", "заголовок по умолчанию: %r" % doc.title)
        _check(not doc.marks, "в чистом документе появились пометки: %r" % doc.marks)
        _check(len(doc.paragraphs) == 3, "абзацев %d, а в файле 3" % len(doc.paragraphs))

        # Фрагмент саммари вставлен как есть: все его непустые строки на месте и
        # в том же порядке.
        expected_summary = [line for line in _fixture_summary().split("\n") if line != ""]
        _check(doc.summary == expected_summary, "фрагмент саммари переформатирован: %r" % doc.summary)

        # Порядок предложений — по index, таймкоды не убывают, текст не переписан.
        fixture = _fixture_punctuated()["sentences"]
        flat = [line for paragraph in doc.paragraphs for line in paragraph]
        _check(len(flat) == len(fixture), "строк %d, а предложений %d" % (len(flat), len(fixture)))
        previous = -1
        for (seconds, body), sentence in zip(flat, fixture):
            _check(body == sentence["text"], "текст предложения изменился: %r" % body)
            _check(seconds == int(sentence["start"]), "таймкод %d при start %s" % (seconds, sentence["start"]))
            _check(seconds >= previous, "таймкоды пошли вспять на %r" % body)
            previous = seconds
        _check("[01:02:03]" in text, "таймкод больше часа не отрендерился как 01:02:03")

        # Технического блока в документе нет.
        for word in TECH_WORDS:
            _check(word not in text, "в документе есть техническая деталь: %s" % word)

        # Отчёт в stdout — на русском и со всеми числами.
        for needle in ("Файл:", "Размер:", "Предложений:   6", "Абзацев:       3", "Пометок:       0"):
            _check(needle in report, "в отчёте нет строки %r" % needle)
        _check(out in report, "в отчёте нет пути к документу")
        _check(str(os.path.getsize(out)) in report, "в отчёте нет размера файла")

        # Заголовок можно задать руками — с пробелами и кириллицей.
        out2 = os.path.join(workdir, "второй.md")
        code, _, _ = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out2,
             "--title", "Созвон команды, 3 октября"]
        )
        _check(code == 0, "прогон с --title вернул код %d" % code)
        with open(out2, "r", encoding="utf-8") as handle:
            _check(
                parse_document(handle.read()).title == "Созвон команды, 3 октября",
                "--title не попал в заголовок",
            )


def _test_fallback_document() -> None:
    """Документ с окном в откате: пометка перед блоком, код возврата 1."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        punctuated = _write_json(
            os.path.join(workdir, "punctuated.json"), _fixture_punctuated(fallback=True)
        )
        summary = os.path.join(workdir, "summary.md")
        write_text_file(summary, _fixture_summary())
        out = os.path.join(workdir, "doc.md")

        code, report, stderr = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out]
        )
        _check(code == 1, "документ с откатом вернул код %d, ожидался 1" % code)

        with open(out, "r", encoding="utf-8") as handle:
            text = handle.read()
        doc = parse_document(text)
        _check(len(doc.marks) == 1, "пометок %d, а окно в откате одно" % len(doc.marks))
        mark = doc.marks[0]
        match = MARK_RE.match(mark)
        _check(match.group("where") == "окно 1, сегменты 100–142", "в пометке не тот диапазон: %r" % mark)
        _check("созвон" in match.group("reason"), "в пометке нет причины отката: %r" % mark)

        # Пометка стоит перед своим блоком и отделена от него пустой строкой:
        # иначе markdown затянул бы первую строку абзаца внутрь цитаты.
        lines = text.rstrip("\n").split("\n")
        at = lines.index(mark)
        _check(lines[at - 1] == "", "перед пометкой нет пустой строки")
        _check(lines[at + 1] == "", "после пометки нет пустой строки")
        _check(lines[at + 2].startswith("[01:02:03] "), "после пометки не блок её окна: %r" % lines[at + 2])
        # Текст отката идёт дословно, без пунктуации — его не «починили».
        _check(lines[at + 2] == "[01:02:03] через час вернулись к таймкодам", "текст отката изменён")

        # Предложения чистого окна остались выше пометки и не помечены.
        _check(at > lines.index(TITLE_BODY), "пометка оказалась выше раздела расшифровки")
        _check(len(doc.paragraphs) == 3, "абзацев %d, а в файле 3" % len(doc.paragraphs))

        for needle in ("Пометок:       1", "окнах: 1"):
            _check(needle in report, "в отчёте нет строки %r" % needle)
        _check(not stderr.strip(), "при корректном входе есть вывод в stderr: %s" % stderr.strip())

        # Несколько абзацев в одном откатившемся окне дают ровно одну пометку.
        payload = _fixture_punctuated(fallback=True)
        payload["paragraphs"][2] = {"index": 2, "window": 1, "sentences": [4, 4]}
        payload["paragraphs"].append({"index": 3, "window": 1, "sentences": [5, 5]})
        many = _write_json(os.path.join(workdir, "many.json"), payload)
        out2 = os.path.join(workdir, "many.md")
        code, _, _ = _run(["--punctuated", many, "--summary", summary, "--out", out2])
        _check(code == 1, "второй прогон с откатом вернул код %d" % code)
        with open(out2, "r", encoding="utf-8") as handle:
            doc2 = parse_document(handle.read())
        _check(len(doc2.marks) == 1, "на два абзаца одного окна выдано %d пометок" % len(doc2.marks))
        _check(len(doc2.paragraphs) == 4, "абзацев %d, а в файле 4" % len(doc2.paragraphs))


def _test_missing_summary() -> None:
    """Саммари нет: расшифровка сохраняется, под «## Саммари» — пометка, код 1."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        punctuated = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        empty = os.path.join(workdir, "пусто.md")
        write_text_file(empty, "\n   \n\n")
        absent = os.path.join(workdir, "нет саммари.md")

        cases = [
            ("файла нет", ["--summary", absent], absent),
            ("файл пуст", ["--summary", empty], empty),
            ("флаг не передан", [], None),
        ]
        for name, extra, path in cases:
            out = os.path.join(workdir, "out-%d.md" % len(name))
            code, report, stderr = _run(["--punctuated", punctuated, "--out", out] + extra)
            _check(code == 1, "случай %r вернул код %d, ожидался 1" % (name, code))

            with open(out, "r", encoding="utf-8") as handle:
                text = handle.read()
            doc = parse_document(text)
            _check(
                doc.summary[0] == SUMMARY_FIRST_SECTION,
                "случай %r: под «## Саммари» нет %r (%r)" % (name, SUMMARY_FIRST_SECTION, doc.summary[:1]),
            )
            note = doc.summary[1]
            _check(
                note.startswith("Саммари не получено:"),
                "случай %r: нет пометки об отсутствии саммари (%r)" % (name, note),
            )
            if path is not None:
                _check(
                    note == "Саммари не получено: файл %s отсутствует или пуст." % path,
                    "случай %r: пометка не того вида: %r" % (name, note),
                )
                _check(path in note, "случай %r: в пометке нет пути к файлу" % name)
            # Расшифровка на месте и целиком.
            _check(
                sum(len(paragraph) for paragraph in doc.paragraphs) == 6,
                "случай %r: расшифровка потерялась" % name,
            )
            _check(not doc.marks, "случай %r: появилась пометка об откате" % name)
            _check("Пометок:       1" in report, "случай %r: в отчёте не та цифра пометок" % name)
            _check("саммари не получено" in report, "случай %r: в отчёте нет причины пометки" % name)

        # Нет ни саммари, ни пунктуации — две пометки, код всё равно 1.
        both = _write_json(os.path.join(workdir, "both.json"), _fixture_punctuated(fallback=True))
        out = os.path.join(workdir, "both.md")
        code, report, _ = _run(["--punctuated", both, "--out", out])
        _check(code == 1, "документ с двумя пометками вернул код %d" % code)
        _check("Пометок:       2" in report, "в отчёте не две пометки: %s" % report)

        # Каталог вместо файла саммари — тот же результат, что и «файла нет».
        code, _, stderr = _run(
            ["--punctuated", punctuated, "--summary", workdir, "--out", os.path.join(workdir, "dir.md")]
        )
        _check(code == 1, "каталог вместо саммари вернул код %d" % code)
        _check(stderr.strip() != "", "про каталог вместо саммари ничего не сказано")


def _test_broken_input() -> None:
    """Непригодный вход: код 2, понятное сообщение, файл --out не создаётся."""
    cases = [
        ("не объект", []),
        ("нет schema", {"sentences": []}),
        ("чужая схема", {"schema": "punctuated/2", "sentences": []}),
        ("схема транскрипта", {"schema": "transcript/1", "sentences": []}),
        ("нет sentences", {"schema": SCHEMA, "paragraphs": []}),
        ("sentences не список", {"schema": SCHEMA, "sentences": {}}),
        ("sentences пуст", {"schema": SCHEMA, "sentences": []}),
        (
            "все предложения пустые",
            {"schema": SCHEMA, "sentences": [{"index": 0, "start": 0.0, "end": 1.0, "text": "   "}]},
        ),
        (
            "нет start",
            {"schema": SCHEMA, "sentences": [{"index": 0, "end": 1.0, "text": "а"}]},
        ),
        (
            "start не число",
            {"schema": SCHEMA, "sentences": [{"index": 0, "start": "ноль", "end": 1.0, "text": "а"}]},
        ),
        (
            "start меньше нуля",
            {"schema": SCHEMA, "sentences": [{"index": 0, "start": -1.0, "end": 1.0, "text": "а"}]},
        ),
        (
            "нет text",
            {"schema": SCHEMA, "sentences": [{"index": 0, "start": 0.0, "end": 1.0}]},
        ),
        (
            "index повторяется",
            {
                "schema": SCHEMA,
                "sentences": [
                    {"index": 0, "start": 0.0, "end": 1.0, "text": "а"},
                    {"index": 0, "start": 1.0, "end": 2.0, "text": "б"},
                ],
            },
        ),
        (
            "границы абзаца перевёрнуты",
            {
                "schema": SCHEMA,
                "sentences": [{"index": 0, "start": 0.0, "end": 1.0, "text": "а"}],
                "paragraphs": [{"index": 0, "window": 0, "sentences": [1, 0]}],
            },
        ),
        (
            "абзац без границ",
            {
                "schema": SCHEMA,
                "sentences": [{"index": 0, "start": 0.0, "end": 1.0, "text": "а"}],
                "paragraphs": [{"index": 0, "window": 0}],
            },
        ),
        (
            "windows не список",
            {
                "schema": SCHEMA,
                "sentences": [{"index": 0, "start": 0.0, "end": 1.0, "text": "а"}],
                "windows": 5,
            },
        ),
    ]
    for name, payload in cases:
        try:
            # Предупреждения разбора глушатся: часть случаев по пути к ошибке
            # успевает пожаловаться на абзацы, и в выводе селф-теста это мусор.
            with redirect_stderr(io.StringIO()):
                parse_punctuated(payload)
        except DocumentError as exc:
            _check(str(exc).strip() != "", "случай %r: пустое сообщение об ошибке" % name)
        else:
            raise AssertionError("случай %r прошёл проверку структуры" % name)

    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        good = _write_json(os.path.join(workdir, "good.json"), _fixture_punctuated())
        empty_sentences = _write_json(
            os.path.join(workdir, "empty.json"),
            dict(_fixture_punctuated(), sentences=[], paragraphs=[], stats={}),
        )
        broken = os.path.join(workdir, "broken.json")
        write_text_file(broken, '{"schema": "punctuated/1", "sentences": [')
        out = os.path.join(workdir, "не должен появиться", "doc.md")

        for argv, what in (
            (["--punctuated", empty_sentences, "--out", out], "пустой список предложений"),
            (["--punctuated", broken, "--out", out], "битый JSON"),
            (["--punctuated", os.path.join(workdir, "нет.json"), "--out", out], "нет файла"),
            (["--punctuated", workdir, "--out", out], "каталог вместо файла"),
            (["--out", out], "без --punctuated"),
            (["--punctuated", good], "без --out"),
            (["--punctuated", good, "--out", good], "--out поверх входа"),
            (["--punctuated", good, "--out", out, "--title", "  "], "пустой --title"),
            (["--punctuated", good, "--out"], "--out без значения"),
            (["--кривая-опция"], "неизвестная опция"),
            (["лишнее"], "лишний аргумент"),
            (["--selftest=1"], "--selftest со значением"),
        ):
            code, _, stderr = _run(argv)
            _check(code == 2, "случай %r завершился кодом %d, ожидался 2" % (what, code))
            _check(
                stderr.startswith(PROG + ":"),
                "случай %r ничего не написал в stderr" % what,
            )
            _check(
                not os.path.exists(out),
                "случай %r оставил файл --out на диске" % what,
            )

        # Существующий документ при непригодном входе не перетирается.
        kept = os.path.join(workdir, "kept.md")
        write_text_file(kept, "старое содержимое\n")
        code, _, _ = _run(["--punctuated", broken, "--out", kept])
        _check(code == 2, "перетирание: код %d вместо 2" % code)
        with open(kept, "r", encoding="utf-8") as handle:
            _check(handle.read() == "старое содержимое\n", "старый документ перетёрт при ошибке входа")


def _test_degraded_input() -> None:
    """Входу чего-то не хватает, но расшифровку можно собрать — собираем."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        # Нет абзацев и нет окон: вся расшифровка одним абзацем, пометок нет.
        payload = _fixture_punctuated()
        payload.pop("paragraphs")
        payload.pop("windows")
        path = _write_json(os.path.join(workdir, "flat.json"), payload)
        out = os.path.join(workdir, "flat.md")
        code, _, stderr = _run(["--punctuated", path, "--out", out, "--title", "Без абзацев"])
        _check(code == 1, "документ без саммари вернул код %d" % code)  # саммари не передано
        with open(out, "r", encoding="utf-8") as handle:
            doc = parse_document(handle.read())
        _check(len(doc.paragraphs) == 1, "абзацев %d вместо одного" % len(doc.paragraphs))
        _check(len(doc.paragraphs[0]) == 6, "строк в единственном абзаце %d" % len(doc.paragraphs[0]))
        _check("абзац" in stderr, "про отсутствие абзацев не предупредили: %s" % stderr)

        # Нет source — заголовок берётся из имени --out.
        payload = _fixture_punctuated()
        payload["source"] = ""
        path = _write_json(os.path.join(workdir, "nosource.json"), payload)
        out = os.path.join(workdir, "запись без source.md")
        code, _, stderr = _run(["--punctuated", path, "--out", out])
        _check(code == 1, "прогон без source вернул код %d" % code)
        with open(out, "r", encoding="utf-8") as handle:
            _check(
                parse_document(handle.read()).title == "запись без source",
                "заголовок не взят из имени --out",
            )
        _check("source" in stderr, "про отсутствие source не предупредили")

        # Окон нет, но предложения помечены fallback — пометка всё равно есть,
        # просто без диапазона сегментов.
        payload = _fixture_punctuated(fallback=True)
        payload.pop("windows")
        path = _write_json(os.path.join(workdir, "noweindows.json"), payload)
        out = os.path.join(workdir, "nowindows.md")
        code, report, _ = _run(["--punctuated", path, "--out", out])
        _check(code == 1, "прогон без windows вернул код %d" % code)
        with open(out, "r", encoding="utf-8") as handle:
            doc = parse_document(handle.read())
        _check(len(doc.marks) == 1, "пометок %d вместо одной" % len(doc.marks))
        _check("сегменты не указаны" in doc.marks[0], "пометка без windows не того вида: %r" % doc.marks[0])
        _check("Пометок:       2" in report, "в отчёте не две пометки")

        # Предложение, не попавшее ни в один абзац, не теряется.
        payload = _fixture_punctuated()
        payload["paragraphs"] = [{"index": 0, "window": 0, "sentences": [0, 2]}]
        path = _write_json(os.path.join(workdir, "orphans.json"), payload)
        out = os.path.join(workdir, "orphans.md")
        code, _, stderr = _run(["--punctuated", path, "--out", out])
        _check(code == 1, "прогон с потерянными предложениями вернул код %d" % code)
        with open(out, "r", encoding="utf-8") as handle:
            doc = parse_document(handle.read())
        _check(
            sum(len(paragraph) for paragraph in doc.paragraphs) == 6,
            "потерялись предложения вне абзацев",
        )
        _check("не попали ни в один абзац" in stderr, "про предложения вне абзацев не предупредили")

        # Окно с номером 0 в откате. Номер абзаца «0» — это номер, а не «номера
        # нет»: чтобы ветки нельзя было перепутать, у предложений этих абзацев
        # номер окна проставлен другой, и пометку обязан дать абзац.
        payload = _fixture_punctuated()
        payload["windows"][0]["status"] = "fallback"
        payload["windows"][0]["reason"] = "claude недоступен"
        for item in payload["sentences"][:4]:
            item["window"] = 1
        path = _write_json(os.path.join(workdir, "window0.json"), payload)
        out = os.path.join(workdir, "window0.md")
        code, _, _ = _run(["--punctuated", path, "--out", out])
        _check(code == 1, "прогон с откатом окна 0 вернул код %d" % code)
        with open(out, "r", encoding="utf-8") as handle:
            doc = parse_document(handle.read())
        _check(len(doc.marks) == 1, "пометок %d вместо одной" % len(doc.marks))
        _check(
            "окно 0, сегменты 0–99" in doc.marks[0],
            "окно 0 не распознано как окно: %r" % doc.marks[0],
        )
        _check(
            doc.paragraphs[0][0][1].startswith("Добрый день"),
            "пометка встала не перед первым абзацем окна 0",
        )

        # А когда поля window у абзаца нет, номер берётся у его предложений.
        payload = _fixture_punctuated(fallback=True)
        for paragraph in payload["paragraphs"]:
            paragraph.pop("window")
        path = _write_json(os.path.join(workdir, "window-from-sentences.json"), payload)
        out = os.path.join(workdir, "window-from-sentences.md")
        code, _, _ = _run(["--punctuated", path, "--out", out])
        _check(code == 1, "прогон без window у абзацев вернул код %d" % code)
        with open(out, "r", encoding="utf-8") as handle:
            doc = parse_document(handle.read())
        _check(
            len(doc.marks) == 1 and "окно 1, сегменты 100–142" in doc.marks[0],
            "номер окна не взят у предложений: %r" % doc.marks,
        )

        # Предложение с пустым текстом выпадает, остальные остаются.
        payload = _fixture_punctuated()
        payload["sentences"][3]["text"] = "  "
        path = _write_json(os.path.join(workdir, "blank.json"), payload)
        out = os.path.join(workdir, "blank.md")
        code, report, stderr = _run(["--punctuated", path, "--out", out])
        _check(code == 1, "прогон с пустым предложением вернул код %d" % code)
        with open(out, "r", encoding="utf-8") as handle:
            doc = parse_document(handle.read())
        _check(
            sum(len(paragraph) for paragraph in doc.paragraphs) == 5,
            "пустое предложение попало в документ",
        )
        _check(len(doc.paragraphs) == 2, "абзац из одного пустого предложения не исчез")
        _check("Предложений:   5" in report, "в отчёте не то число предложений")


def _speaker_lines(doc: ParsedDoc) -> list[tuple[int, str]]:
    """Все строки расшифровки подряд — абзацы здесь проверяются отдельно."""
    return [line for paragraph in doc.paragraphs for line in paragraph]


def _test_speaker_labels() -> None:
    """Расшифровка фразами: роль из roles/1, имя без него, строка без ярлыка."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        punctuated = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        speakers = _write_json(os.path.join(workdir, "speakers.json"), _fixture_speakers())
        roles = _write_json(os.path.join(workdir, "roles.json"), _fixture_roles())
        summary = os.path.join(workdir, "summary.md")
        write_text_file(summary, _fixture_summary())
        out = os.path.join(workdir, "с ролями.md")

        code, report, stderr = _run(
            ["--punctuated", punctuated, "--summary", summary,
             "--speakers", speakers, "--roles", roles, "--out", out]
        )
        _check(code == 0, "документ с ярлыками вернул код %d (stderr: %s)" % (code, stderr.strip()))
        _check(not stderr.strip(), "при годных speakers и roles есть stderr: %s" % stderr.strip())

        with open(out, "r", encoding="utf-8") as handle:
            text = handle.read()
        doc = parse_document(text)
        _check(not doc.notes, "при годной диаризации появилась пометка: %r" % doc.notes)
        _check(not doc.marks, "в чистом документе появились пометки об откате")
        _check(len(doc.paragraphs) == 3, "абзацев %d, а в speakers их 3" % len(doc.paragraphs))

        # Фраза собрана из двух предложений, у третьей говорящего нет.
        expected = [
            (0, "Преподаватель", "Добрый день, начинаем созвон. Сегодня разбираем формат итогового файла."),
            (9, "Студент 1", "Первый вопрос — заголовок документа."),
            (59, None, "Второй абзац того же окна."),
            (3723, "Преподаватель", "Через час вернулись к таймкодам."),
            (3730, "Студент 1", "На этом закончили."),
        ]
        lines = _speaker_lines(doc)
        _check(len(lines) == len(expected), "строк %d, а фраз %d" % (len(lines), len(expected)))
        for (seconds, body), (want_seconds, want_label, want_text) in zip(lines, expected):
            _check(seconds == want_seconds, "таймкод %d вместо %d у %r" % (seconds, want_seconds, body))
            match = LABEL_RE.match(body)
            if want_label is None:
                _check(match is None, "у фразы без говорящего появился ярлык: %r" % body)
                _check(body == want_text, "текст фразы без говорящего изменился: %r" % body)
                continue
            _check(match is not None, "в строке нет ярлыка говорящего: %r" % body)
            _check(match.group("label") == want_label, "ярлык %r вместо %r" % (match.group("label"), want_label))
            _check(match.group("text") == want_text, "текст фразы изменился: %r" % body)
        _check(
            "[00:00:00] Преподаватель: Добрый день" in text,
            "строка с ярлыком не того вида",
        )
        _check("Фраз:          5" in report, "в отчёте нет числа фраз: %s" % report)

        # Без --roles ярлык — имя говорящего от диаризации.
        out2 = os.path.join(workdir, "без ролей.md")
        code, _, _ = _run(
            ["--punctuated", punctuated, "--summary", summary,
             "--speakers", speakers, "--out", out2]
        )
        _check(code == 0, "документ без --roles вернул код %d" % code)
        with open(out2, "r", encoding="utf-8") as handle:
            doc2 = parse_document(handle.read())
        labels = [LABEL_RE.match(body) for _, body in _speaker_lines(doc2)]
        got = [match.group("label") if match else None for match in labels]
        _check(
            got == ["Говорящий 1", "Говорящий 2", None, "Говорящий 1", "Говорящий 2"],
            "без --roles ярлыки не стали именами говорящих: %r" % got,
        )

        # Окно в откате: пометка та же и про тот же диапазон сегментов.
        fall_punctuated = _write_json(
            os.path.join(workdir, "fall.json"), _fixture_punctuated(fallback=True)
        )
        fall_speakers = _write_json(
            os.path.join(workdir, "fall-speakers.json"), _fixture_speakers(fallback=True)
        )
        out3 = os.path.join(workdir, "откат.md")
        code, _, _ = _run(
            ["--punctuated", fall_punctuated, "--summary", summary,
             "--speakers", fall_speakers, "--roles", roles, "--out", out3]
        )
        _check(code == 1, "документ фразами с откатом вернул код %d" % code)
        with open(out3, "r", encoding="utf-8") as handle:
            fall_text = handle.read()
        doc3 = parse_document(fall_text)
        _check(len(doc3.marks) == 1, "пометок об откате %d вместо одной" % len(doc3.marks))
        _check(
            "окно 1, сегменты 100–142" in doc3.marks[0],
            "в пометке не тот диапазон сегментов: %r" % doc3.marks[0],
        )
        at = fall_text.rstrip("\n").split("\n").index(doc3.marks[0])
        after = fall_text.rstrip("\n").split("\n")[at + 2]
        _check(
            after.startswith("[01:02:03] Преподаватель: "),
            "после пометки не блок её окна: %r" % after,
        )


def _test_speakers_degraded() -> None:
    """Негодные speakers/1 и roles/1: расшифровка предложениями, пометка, код 1."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        punctuated = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        roles = _write_json(os.path.join(workdir, "roles.json"), _fixture_roles())
        summary = os.path.join(workdir, "summary.md")
        write_text_file(summary, _fixture_summary())

        no_coverage = _fixture_speakers()
        no_coverage["phrases"] = no_coverage["phrases"][:-1]  # предложение 5 осталось ничьим
        overlap = _fixture_speakers()
        overlap["phrases"][1]["sentences"] = [0, 2]
        empty_text = _fixture_speakers()
        empty_text["phrases"][2]["text"] = "   "
        bad_speaker = _fixture_speakers()
        bad_speaker["phrases"][0]["speaker"] = 17
        no_bounds = _fixture_speakers()
        no_bounds["phrases"][0].pop("sentences")

        broken = os.path.join(workdir, "битый.json")
        write_text_file(broken, '{"schema": "speakers/1", "phrases": [')

        cases: list[tuple[str, list[str]]] = []
        for name, payload in (
            ("чужая схема", dict(_fixture_speakers(), schema="speakers/2")),
            ("схема punctuated", dict(_fixture_speakers(), schema=SCHEMA)),
            ("нет phrases", {"schema": SPEAKERS_SCHEMA, "speakers": []}),
            ("phrases не список", {"schema": SPEAKERS_SCHEMA, "phrases": {}}),
            ("phrases пуст", {"schema": SPEAKERS_SCHEMA, "phrases": []}),
            ("фразы не покрывают предложения", no_coverage),
            ("фразы перекрываются", overlap),
            ("пустой текст фразы", empty_text),
            ("speaker не строка", bad_speaker),
            ("нет границ по предложениям", no_bounds),
        ):
            path = _write_json(os.path.join(workdir, "s-%d.json" % len(cases)), payload)
            cases.append((name, ["--speakers", path]))
        cases.append(("битый JSON", ["--speakers", broken]))
        cases.append(("нет файла", ["--speakers", os.path.join(workdir, "нет.json")]))
        cases.append(("каталог вместо файла", ["--speakers", workdir]))

        good = _write_json(os.path.join(workdir, "speakers.json"), _fixture_speakers())
        for name, payload in (
            ("roles чужой схемы", dict(_fixture_roles(), schema="roles/2")),
            ("roles без списка", {"schema": ROLES_SCHEMA, "status": "ok"}),
            ("roles не список", {"schema": ROLES_SCHEMA, "roles": 5}),
            ("роль не строка", {"schema": ROLES_SCHEMA, "roles": [{"speaker": "Говорящий 1", "role": 1}]}),
            (
                "говорящий дважды",
                {
                    "schema": ROLES_SCHEMA,
                    "roles": [
                        {"speaker": "Говорящий 1", "role": "Преподаватель"},
                        {"speaker": "Говорящий 1", "role": "Студент 1"},
                    ],
                },
            ),
        ):
            path = _write_json(os.path.join(workdir, "r-%d.json" % len(cases)), payload)
            cases.append((name, ["--speakers", good, "--roles", path]))

        fixture = _fixture_punctuated()["sentences"]
        for number, (name, extra) in enumerate(cases):
            out = os.path.join(workdir, "out-%d.md" % number)
            code, report, stderr = _run(
                ["--punctuated", punctuated, "--summary", summary, "--out", out] + extra
            )
            _check(code == 1, "случай %r вернул код %d, ожидался 1" % (name, code))
            _check(stderr.strip() != "", "случай %r ничего не сказал в stderr" % name)

            with open(out, "r", encoding="utf-8") as handle:
                text = handle.read()
            doc = parse_document(text)
            _check(len(doc.notes) == 1, "случай %r: пометок о ярлыках %d" % (name, len(doc.notes)))
            reason = NOTE_RE.match(doc.notes[0]).group("reason")
            _check(reason.strip() != "", "случай %r: пометка без причины" % name)

            lines = _speaker_lines(doc)
            _check(
                [body for _, body in lines] == [item["text"] for item in fixture],
                "случай %r: расшифровка собралась не предложениями" % name,
            )
            _check(
                text.replace(doc.notes[0] + "\n\n", "", 1) == PLAIN_DOCUMENT,
                "случай %r: документ отличается от обычного не только пометкой" % name,
            )
            _check(
                "ярлыки говорящих не проставлены" in report,
                "случай %r: в отчёте нет причины пометки" % name,
            )

        # Роли есть, но ни одна не про этих говорящих: это не беда файла —
        # ярлыками становятся имена, пометки в документе нет, код 0.
        strangers = _write_json(
            os.path.join(workdir, "чужие роли.json"),
            dict(
                _fixture_roles(),
                roles=[{"speaker": "Спикер A", "role": "Преподаватель", "share": 1.0, "source": "claude"}],
            ),
        )
        out = os.path.join(workdir, "чужие роли.md")
        code, _, stderr = _run(
            ["--punctuated", punctuated, "--summary", summary,
             "--speakers", good, "--roles", strangers, "--out", out]
        )
        _check(code == 0, "чужие роли вернули код %d, ожидался 0" % code)
        _check("не совпала" in stderr, "про несовпадение ролей и имён не предупредили: %s" % stderr)
        with open(out, "r", encoding="utf-8") as handle:
            doc = parse_document(handle.read())
        _check(not doc.notes, "при чужих ролях появилась пометка")
        first = LABEL_RE.match(_speaker_lines(doc)[0][1])
        _check(
            first is not None and first.group("label") == "Говорящий 1",
            "ярлыком не стало имя говорящего: %r" % _speaker_lines(doc)[0][1],
        )


def _test_speakers_note() -> None:
    """--speakers-note: пометка под заголовком, код 1, расшифровка предложениями."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        punctuated = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        speakers = _write_json(os.path.join(workdir, "speakers.json"), _fixture_speakers())
        roles = _write_json(os.path.join(workdir, "roles.json"), _fixture_roles())
        summary = os.path.join(workdir, "summary.md")
        write_text_file(summary, _fixture_summary())
        out = os.path.join(workdir, "пометка.md")

        reason = "диаризация не запускалась (--no-diarize)"
        code, report, _ = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out,
             "--speakers-note", reason]
        )
        _check(code == 1, "--speakers-note вернул код %d, ожидался 1" % code)
        with open(out, "r", encoding="utf-8") as handle:
            text = handle.read()
        doc = parse_document(text)
        _check(len(doc.notes) == 1, "пометок о ярлыках %d вместо одной" % len(doc.notes))
        _check(
            doc.notes[0] == "> ⚠️ Ярлыки говорящих не проставлены: %s." % reason,
            "пометка не того вида: %r" % doc.notes[0],
        )
        # Пометка стоит сразу под заголовком раздела и отделена пустой строкой.
        lines = text.rstrip("\n").split("\n")
        at = lines.index(doc.notes[0])
        _check(lines[at - 2] == TITLE_BODY, "над пометкой не заголовок расшифровки")
        _check(lines[at + 1] == "", "после пометки нет пустой строки")
        _check(
            text.replace(doc.notes[0] + "\n\n", "", 1) == PLAIN_DOCUMENT,
            "документ с пометкой отличается от обычного не только ею",
        )
        _check("Пометок:       1" in report, "в отчёте не та цифра пометок")
        _check(reason in report, "в отчёте нет причины: %s" % report)

        # Причина уже с точкой на конце не даёт двух точек подряд.
        out2 = os.path.join(workdir, "с точкой.md")
        code, _, _ = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out2,
             "--speakers-note", "claude недоступен."]
        )
        _check(code == 1, "вторая пометка вернула код %d" % code)
        with open(out2, "r", encoding="utf-8") as handle:
            doc2 = parse_document(handle.read())
        _check(
            doc2.notes[0].endswith("claude недоступен."),
            "в пометке две точки подряд: %r" % doc2.notes[0],
        )

        # Пометка сильнее файла: ярлыков не будет, даже если speakers годный.
        out3 = os.path.join(workdir, "вместе.md")
        code, _, stderr = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out3,
             "--speakers", speakers, "--roles", roles, "--speakers-note", reason]
        )
        _check(code == 1, "пометка вместе с --speakers вернула код %d" % code)
        _check("--speakers-note" in stderr, "про спор флагов не предупредили: %s" % stderr)
        with open(out3, "r", encoding="utf-8") as handle:
            text3 = handle.read()
        _check("Преподаватель:" not in text3, "при пометке в документ попали ярлыки")
        _check(
            text3.replace("> ⚠️ Ярлыки говорящих не проставлены: %s.\n\n" % reason, "", 1)
            == PLAIN_DOCUMENT,
            "документ при споре флагов собран не предложениями",
        )

        # Пустая причина — ошибка вызова, а не пометка без текста.
        code, _, stderr = _run(
            ["--punctuated", punctuated, "--out", os.path.join(workdir, "нет.md"),
             "--speakers-note", "   "]
        )
        _check(code == 2, "пустой --speakers-note вернул код %d, ожидался 2" % code)
        _check(not os.path.exists(os.path.join(workdir, "нет.md")), "пустая причина оставила файл")


def _test_without_speakers() -> None:
    """Без --speakers документ прежний — ни одного изменённого байта."""
    with tempfile.TemporaryDirectory(prefix="render-doc-selftest-") as workdir:
        punctuated = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        roles = _write_json(os.path.join(workdir, "roles.json"), _fixture_roles())
        summary = os.path.join(workdir, "summary.md")
        write_text_file(summary, _fixture_summary())

        out = os.path.join(workdir, "обычный.md")
        code, report, stderr = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out]
        )
        _check(code == 0, "обычный прогон вернул код %d (stderr: %s)" % (code, stderr.strip()))
        with open(out, "r", encoding="utf-8") as handle:
            text = handle.read()
        _check(text == PLAIN_DOCUMENT, "документ без --speakers изменился:\n%s" % text)
        _check("Предложений:   6" in report, "в отчёте не предложения, а что-то другое")

        # --roles без --speakers ничего не меняет: только предупреждение.
        out2 = os.path.join(workdir, "только роли.md")
        code, _, stderr = _run(
            ["--punctuated", punctuated, "--summary", summary, "--out", out2, "--roles", roles]
        )
        _check(code == 0, "--roles без --speakers вернул код %d" % code)
        _check("--roles без --speakers" in stderr, "про бесполезный --roles не предупредили")
        with open(out2, "r", encoding="utf-8") as handle:
            _check(handle.read() == PLAIN_DOCUMENT, "--roles без --speakers изменил документ")


def _test_help() -> None:
    """Справка — на русском, со всеми флагами, код 0."""
    for flag in ("-h", "--help"):
        code, out, _ = _run([flag])
        _check(code == 0, "%s вернул код %d" % (flag, code))
        for needle in (
            "--punctuated", "--summary", "--out", "--title", "--selftest",
            "--speakers", "--roles", "--speakers-note", "Использование",
        ):
            _check(needle in out, "в справке (%s) нет %r" % (flag, needle))
        _check(
            re.search(r"[а-яА-Я]", out) is not None,
            "справка не на русском",
        )


def selftest() -> int:
    """Прогоняет встроенные проверки. Код возврата 0 — все прошли."""
    tests = (
        ("формат таймкодов", _test_timecodes),
        ("обычный документ", _test_plain_document),
        ("документ с окном в откате", _test_fallback_document),
        ("отсутствующее саммари", _test_missing_summary),
        ("непригодный вход и коды возврата", _test_broken_input),
        ("неполный вход: абзацы, окна, source", _test_degraded_input),
        ("ярлыки говорящих в строках расшифровки", _test_speaker_labels),
        ("негодные speakers и roles", _test_speakers_degraded),
        ("пометка --speakers-note", _test_speakers_note),
        ("без --speakers документ прежний", _test_without_speakers),
        ("справка", _test_help),
    )
    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ render_doc.py ================\n")
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
        data = load_punctuated(opts["punctuated"])
        summary, note = read_summary(opts["summary"])
        if note is not None:
            # Нечитаемый фрагмент и вовсе не переданный --summary — не ошибка
            # вызова, но и не норма продуктового прогона: про них говорим.
            warn(note)
        title = opts["title"].strip() if opts["title"] else default_title(data.source, opts["out"])

        body, note, by_phrases = build_body(data, opts)

        # Документ собирается целиком до записи: лучше упасть с внятным
        # сообщением, чем оставить на диске половину результата.
        content, stats = render_document(data, title, summary, opts["summary"], body, note)
        stats.by_phrases = by_phrases
        write_text_file(opts["out"], content)
    except (UsageError, DocumentError) as exc:
        return die(str(exc))

    try:
        size = os.path.getsize(opts["out"])
    except OSError:
        size = len(content.encode("utf-8"))
    print_report(opts["out"], size, stats)

    # Код 1 — документ записан, но в нём есть пометки: сборочный скрипт обязан
    # это заметить и сказать человеку, что прогон вышел неполным.
    return 1 if stats.marks else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
