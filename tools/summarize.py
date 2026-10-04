#!/usr/bin/env python3
"""summarize.py — саммари записи одним вызовом claude.

На вход — файл схемы punctuated/1 (его пишет tools/punctuate.py: предложения с
таймкодами) или, как подстраховка, transcript/1 от tools/transcribe.py (там
дословный текст без пунктуации — саммари по нему хуже, но делается). На выход —
один markdown-фрагмент с восемью разделами уровня «###», всегда в этом порядке:

  ### Изложение                       — связный текст, 3–8 абзацев;
  ### Ключевые моменты                — «- [ЧЧ:ММ:СС] формулировка»;
  ### Темы                            — «- [ЧЧ:ММ:СС] название темы»;
  ### Домашние задания                — «- [ЧЧ:ММ:СС] задание — срок: …»;
  ### Сроки и дедлайны                — «- [ЧЧ:ММ:СС] что сделать — срок: …»;
  ### Решения, задачи, договорённости — «- [ЧЧ:ММ:СС] формулировка»;
  ### Термины и определения           — «- [ЧЧ:ММ:СС] термин — определение»;
  ### Материалы и ссылки              — «- [ЧЧ:ММ:СС] название или ссылка».

Порядок и формулировки восьми заголовков заданы владельцем дословно, менять их
нельзя: по ним человек читает два саммари подряд, не перестраиваясь. Таймкод
обязателен в каждой строке всех семи списочных разделов, включая термины и
материалы, — иначе пункт саммари нельзя сверить с записью.

Срок домашнего задания стоит в двух местах сразу: в самой строке задания и
отдельной строкой в разделе «Сроки и дедлайны». Это не дублирование от
небрежности: по списку заданий видно, что делать, а по списку сроков — что
горит, и смотрят в них в разное время. Срок цитируется так, как прозвучал
(«к следующему занятию»), а дата в скобках добавляется только если прямо названа
в записи; пересчёт относительных сроков в даты запрещён и промптом, и здесь —
сегодняшнего числа инструмент не знает, а дата, которой в записи не было, это
выдуманный факт в документе, по которому человек будет сдавать работу.

Что оставлено на усмотрение исполнителя и выбрано здесь: строка термина — это
«термин — определение, как прозвучало» (определение в той же строке, чтобы
раздел читался как глоссарий, а не как список слов); строка материала — это
«название или ссылка, как прозвучала» (без попыток разложить её на автора,
название и год: в расшифровке речи этого обычно и нет); отсутствующий срок
домашнего задания записывается как «срок: не назван» — именно «не назван», а не
«не указан» или прочерк, потому что это утверждение о записи, а не о задании:
срок мог быть, но в записи не прозвучал.

Заголовков уровня «#» и «##» во фрагменте нет сознательно: его вставляет в
итоговый документ tools/render_doc.py под готовый заголовок «## Саммари», и свой
заголовок верхнего уровня здесь испортил бы структуру документа.

Вызов claude — ровно один и по всему тексту сразу (решение владельца: «саммари
отдельным вызовом по всему тексту»). Карты-свёртки по частям здесь нет: саммари
по саммари теряет именно то, за чем его читают, — связь между началом и концом
разговора. Расплата за это — предел длины аргумента командной строки: промпт
полной двухчасовой записи может в него не влезть, и тогда claude_cli поднимает
ClaudeFailed, а инструмент честно пишет «не получено» с этой причиной.

Главное свойство: фрагмент пишется всегда, даже если claude недоступен, не
авторизован, не ответил или ответил не тем. Расшифровка стоит двух часов
распознавания, и терять её из-за саммари нельзя — поэтому «не получено» это
пометка в файле и код возврата 1, а не отсутствие файла. Код 2 оставлен для
непригодного входа: там --out не создаётся и существующий файл не перетирается.

Почему ответ claude не кладётся в файл дословно. Из восьми разделов собирается
ровно та структура, которую ждёт сборщик документа: заголовки приводятся к
«###», строки списков — к виду «- [ЧЧ:ММ:СС] текст». Слова при этом не
переписываются: меняется только разметка вокруг них. Если же все восемь разделов
в ответе не нашлись, ответ сохраняется как есть под «### Изложение» (с одной
поправкой: его собственные заголовки опускаются ниже «###» — этот уровень занят
разделами фрагмента, а «#» и «##» разорвали бы документ), в stderr уходит
предупреждение, а код возврата остаётся нулевым — саммари всё-таки получено.

Зависимости: только стандартная библиотека. Сам claude запускается через
tools/claude_cli.py — это единственное место в проекте, где собирается его
командная строка; своего subprocess здесь нет. Формат таймкода берётся оттуда
же, откуда его читает сборщик документа, — из tools/render_doc.py, чтобы
«[01:02:03]» в саммари и в расшифровке означало одно и то же.

Проверки живут в самом файле (--selftest): в окружении проекта нет pytest.
Установленный claude для них не нужен и намеренно не используется — все вызовы
идут на подставных sh-скриптах.

Использование:
  python tools/summarize.py --punctuated <файл.json> --out <файл.md>
  python tools/summarize.py --transcript <файл.json> --out <файл.md>
  python tools/summarize.py --selftest
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

# Каталог tools/ в sys.path: при запуске файла скриптом он там уже первый, но
# при импорте модуля откуда-то ещё — нет.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import claude_cli  # noqa: E402 — импорт возможен только после правки sys.path выше
from render_doc import format_timecode  # noqa: E402 — тот же формат, что в документе

PROG = "summarize.py"

PUNCTUATED_SCHEMA = "punctuated/1"
TRANSCRIPT_SCHEMA = "transcript/1"

# Заголовки разделов фрагмента: порядок и формулировки фиксированы, на них
# опирается и сборщик документа, и человек, читающий два саммари подряд.
SECTION_OVERVIEW = "### Изложение"
SECTION_KEY_POINTS = "### Ключевые моменты"
SECTION_TOPICS = "### Темы"
SECTION_HOMEWORK = "### Домашние задания"
SECTION_DEADLINES = "### Сроки и дедлайны"
SECTION_ACTIONS = "### Решения, задачи, договорённости"
SECTION_TERMS = "### Термины и определения"
SECTION_MATERIALS = "### Материалы и ссылки"

OVERVIEW = "overview"
HOMEWORK = "homework"
DEADLINES = "deadlines"
TOPICS = "topics"


@dataclass(frozen=True)
class Section:
    """Один раздел фрагмента.

    key     — внутреннее имя: им помечаются строки при разборе ответа;
    heading — заголовок во фрагменте, слово в слово;
    name    — название раздела для сообщений в stderr;
    label   — ярлык строки отчёта в stdout.
    """

    key: str
    heading: str
    name: str
    label: str


# Единственный список разделов: его порядок — это и есть порядок разделов во
# фрагменте, заданный владельцем. Всё остальное (промпт-образец, распознавание
# заголовков в ответе, рендер, отчёт) считается отсюда, чтобы разделы нельзя
# было рассогласовать правкой в одном месте.
SECTIONS: tuple[Section, ...] = (
    Section(OVERVIEW, SECTION_OVERVIEW, "Изложение", "Абзацев"),
    Section("key_points", SECTION_KEY_POINTS, "Ключевые моменты", "Ключевых моментов"),
    Section(TOPICS, SECTION_TOPICS, "Темы", "Тем"),
    Section(HOMEWORK, SECTION_HOMEWORK, "Домашние задания", "Домашних заданий"),
    Section(DEADLINES, SECTION_DEADLINES, "Сроки и дедлайны", "Сроков"),
    Section("actions", SECTION_ACTIONS, "Решения, задачи, договорённости", "Решений"),
    Section("terms", SECTION_TERMS, "Термины и определения", "Терминов"),
    Section("materials", SECTION_MATERIALS, "Материалы и ссылки", "Материалов"),
)

# Семь списочных разделов — все, кроме «Изложения»: в них строки «- [..] текст».
LIST_SECTIONS: tuple[Section, ...] = tuple(s for s in SECTIONS if s.key != OVERVIEW)
LIST_KEYS: tuple[str, ...] = tuple(s.key for s in LIST_SECTIONS)

# Строка, которой заполняется раздел, в котором нечего перечислять.
NOTHING = "- не прозвучало"

# Так записывается домашнее задание, срок которого в записи не прозвучал.
DEADLINE_UNKNOWN = "не назван"

# Строка для списков в случае, когда восемь разделов в ответе claude не нашлись
# и весь ответ ушёл в «Изложение»: врать «не прозвучало» здесь нельзя —
# неизвестно, прозвучало или нет.
NOT_PARSED = "- не выделено: восемь разделов в ответе claude не найдены, см. «Изложение»"

# Уровень, до которого опускаются заголовки, пришедшие из ответа claude:
# разделы фрагмента занимают «###», всё внутри них стоит ниже.
INNER_HEADING_LEVEL = 4

# Границы связного изложения из критериев приёмки. Выход за них не ошибка —
# переписывать текст claude инструмент не вправе, — но о нём говорится в stderr.
OVERVIEW_MIN_PARAGRAPHS = 3
OVERVIEW_MAX_PARAGRAPHS = 8

# Префикс имён файлов лога вызова (tools/claude_cli.py кладёт <tag>-NNN.*.txt).
LOG_TAG = "саммари"

HELP = """Использование:
  python tools/summarize.py --punctuated <файл.json> --out <файл.md> [опции]
  python tools/summarize.py --transcript <файл.json> --out <файл.md> [опции]
  python tools/summarize.py --selftest

Делает саммари записи одним вызовом claude по всему тексту и пишет
markdown-фрагмент с восемью разделами уровня «###», всегда в этом порядке:

  «Изложение»                        связный текст, 3–8 абзацев, без списков
                                     и таймкодов
  «Ключевые моменты»                 - [ЧЧ:ММ:СС] формулировка
  «Темы»                             - [ЧЧ:ММ:СС] название темы
  «Домашние задания»                 - [ЧЧ:ММ:СС] задание — срок: <как сказано
                                     в записи>; срок не прозвучал — в той же
                                     позиции «срок: {unknown}»
  «Сроки и дедлайны»                 - [ЧЧ:ММ:СС] что сделать — срок: <как
                                     сказано в записи>; каждый названный срок
                                     задания повторяется здесь строкой
  «Решения, задачи, договорённости»  - [ЧЧ:ММ:СС] формулировка
  «Термины и определения»            - [ЧЧ:ММ:СС] термин — определение
  «Материалы и ссылки»               - [ЧЧ:ММ:СС] название или ссылка

Таймкод обязателен в каждой строке всех семи списочных разделов: строка без
таймкода во фрагмент не попадает, о чём говорится в stderr с названием раздела.
Если в разделе перечислять нечего, в нём стоит одна строка «не прозвучало».
Заголовков «#» и «##» во фрагменте нет: он вставляется в итоговый документ под
готовый заголовок «## Саммари».

Срок цитируется так, как прозвучал в записи («к следующему занятию», «через две
недели»); дата в скобках добавляется только если она прямо названа в записи
(«до 15 октября»). Пересчёт относительных сроков в даты промпт запрещает:
сегодняшнего числа инструмент не знает.

В промпт текст подаётся строками «[ЧЧ:ММ:СС] предложение», и промпт требует
брать таймкоды только из этих строк. Сам claude запускается через
tools/claude_cli.py как `claude -p <промпт>`; модель не задаётся.

Опции:
  --punctuated <файл>  JSON схемы punctuated/1 от tools/punctuate.py
  --transcript <файл>  JSON схемы transcript/1 от tools/transcribe.py —
                       подстраховка на случай, если пунктуации ещё нет: текст
                       там дословный, саммари по нему хуже. При обоих флагах
                       берётся --punctuated
  --out <файл>         куда записать markdown-фрагмент (обязателен)
  --timeout <сек>      сколько ждать один вызов claude (по умолчанию {timeout:g})
  --attempts <N>       сколько попыток вызова делать (по умолчанию {attempts};
                       повтор бывает только на таймауте и ненулевом коде)
  --log-dir <каталог>  куда tools/claude_cli.py положит промпт и ответ
                       ({tag}-NNN.prompt.txt и {tag}-NNN.response.txt)
  --selftest           прогнать встроенные проверки и выйти (claude не нужен)
  -h, --help           показать эту справку

Каталог для --out создаётся при необходимости, файл пишется в UTF-8.

Коды возврата:
  0  саммари получено и записано (в том числе когда восемь разделов в ответе не
     нашлись и ответ сохранён как есть под «### Изложение»)
  1  саммари не получено (нет claude, не авторизован, таймаут, ненулевой код,
     пустой ответ, слишком длинный промпт) — файл записан с пометкой и причиной
  2  вход непригоден (нет файла, не та схема, нет предложений) или ошибка в
     аргументах — файл --out не создаётся и не перетирается

Примеры:
  python tools/summarize.py --punctuated data/output/лекция/punctuated.json \\
      --out data/output/лекция/summary.md --log-dir data/output/лекция/claude
  python tools/summarize.py --transcript data/output/лекция/transcript.json \\
      --out data/output/лекция/summary.md --timeout 600
""".format(
    timeout=claude_cli.DEFAULT_TIMEOUT,
    attempts=claude_cli.DEFAULT_ATTEMPTS,
    tag=LOG_TAG,
    unknown=DEADLINE_UNKNOWN,
)


class UsageError(Exception):
    """Ошибка в аргументах командной строки."""


class InputError(Exception):
    """Вход не читается, не той схемы или в нём нет предложений."""


def die(message: str, code: int = 2) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main().

    По умолчанию 2: и негодные аргументы, и непригодный вход означают «саммари
    не делалось вовсе». Код 1 в этом инструменте занят другим смыслом — файл
    записан, но саммари в нём нет.
    """
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return code


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но пользователь должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


# ------------------------------------------------------------ чтение входа ---


@dataclass
class Line:
    """Одна строка расшифровки для промпта: таймкод начала и текст."""

    start: float
    text: str


@dataclass
class Record:
    """То, что инструменту нужно от входного файла.

    duration — длина записи в секундах, если она в файле есть; по ней видно,
    что таймкод в ответе claude выдуман (ушёл за конец записи).
    """

    path: str
    schema: str
    source: str
    duration: float | None
    lines: list[Line]

    @property
    def last_second(self) -> float:
        """Последняя секунда, на которую вообще может ссылаться саммари."""
        tail = max((line.start for line in self.lines), default=0.0)
        if self.duration is not None:
            return max(tail, self.duration)
        return tail


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
            "%s: поле %s меньше нуля (%s) — таймкод считается от начала записи"
            % (where, key, value)
        )
    return float(value)


def _as_text(holder: dict, key: str, where: str) -> str:
    """Достаёт из словаря обязательную строку, иначе — понятная ошибка."""
    if key not in holder:
        raise InputError("%s: нет обязательного поля %s" % (where, key))
    value = holder[key]
    if not isinstance(value, str):
        raise InputError(
            "%s: поле %s должно быть строкой, получено: %s"
            % (where, key, _typename(value))
        )
    return value


def _optional_duration(data: dict) -> float | None:
    value = data.get("source_duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _parse_items(data: dict, key: str, what: str) -> list[dict]:
    """Общая проверка списка предложений или сегментов."""
    if key not in data:
        raise InputError("в файле нет поля %s" % key)
    items = data[key]
    if not isinstance(items, list):
        raise InputError(
            "поле %s должно быть списком, получено: %s" % (key, _typename(items))
        )
    if not items:
        raise InputError("список %s пуст — саммари делать не из чего" % key)
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise InputError(
                "%s №%d должно быть объектом JSON, получено: %s"
                % (what, position + 1, _typename(item))
            )
    return items


def _lines_from(items: list[dict], what: str) -> list[Line]:
    """Превращает предложения (или сегменты) в строки промпта.

    Пустой текст пропускается: строка из одного таймкода claude ничего не
    скажет, а падать из-за неё незачем. Если пустыми окажутся все — об этом
    скажет вызывающий, там это уже непригодный вход.
    """
    lines: list[Line] = []
    skipped = 0
    for position, item in enumerate(items):
        where = "%s №%d" % (what, position + 1)
        start = _as_number(item, "start", where)
        text = " ".join(_as_text(item, "text", where).split())
        if not text:
            skipped += 1
            continue
        lines.append(Line(start=start, text=text))
    if skipped:
        warn("в промпт не попало %d %s с пустым текстом" % (skipped, what))
    # Порядок по времени: саммари с таймкодами читается сверху вниз, и строки
    # промпта обязаны идти так же, даже если в файле они перепутаны.
    lines.sort(key=lambda line: line.start)
    return lines


def parse_punctuated(data: object, path: str) -> Record:
    """Разбирает JSON схемы punctuated/1: строка промпта на предложение."""
    if not isinstance(data, dict):
        raise InputError(
            "на верхнем уровне ожидается объект JSON, получено: %s" % _typename(data)
        )
    _check_schema(data, PUNCTUATED_SCHEMA)
    items = _parse_items(data, "sentences", "предложение")
    lines = _lines_from(items, "предложение")
    if not lines:
        raise InputError("во всех предложениях пустой текст — саммари делать не из чего")
    source = data.get("source")
    return Record(
        path=path,
        schema=PUNCTUATED_SCHEMA,
        source=source.strip() if isinstance(source, str) else "",
        duration=_optional_duration(data),
        lines=lines,
    )


def parse_transcript(data: object, path: str) -> Record:
    """Разбирает JSON схемы transcript/1: строка промпта на сегмент.

    Текст сегментов дословный, без пунктуации и заглавных: саммари по нему хуже,
    поэтому режим и назван подстраховкой. Пословные таймкоды из поля words здесь
    не нужны — таймкод строки это начало сегмента.
    """
    if not isinstance(data, dict):
        raise InputError(
            "на верхнем уровне ожидается объект JSON, получено: %s" % _typename(data)
        )
    _check_schema(data, TRANSCRIPT_SCHEMA)
    items = _parse_items(data, "segments", "сегмент")
    lines = _lines_from(items, "сегмент")
    if not lines:
        raise InputError("во всех сегментах пустой текст — саммари делать не из чего")
    source = data.get("source")
    return Record(
        path=path,
        schema=TRANSCRIPT_SCHEMA,
        source=source.strip() if isinstance(source, str) else "",
        duration=_optional_duration(data),
        lines=lines,
    )


def _check_schema(data: dict, expected: str) -> None:
    if "schema" not in data:
        raise InputError("в файле нет поля schema (ожидается %r)" % expected)
    if data["schema"] != expected:
        raise InputError("не та схема: ожидается %r, в файле %r" % (expected, data["schema"]))


def load_record(path: str, schema: str) -> Record:
    """Читает входной файл. Любая беда на входе — InputError, не трейсбек."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise InputError("входной файл не найден: %s" % path) from None
    except IsADirectoryError:
        raise InputError("вместо входного файла указан каталог: %s" % path) from None
    except PermissionError:
        raise InputError("входной файл недоступен для чтения: %s" % path) from None
    except UnicodeDecodeError:
        raise InputError("входной файл не в кодировке UTF-8: %s" % path) from None
    except json.JSONDecodeError as exc:
        raise InputError(
            "входной файл не разбирается как JSON: %s (строка %d, символ %d: %s)"
            % (path, exc.lineno, exc.colno, exc.msg)
        ) from None
    except OSError as exc:
        raise InputError(
            "не удалось прочитать входной файл: %s (%s)" % (path, exc.strerror or exc)
        ) from None

    if schema == PUNCTUATED_SCHEMA:
        return parse_punctuated(data, path)
    return parse_transcript(data, path)


# ----------------------------------------------------------------- промпт ---

PROMPT_HEAD = """Сделай саммари рабочей записи (лекция или рабочий созвон).

Ниже — расшифровка записи: каждая строка это одно высказывание с таймкодом в
начале, вида «[ЧЧ:ММ:СС] текст». Речь русская, в тексте встречаются имена
коллег и названия продуктов.

Требования к ответу:

1. Отвечай по-русски. Кроме самого саммари в ответе не должно быть ничего: ни
   вступления, ни пояснений, ни замечаний о том, как ты его делал.
2. Опирайся только на расшифровку ниже. Не добавляй фактов, выводов, имён,
   названий, цифр и сроков, которых в ней нет. Чего в записи не было — того в
   саммари нет. Если что-то в тексте непонятно, так и напиши, но не додумывай.
3. Таймкоды бери ТОЛЬКО из начала строк расшифровки — ровно те, что стоят в
   строках, к которым относится твоя формулировка. Своих таймкодов не выдумывай,
   не округляй их и не пересчитывай: таймкод, которого нет ни в одной строке
   расшифровки, — ошибка.
4. Ответ состоит ровно из восьми разделов, в этом порядке и с этими
   заголовками, слово в слово:

### Изложение

Связный текст от 3 до 8 абзацев: о чём была запись, что обсуждали, какие
доводы приводили и к чему пришли. Без списков, без таймкодов, без заголовков
внутри раздела.

### Ключевые моменты

Главное из записи: мысли, выводы и факты, которые нужно запомнить, в порядке их
появления, по строке на пункт, строго в таком виде:
- [ЧЧ:ММ:СС] формулировка

### Темы

Темы записи в порядке их появления, по строке на тему, строго в таком виде:
- [ЧЧ:ММ:СС] название темы
Таймкод — начало обсуждения темы, взятый из строки расшифровки.

### Домашние задания

Всё, что в записи задано сделать самостоятельно, по строке на задание, строго в
таком виде:
- [ЧЧ:ММ:СС] задание — срок: как сказано в записи
Срок цитируй теми же словами, какими он прозвучал: «к следующему занятию»,
«через две недели», «до конца семестра». Дату в скобках добавляй только если она
прямо названа в записи: «к следующему занятию (до 15 октября)». Пересчитывать
относительные сроки в даты запрещено — сегодняшнего числа ты не знаешь, и дата,
которой в записи не было, это ошибка. Если срок задания в записи не назван,
напиши в той же позиции ровно «срок: не назван».

### Сроки и дедлайны

Все названные в записи сроки, по строке на срок, строго в таком виде:
- [ЧЧ:ММ:СС] что сделать — срок: как сказано в записи
Каждый названный срок домашнего задания повтори здесь отдельной строкой, тем же
текстом срока, что и в разделе «Домашние задания». Сроки, у которых задания не
было (сдача проекта, экзамен, следующая встреча), тоже идут сюда. Правило про
цитирование срока и запрет пересчёта действует и здесь.

### Решения, задачи, договорённости

Принятые решения, поставленные задачи и договорённости, по строке на пункт,
строго в таком виде:
- [ЧЧ:ММ:СС] формулировка

### Термины и определения

Термины, названия и сокращения, которые в записи объясняли, по строке на термин,
строго в таком виде:
- [ЧЧ:ММ:СС] термин — определение, как прозвучало

### Материалы и ссылки

Книги, статьи, ссылки, инструменты и прочие материалы, названные в записи, по
строке на материал, строго в таком виде:
- [ЧЧ:ММ:СС] название или ссылка, как прозвучала

5. В каждом из семи списочных разделов (все, кроме «Изложения») строка
   начинается с таймкода в квадратных скобках: «- [ЧЧ:ММ:СС] текст». Строка без
   таймкода в саммари не попадёт.
6. Если в разделе перечислять нечего, напиши в нём ровно одну строку:
- не прозвучало
   Заголовок раздела при этом всё равно нужен: разделов в ответе всегда восемь.
7. Заголовков уровня «#» и «##» в ответе быть не должно: только «###» у восьми
   разделов выше.

Расшифровка записи:

"""


def render_line(line: Line) -> str:
    """Строка расшифровки для промпта: «[ЧЧ:ММ:СС] текст»."""
    return "[%s] %s" % (format_timecode(line.start), line.text)


def build_prompt(record: Record) -> str:
    """Собирает промпт одного вызова: требования и вся расшифровка целиком."""
    body = "\n".join(render_line(line) for line in record.lines)
    return PROMPT_HEAD + body + "\n"


# ------------------------------------------------------- разбор ответа ---

# Заголовок markdown любого уровня. Пробел после решёток обязателен, иначе
# «#тема» в тексте сошло бы за заголовок.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")

# Жирная строка целиком — claude иногда помечает разделы так вместо заголовка.
BOLD_RE = re.compile(r"^\*\*(.+?)\*\*:?$")

# Таймкод в строке списка: сначала ищем ЧЧ:ММ:СС, потом ММ:СС. Квадратные скобки
# необязательны — claude мог их потерять, а выбрасывать из-за этого тему жалко.
TIME_HMS_RE = re.compile(r"\[?\s*(\d{1,4}):([0-5]?\d):([0-5]?\d)\s*\]?")
TIME_MS_RE = re.compile(r"\[?\s*(\d{1,4}):([0-5]?\d)\s*\]?")

# Маркеры списка и знаки, которыми claude отделяет таймкод от текста.
BULLET_RE = re.compile(r"^\s*(?:[-*•+]|\d{1,3}[.)])\s*")
SEPARATOR_RE = re.compile(r"^\s*[—–:-]+\s*")

# Метка срока в строке домашнего задания или дедлайна: «— срок:», «, срок —»,
# «дедлайн:». По ней строка делится на задание и срок, чтобы собрать её заново в
# виде «задание — срок: <как сказано>». \b после слова нужен, чтобы «сроки
# проектов» в тексте задания за метку не сошли.
DEADLINE_MARK_RE = re.compile(
    r"\s*[—–,;:-]*\s*(?:срок|дедлайн)\b\s*(?:сдачи|выполнения)?\s*[:—–-]*\s*",
    re.IGNORECASE,
)

# Знаки в конце обрезка строки, которые после деления по метке срока висят в
# воздухе: «сделать задание, срок: пятница» -> задание без запятой на конце.
TRAILING_RE = re.compile(r"[\s.,;:—–-]+$")

# Готовая строка домашнего задания — ею же проверяется, что срок назван.
HOMEWORK_LINE_RE = re.compile(r"^- \[[^\]]+\] (?P<task>.+?) — срок: (?P<deadline>.+)$")


def _section_key(text: str) -> str:
    """Название раздела -> ключ сравнения: только буквы и цифры, ё -> е."""
    lowered = text.strip().lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", lowered)


# Названия разделов, приведённые к виду «только буквы, строчные, без ё»: по ним
# заголовок раздела узнаётся в ответе claude, как бы он его ни разметил.
SECTION_KEYS = {_section_key(section.name): section.key for section in SECTIONS}

# Так claude говорит «перечислять нечего»: такая строка не идёт в список как
# пункт, вместо неё раздел получает единственную строку NOTHING.
NOTHING_KEYS = {"непрозвучало", "ничего", "ничегонепрозвучало", "нет", "небыло"}

# Так claude говорит «срок не назван»: любая из этих формулировок приводится к
# одной — DEADLINE_UNKNOWN, чтобы строки заданий читались одинаково.
UNKNOWN_DEADLINE_KEYS = {
    "неназван",
    "неназвано",
    "неназвана",
    "неуказан",
    "неуказано",
    "неназначен",
    "неизвестен",
    "неизвестно",
    "непрозвучал",
    "непрозвучало",
    "отсутствует",
    "нет",
    "",
}


def section_of(line: str) -> str | None:
    """Это строка-заголовок одного из восьми разделов? Тогда какого.

    Заголовок распознаётся в трёх видах: markdown-заголовком любого уровня,
    жирной строкой и просто строкой с названием раздела. Разметку claude
    выбирает сам, а разделы должны найтись и в том случае, если он поставил
    «## Темы» или «**Темы**» вместо «### Темы».
    """
    text = line.strip()
    if not text:
        return None
    heading = HEADING_RE.match(text)
    if heading is not None:
        text = heading.group(2)
    else:
        bold = BOLD_RE.match(text)
        if bold is not None:
            text = bold.group(1)
        elif any(mark in text for mark in "#*_`[>"):
            # Строка с разметкой, которая не разобралась ни как заголовок, ни как
            # жирный текст: «#темы» в тексте заголовком раздела не считается.
            return None
    return SECTION_KEYS.get(_section_key(text))


def demote_heading(line: str, minimum: int = INNER_HEADING_LEVEL) -> str:
    """Опускает заголовок до уровня не выше minimum решёток.

    Зачем это нужно. Фрагмент вставляется в документ под готовый «## Саммари»,
    поэтому «#» и «##» внутри него разорвали бы документ на части. А уровень
    «###» занят восьмью разделами самого фрагмента: заголовок, пришедший из ответа
    claude, обязан стоять ниже них, иначе читатель (и сборщик документа) не
    отличит его от начала раздела. Текст заголовка при этом не меняется —
    меняется только число решёток.
    """
    heading = HEADING_RE.match(line.strip())
    if heading is None:
        return line.rstrip()
    level = max(len(heading.group(1)), minimum)
    return "%s %s" % ("#" * level, heading.group(2))


def parse_timecode(text: str) -> tuple[float, str] | None:
    """Вытаскивает из строки таймкод и остаток текста.

    Возвращает (секунды, текст без таймкода) или None, если таймкода нет.
    Понимает «[ЧЧ:ММ:СС]», «ЧЧ:ММ:СС», «[ММ:СС]» и «ММ:СС»: лишний формат
    дешевле, чем потерянная тема из-за забытых скобок.
    """
    match = TIME_HMS_RE.search(text)
    if match is not None:
        hours, minutes, seconds = (int(match.group(number)) for number in (1, 2, 3))
        total = float(hours * 3600 + minutes * 60 + seconds)
    else:
        match = TIME_MS_RE.search(text)
        if match is None:
            return None
        minutes, seconds = int(match.group(1)), int(match.group(2))
        total = float(minutes * 60 + seconds)

    rest = text[: match.start()] + " " + text[match.end():]
    rest = BULLET_RE.sub("", rest, count=1)
    rest = SEPARATOR_RE.sub("", rest, count=1)
    return total, " ".join(rest.split())


@dataclass
class Summary:
    """Разобранное саммари — то, из чего собирается фрагмент.

    overview — абзацы «Изложения», lists — строки семи списочных разделов по
    ключу раздела. Отдельного поля на раздел нет сознательно: разделов восемь, и
    поимённое перечисление их в полях, в рендере, в отчёте и в пометке «не
    получено» разошлось бы при первой же правке состава.
    """

    overview: list[str] = field(default_factory=list)
    lists: dict[str, list[str]] = field(default_factory=dict)
    parsed: bool = True
    missing: str = ""

    @property
    def received(self) -> bool:
        return not self.missing

    def items(self, key: str) -> list[str]:
        """Строки списочного раздела по его ключу."""
        return self.lists.get(key, [])


def _split_blocks(lines: list[str]) -> list[list[str]]:
    """Делит строки на блоки по пустым строкам."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _overview_paragraphs(lines: list[str]) -> list[str]:
    """Изложение -> абзацы по одной строке на абзац.

    Перенос строк внутри абзаца снимается: слова не меняются, но абзац в файле
    становится одной строкой — так его проще и читать, и разбирать построчно.
    Заголовок, если claude его внутрь раздела всё же поставил, опускается ниже
    «###»: этот уровень занят восемью разделами самого фрагмента.
    """
    paragraphs: list[str] = []
    for block in _split_blocks(lines):
        if len(block) == 1 and HEADING_RE.match(block[0].strip()) is not None:
            paragraphs.append(demote_heading(block[0]))
            continue
        text = " ".join(" ".join(row.split()) for row in block).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _squeeze(text: str) -> str:
    """Один пробел между словами и никаких висящих знаков на конце."""
    return TRAILING_RE.sub("", " ".join(text.split()))


def homework_body(body: str) -> str | None:
    """Текст пункта ДЗ -> «задание — срок: <как сказано>». None — задания нет.

    Срок берётся из самой строки claude: он уже обязан стоять там по промпту, но
    отделён может быть чем угодно — тире, запятой, двоеточием. Текст срока при
    этом не переписывается (кроме формулировок вида «не указан» — они приводятся
    к одной), а отсутствующий становится «срок: не назван»: в этой позиции
    строки всегда что-то стоит, иначе список заданий нельзя читать по колонке.

    Меток в строке может оказаться несколько — «прочитать главу про сроки сдачи
    — срок: к пятнице». Берётся последняя, слева и справа от которой есть текст:
    слово «срок» внутри самого задания делит строку так, что справа остаётся
    вторая метка, и по этому признаку ложное деление отличается от настоящего.
    """
    chosen: tuple[str, str] | None = None
    last: tuple[str, str] | None = None
    for match in DEADLINE_MARK_RE.finditer(body):
        task = _squeeze(body[: match.start()])
        deadline = _squeeze(body[match.end():])
        last = (task, deadline)
        if task and deadline:
            chosen = (task, deadline)
    task, deadline = chosen or last or (_squeeze(body), "")
    if not task:
        # Остался один срок без задания: вписывать его в список ДЗ нечем, а
        # придумывать задание инструмент не вправе.
        return None
    if _section_key(deadline) in UNKNOWN_DEADLINE_KEYS:
        deadline = DEADLINE_UNKNOWN
    return "%s — срок: %s" % (task, deadline)


def _list_items(lines: list[str], record: Record, section: Section) -> list[str]:
    """Строки списочного раздела -> строки «- [ЧЧ:ММ:СС] текст».

    Строки без таймкода выбрасываются: это либо «Основные темы:» перед списком,
    либо пункт, у которого таймкода нет вовсе, а выдумать его инструмент не
    вправе. Сколько выброшено и из какого раздела — говорится в stderr.
    """
    items: list[str] = []
    dropped = 0
    limit = record.last_second
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if _section_key(text) in NOTHING_KEYS:
            # Claude сказал, что перечислять нечего: ниже это станет NOTHING.
            continue
        found = parse_timecode(text)
        if found is None:
            dropped += 1
            continue
        seconds, body = found
        if section.key == HOMEWORK and body:
            body = homework_body(body) or ""
        if not body:
            dropped += 1
            continue
        if limit and seconds > limit + 1:
            warn(
                "%s: таймкод %s за пределами записи (%s) — строка оставлена как есть"
                % (section.name, format_timecode(seconds), format_timecode(limit))
            )
        items.append("- [%s] %s" % (format_timecode(seconds), body))
    if dropped:
        warn(
            "%s: %d строк без таймкода не попали во фрагмент" % (section.name, dropped)
        )
    return items


def _check_deadlines_repeated(homework: list[str], deadlines: list[str]) -> None:
    """Названный срок ДЗ обязан повториться строкой в «Сроках и дедлайнах».

    Этого требует промпт, но проверить его исполнение можно только здесь.
    Переносить срок в раздел сроков самостоятельно инструмент не станет: строка
    там — это «что сделать к сроку», а составить её из строки задания — значит
    написать за claude текст, которого в ответе не было. Поэтому только stderr.
    """
    said = " ".join(_section_key(line) for line in deadlines)
    lost: list[str] = []
    for line in homework:
        match = HOMEWORK_LINE_RE.match(line)
        if match is None:
            continue
        deadline = match.group("deadline")
        if deadline == DEADLINE_UNKNOWN:
            continue
        if _section_key(deadline) not in said:
            lost.append(deadline)
    if lost:
        warn(
            "срок%s %s назван в «%s», но не повторён строкой в «%s»"
            % (
                "и" if len(lost) > 1 else "",
                ", ".join("«%s»" % text for text in lost),
                SECTION_HOMEWORK[4:],
                SECTION_DEADLINES[4:],
            )
        )


def parse_answer(answer: str, record: Record) -> Summary:
    """Разбирает ответ claude на восемь разделов.

    Нашлись не все восемь заголовков — ответ уходит в «Изложение» как есть, а
    Summary.parsed становится ложью: вызывающий по этому печатает предупреждение
    и всё равно возвращает ноль. Саммари получено, пусть и не той формы.
    """
    buckets: dict[str, list[str]] = {section.key: [] for section in SECTIONS}
    seen: set[str] = set()
    current: str | None = None
    preamble = 0
    for line in answer.splitlines():
        found = section_of(line)
        if found is not None:
            current = found
            seen.add(found)
            continue
        if current is None:
            if line.strip():
                preamble += 1
            continue
        buckets[current].append(line)

    # Считаются именно заголовки, а не их наполнение: раздел, который claude
    # объявил и оставил пустым, — это всё-таки найденный раздел, и ниже он
    # получит пометку «не прозвучало», а не утянет весь ответ в «Изложение».
    if len(seen) < len(SECTIONS):
        text = [demote_heading(line) if line.strip().startswith("#") else line.rstrip()
                for line in answer.splitlines()]
        blocks = ["\n".join(block) for block in _split_blocks(text)]
        return Summary(
            overview=blocks or [answer.strip()],
            lists={key: [NOT_PARSED] for key in LIST_KEYS},
            parsed=False,
        )

    if preamble:
        warn("в ответе claude было %d строк до первого раздела — они отброшены" % preamble)

    overview = _overview_paragraphs(buckets[OVERVIEW])
    lists: dict[str, list[str]] = {}
    for section in LIST_SECTIONS:
        items = _list_items(buckets[section.key], record, section)
        if not items:
            if section.key == TOPICS:
                # Запись без тем — это не «в ней их не было», а негодный ответ:
                # про остальные разделы такого не скажешь, поэтому предупреждение
                # только здесь.
                warn("раздел «%s» в ответе claude пуст" % section.name)
            items = [NOTHING]
        lists[section.key] = items
    _check_deadlines_repeated(lists[HOMEWORK], lists[DEADLINES])

    if not overview:
        warn("раздел «Изложение» в ответе claude пуст")
        overview = ["Изложение не получено: claude оставил раздел пустым."]
    elif not OVERVIEW_MIN_PARAGRAPHS <= len(overview) <= OVERVIEW_MAX_PARAGRAPHS:
        warn(
            "в изложении %d абзацев вместо %d–%d — текст claude оставлен как есть"
            % (len(overview), OVERVIEW_MIN_PARAGRAPHS, OVERVIEW_MAX_PARAGRAPHS)
        )

    return Summary(overview=overview, lists=lists, parsed=True)


def missing_summary(reason: str) -> Summary:
    """Саммари не получено: тот же фрагмент, но из одной пометки с причиной."""
    text = " ".join(str(reason).split()).rstrip(".")
    note = "Саммари не получено: %s." % (text or "причина не названа")
    return Summary(
        overview=[note],
        lists={key: ["- %s" % note] for key in LIST_KEYS},
        parsed=False,
        missing=note,
    )


# ----------------------------------------------------------------- рендер ---


def render_fragment(summary: Summary) -> str:
    """Собирает markdown-фрагмент: восемь разделов «###» в порядке SECTIONS."""
    blocks: list[str] = []
    for section in SECTIONS:
        blocks.append(section.heading)
        if section.key == OVERVIEW:
            blocks.extend(summary.overview)
        else:
            blocks.append("\n".join(summary.items(section.key) or [NOTHING]))
    return "\n\n".join(blocks) + "\n"


def write_text_file(path: str, content: str) -> None:
    """Пишет файл в UTF-8 с переводами строк LF.

    newline="\\n" задан явно: на Windows иначе получились бы CRLF, и «ровно одна
    пустая строка между разделами» перестала бы быть правдой.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except OSError as exc:
        raise InputError(
            "не удалось записать файл %s (%s)" % (path, exc.strerror or exc)
        ) from None


# Ширина колонки ярлыков в отчёте: считается по самому длинному ярлыку, чтобы
# цифры стояли в колонку и после правки состава разделов.
REPORT_WIDTH = max([len(section.label) for section in SECTIONS] + [len("Саммари")]) + 2


def report_row(label: str, value: str) -> str:
    """Строка отчёта: ярлык с двоеточием, выровненный по колонке, и значение."""
    return "%-*s%s\n" % (REPORT_WIDTH, label + ":", value)


def print_report(path: str, size: int, record: Record, summary: Summary) -> None:
    """Печатает итоги — то, что потом уходит в отчёт о прогоне."""
    sys.stdout.write("================ САММАРИ ================\n")
    sys.stdout.write(report_row("Файл", path))
    sys.stdout.write(report_row("Размер", "%d байт" % size))
    sys.stdout.write(report_row("Вход", "%s (%s)" % (record.path, record.schema)))
    sys.stdout.write(report_row("Строк", "%d" % len(record.lines)))
    if summary.received:
        # Пунктов по каждому из семи списочных разделов: по этим цифрам в логе
        # прогона видно, что именно саммари вытащило из записи.
        for section in SECTIONS:
            if section.key == OVERVIEW:
                sys.stdout.write(report_row(section.label, "%d" % len(summary.overview)))
            else:
                sys.stdout.write(
                    report_row(section.label, "%d" % len(summary.items(section.key)))
                )
        if not summary.parsed:
            sys.stdout.write("  * восемь разделов в ответе не найдены, ответ сохранён как есть\n")
    else:
        sys.stdout.write(report_row("Саммари", "не получено"))
        sys.stdout.write("  * %s\n" % summary.missing)


# --------------------------------------------------------- разбор аргументов ---

_FLAGS_VALUE = {
    "--punctuated": "punctuated",
    "--transcript": "transcript",
    "--out": "out",
    "--timeout": "timeout",
    "--attempts": "attempts",
    "--log-dir": "log_dir",
}


def parse_args(argv: list[str]) -> dict:
    """Разбирает аргументы так же, как это делают остальные инструменты проекта.

    Своим разбором, а не argparse: тот печатает usage и свои ошибки по-английски,
    а весь остальной вывод инструментов проекта — на русском.
    """
    opts: dict = {
        "punctuated": None,
        "transcript": None,
        "out": None,
        "timeout": claude_cli.DEFAULT_TIMEOUT,
        "attempts": claude_cli.DEFAULT_ATTEMPTS,
        "log_dir": None,
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
            if name == "timeout":
                opts[name] = _positive_float(token, value)
            elif name == "attempts":
                opts[name] = _positive_int(token, value)
            else:
                opts[name] = value
            continue

        if token.startswith("-"):
            raise UsageError("неизвестная опция: %s (см. --help)" % token)
        raise UsageError("лишний аргумент: %s (см. --help)" % token)

    return opts


def _positive_float(token: str, value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise UsageError("%s ожидает число секунд, получено: %s" % (token, value)) from None
    if not math.isfinite(number) or number <= 0:
        raise UsageError("%s должен быть больше нуля, получено: %s" % (token, value))
    return number


def _positive_int(token: str, value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise UsageError("%s ожидает целое число, получено: %s" % (token, value)) from None
    if number < 1:
        raise UsageError("%s должно быть не меньше 1, получено: %s" % (token, value))
    return number


def check_args(opts: dict) -> tuple[str, str]:
    """Проверяет набор аргументов целиком и выбирает вход.

    Возвращает (путь, схема). При обоих флагах берётся --punctuated: там текст с
    пунктуацией, и саммари по нему лучше.
    """
    if not opts["punctuated"] and not opts["transcript"]:
        raise UsageError("не указан ни --punctuated, ни --transcript (см. --help)")
    if not opts["out"]:
        raise UsageError("не указан --out (см. --help)")

    if opts["punctuated"]:
        if opts["transcript"]:
            warn("указаны и --punctuated, и --transcript: взят --punctuated")
        path, schema = opts["punctuated"], PUNCTUATED_SCHEMA
    else:
        path, schema = opts["transcript"], TRANSCRIPT_SCHEMA

    # Перетереть собственный вход — это потерять прогон целиком: заново
    # распознавать запись нечем, а саммари делается в секунды.
    if os.path.abspath(opts["out"]) == os.path.abspath(path):
        raise UsageError("--out указывает на тот же файл, что и вход: %s" % opts["out"])
    return path, schema


# -------------------------------------------------------------------- main ---


def make_summary(record: Record, opts: dict) -> Summary:
    """Один вызов claude и разбор ответа. Неудача — это Summary с пометкой."""
    prompt = build_prompt(record)
    try:
        answer = claude_cli.ask(
            prompt,
            timeout=opts["timeout"],
            attempts=opts["attempts"],
            log_dir=opts["log_dir"],
            tag=LOG_TAG,
        )
    except claude_cli.ClaudeUnavailable as exc:
        return missing_summary("вызывать claude нечем: %s" % exc)
    except claude_cli.ClaudeFailed as exc:
        return missing_summary("claude не ответил: %s" % exc)
    return parse_answer(answer, record)


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
        path, schema = check_args(opts)
        record = load_record(path, schema)
    except (UsageError, InputError) as exc:
        # Вход непригоден: файл --out не создаётся и не перетирается.
        return die(str(exc))

    if schema == TRANSCRIPT_SCHEMA:
        warn("вход — транскрипт без пунктуации: саммари по нему хуже, чем по punctuated/1")

    summary = make_summary(record, opts)
    try:
        write_text_file(opts["out"], render_fragment(summary))
    except InputError as exc:
        # Вход был годен и саммари, возможно, получено — не записался сам файл.
        # Это не код 2: вход тут ни при чём, а результата всё равно нет.
        return die("%s — фрагмент не записан" % exc, 1)

    try:
        size = os.path.getsize(opts["out"])
    except OSError:  # pragma: no cover — файл только что записан
        size = 0
    print_report(opts["out"], size, record, summary)

    if not summary.received:
        sys.stderr.write("%s: %s\n" % (PROG, summary.missing))
        return 1
    if not summary.parsed:
        warn(
            "восемь разделов в ответе claude не найдены — ответ сохранён как есть под "
            "«%s»" % SECTION_OVERVIEW
        )
    return 0


# ---------------------------------------------------------------- селф-тест ---
#
# Установленный claude здесь не нужен и намеренно не используется: все вызовы
# идут на подставных sh-скриптах, путь к которым подсовывается через CLAUDE_BIN.
# Каждая заглушка лежит в своём каталоге и пишет рядом с собой, сколько раз её
# позвали и с какими аргументами, — так проверяется и число вызовов, и промпт.

STUB_PREAMBLE = """#!/bin/sh
dir=${0%/*}
printf 'x\\n' >> "$dir/calls.txt"
: > "$dir/argv.txt"
for a in "$@"; do printf '%s\\n' "$a" >> "$dir/argv.txt"; done
"""

# Строка фрагмента со темой или решением — ровно того вида, что требуют критерии.
ITEM_RE = re.compile(r"^- \[(\d{2,}):([0-5]\d):([0-5]\d)\] (\S.*)$")

# Восемь заголовков фрагмента, выписанные литералами и в том порядке, который
# задал владелец. Сверять фрагмент с SECTIONS бессмысленно: из неё он и собран, и
# переставленные там разделы такая проверка пропустила бы. Эта строка — и есть
# требование задачи, и менять её без нового решения владельца нельзя.
EXPECTED_HEADINGS = (
    "### Изложение",
    "### Ключевые моменты",
    "### Темы",
    "### Домашние задания",
    "### Сроки и дедлайны",
    "### Решения, задачи, договорённости",
    "### Термины и определения",
    "### Материалы и ссылки",
)


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _stub(root: str, name: str, body: str) -> str:
    """Создаёт каталог с исполняемой заглушкой `claude` внутри."""
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "claude")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(STUB_PREAMBLE + body)
    os.chmod(path, 0o755)
    return path


def _answering_stub(root: str, name: str, answer: str) -> str:
    """Заглушка, печатающая заранее заданный ответ из файла рядом с собой."""
    stub = _stub(root, name, 'cat "$dir/answer.txt"\nexit 0\n')
    with open(os.path.join(os.path.dirname(stub), "answer.txt"), "w", encoding="utf-8") as h:
        h.write(answer)
    return stub


def _calls(stub: str) -> int:
    counter = os.path.join(os.path.dirname(stub), "calls.txt")
    if not os.path.exists(counter):
        return 0
    with open(counter, "r", encoding="utf-8") as handle:
        return len([row for row in handle.read().splitlines() if row])


def _prompt_of(stub: str) -> str:
    """Промпт, с которым позвали заглушку: второй аргумент после -p."""
    recorded = os.path.join(os.path.dirname(stub), "argv.txt")
    with open(recorded, "r", encoding="utf-8") as handle:
        text = handle.read()
    _check(text.startswith("-p\n"), "команда не начинается с -p: %r" % text[:20])
    return text[len("-p\n"):]


def _run(argv: list[str], binary: str | None = None) -> tuple[int, str, str]:
    """Сквозной прогон main() с перехватом вывода и подменой CLAUDE_BIN."""
    saved = os.environ.get("CLAUDE_BIN")
    if binary is not None:
        os.environ["CLAUDE_BIN"] = binary
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
    finally:
        if binary is not None:
            if saved is None:
                os.environ.pop("CLAUDE_BIN", None)
            else:
                os.environ["CLAUDE_BIN"] = saved
    return code, stdout.getvalue(), stderr.getvalue()


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_json(path: str, payload: object) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _fixture_punctuated() -> dict:
    """Файл схемы punctuated/1 для проверок: собирается в памяти, без claude.

    Последнее предложение начинается на 9329-й секунде — это 02:35:29, случай
    «два с половиной часа» из критериев приёмки.
    """
    sentences = [
        (0.0, 6.5, "Добрый день, начинаем разбор формата итогового документа."),
        (6.5, 14.25, "Сначала обсудим расшифровку, потом саммари."),
        (600.0, 609.5, "Таймкод у каждого предложения, это решение уже принято."),
        (3723.0, 3731.0, "Через час вернулись к нарезке текста на окна."),
        (9329.4, 9335.0, "На этом закончили, задачи разошлись по исполнителям."),
    ]
    return {
        "schema": PUNCTUATED_SCHEMA,
        "source": "data/output/созвон команды/созвон команды.webm",
        "source_duration": 9340.0,
        "transcript": "data/output/созвон команды/transcript.json",
        "model": "v3_rnnt",
        "windows": [{"index": 0, "segments": [0, 142], "status": "ok", "reason": ""}],
        "paragraphs": [{"index": 0, "window": 0, "sentences": [0, len(sentences) - 1]}],
        "sentences": [
            {
                "index": number,
                "start": start,
                "end": end,
                "text": text,
                "window": 0,
                "fallback": False,
            }
            for number, (start, end, text) in enumerate(sentences)
        ],
        "text": " ".join(text for _, _, text in sentences),
        "stats": {
            "windows": 1,
            "fallback_windows": 0,
            "paragraphs": 1,
            "sentences": len(sentences),
            "words": sum(len(text.split()) for _, _, text in sentences),
        },
    }


def _fixture_transcript() -> dict:
    """Файл схемы transcript/1: дословный текст без пунктуации, по сегментам."""
    segments = [
        (0.0, 20.0, "добрый день начинаем разбор формата итогового документа"),
        (600.0, 620.0, "таймкод у каждого предложения это решение уже принято"),
        (9329.4, 9340.0, "на этом закончили задачи разошлись по исполнителям"),
    ]
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "source": "data/output/созвон команды/созвон команды.wav",
        "source_duration": 9340.0,
        "model": "v3_rnnt",
        "segments": [
            {"index": number, "start": start, "end": end, "text": text, "words": []}
            for number, (start, end, text) in enumerate(segments)
        ],
        "metrics": {"words": 0},
    }


_OVERVIEW_DEFAULT = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."


def _answer(**bodies: str) -> str:
    """Собирает ответ claude из восьми разделов в правильном порядке.

    Чего не передали — у списочного раздела «не прозвучало», у «Изложения» три
    коротких абзаца. Так фикстура в каждой проверке занимает ровно те разделы, о
    которых проверка и говорит, но форма ответа остаётся правильной: иначе
    сработал бы откат «восемь разделов не найдены» и проверялось бы не то.
    """
    parts: list[str] = []
    for section in SECTIONS:
        if section.key == OVERVIEW:
            body = bodies.get(OVERVIEW, _OVERVIEW_DEFAULT)
        else:
            body = bodies.get(section.key, NOTHING)
        parts.append("%s\n\n%s" % (section.heading, body))
    return "\n\n".join(parts) + "\n"


def _good_answer() -> str:
    """Ответ claude, какого требует промпт: восемь разделов и таймкоды из строк.

    Срок первого домашнего задания назван и повторён строкой в «Сроках и
    дедлайнах» — как того требует промпт; у второго задания срока в записи нет,
    и claude про него ничего не написал.
    """
    return _answer(
        overview=(
            "Созвон был посвящён формату итогового документа: сначала разобрали, как\n"
            "выглядит расшифровка, потом перешли к саммари.\n"
            "\n"
            "Отдельно обсудили таймкоды и подтвердили прежнее решение — ставить их\n"
            "на каждое предложение.\n"
            "\n"
            "В конце вернулись к нарезке текста на окна и разошлись по задачам."
        ),
        key_points=(
            "- [00:00:00] Итоговый документ состоит из саммари и расшифровки\n"
            "- [00:10:00] Таймкод нужен у каждого предложения\n"
            "- [01:02:03] Текст уходит в обработку окнами"
        ),
        topics=(
            "- [00:00:00] Разбор формата итогового документа\n"
            "- [00:10:00] Таймкод у каждого предложения\n"
            "- [01:02:03] Нарезка текста на окна"
        ),
        homework=(
            "- [01:02:03] Прочитать главу про нарезку текста — срок: к следующему занятию\n"
            "- [02:35:29] Прислать примеры своих записей"
        ),
        deadlines=(
            "- [01:02:03] Прочитать главу про нарезку текста — срок: к следующему занятию\n"
            "- [02:35:29] Сдать итоговый документ — срок: до 15 октября"
        ),
        actions=(
            "- [00:10:00] Таймкод ставится на каждое предложение\n"
            "- [02:35:29] Задачи разошлись по исполнителям"
        ),
        terms=(
            "- [00:10:00] Таймкод — время от начала записи в виде ЧЧ:ММ:СС\n"
            "- [01:02:03] Окно — кусок текста, уходящий в один вызов"
        ),
        materials="- [01:02:03] Глава про нарезку текста в документации проекта",
    )


def parse_fragment(text: str) -> dict:
    """Обратный разбор собственного фрагмента. Отступление от формата — AssertionError.

    Это и проверка структуры, и демонстрация того, что фрагмент читается без
    markdown-парсера: сборщику документа хватает split("\\n").
    """
    _check(text, "фрагмент пуст")
    _check(text.endswith("\n"), "фрагмент не заканчивается переводом строки")
    _check("\r" not in text, "в фрагменте есть возврат каретки")
    _check("\n\n\n" not in text, "в фрагменте больше одной пустой строки подряд")

    lines = text.rstrip("\n").split("\n")
    headings = [line for line in lines if HEADING_RE.match(line.strip()) is not None]
    for heading in headings:
        level = len(HEADING_RE.match(heading.strip()).group(1))
        _check(level >= 3, "в фрагменте есть заголовок уровня %d: %r" % (level, heading))
    found = [line for line in lines if line.startswith("### ")]
    _check(
        found == list(EXPECTED_HEADINGS),
        "не те восемь разделов или не в том порядке: %r" % found,
    )

    bounds = [lines.index(section.heading) for section in SECTIONS] + [len(lines)]
    fragment: dict[str, list[str]] = {}
    for number, section in enumerate(SECTIONS):
        body = lines[bounds[number] + 1:bounds[number + 1]]
        fragment[section.key] = [line for line in body if line]
    return fragment


def _test_section_order() -> None:
    """Состав, порядок и формулировки восьми разделов и строк-пометок."""
    _check(
        tuple(section.heading for section in SECTIONS) == EXPECTED_HEADINGS,
        "не тот состав или порядок разделов: %r"
        % [section.heading for section in SECTIONS],
    )
    _check(len(LIST_KEYS) == 7, "списочных разделов %d, а должно быть семь" % len(LIST_KEYS))
    _check(SECTIONS[0].key == OVERVIEW, "первым разделом идёт не «Изложение»")
    # Формулировки пометок: по ним фрагмент читает человек, а не только код.
    _check(NOTHING == "- не прозвучало", "не та строка пустого раздела: %r" % NOTHING)
    _check(DEADLINE_UNKNOWN == "не назван", "не та пометка срока: %r" % DEADLINE_UNKNOWN)
    _check(NOT_PARSED.startswith("- не выделено:"), "не та пометка разбора: %r" % NOT_PARSED)

    # Пустое саммари — и то даёт ровно восемь заголовков в этом порядке.
    empty = parse_fragment(render_fragment(Summary()))
    _check(sorted(empty) == sorted([OVERVIEW] + list(LIST_KEYS)), "не те разделы: %r" % empty)
    for key in LIST_KEYS:
        _check(empty[key] == [NOTHING], "раздел %s пустого саммари: %r" % (key, empty[key]))


def _test_timecodes() -> None:
    """Формат таймкода во фрагменте и разбор таймкодов из ответа claude."""
    _check(format_timecode(9329.4) == "02:35:29", "02:35:29 не отрендерился")
    _check(format_timecode(0.0) == "00:00:00", "ноль не отрендерился")
    _check(format_timecode(3723.9) == "01:02:03", "01:02:03 не отрендерился")
    _check(format_timecode(360_000.0) == "100:00:00", "сутки с лишним обрезались")

    cases = [
        ("- [00:10:00] Тема", 600.0, "Тема"),
        ("- 00:10:00 Тема", 600.0, "Тема"),
        ("* [02:35:29] — Итоги", 9329.0, "Итоги"),
        ("1. [01:02:03]: Нарезка", 3723.0, "Нарезка"),
        ("- [10:00] Тема без часов", 600.0, "Тема без часов"),
        ("- Тема в [00:00:30] посередине", 30.0, "Тема в посередине"),
    ]
    for line, seconds, body in cases:
        found = parse_timecode(line)
        _check(found is not None, "таймкод не найден в %r" % line)
        _check(found[0] == seconds, "%r -> %s, ожидалось %s" % (line, found[0], seconds))
        _check(found[1] == body, "%r -> текст %r, ожидалось %r" % (line, found[1], body))
    for line in ("- Тема без таймкода", "Основные темы:", "- не прозвучало"):
        _check(parse_timecode(line) is None, "в %r нашёлся таймкод" % line)


def _test_sections() -> None:
    """Заголовки разделов узнаются в разной разметке, лишние — не узнаются."""
    # Все восемь заголовков фрагмента узнаются в своём же написании.
    for section in SECTIONS:
        _check(
            section_of(section.heading) == section.key,
            "заголовок %r не узнан: %r" % (section.heading, section_of(section.heading)),
        )
    for line, expected in (
        ("### Изложение", "overview"),
        ("## Темы", "topics"),
        ("# Решения, задачи, договорённости", "actions"),
        ("**Темы**", "topics"),
        ("Решения, задачи, договоренности", "actions"),
        ("###  темы  ", "topics"),
        ("**Ключевые моменты**", "key_points"),
        ("## Домашние задания", "homework"),
        ("Сроки и дедлайны", "deadlines"),
        ("#### Термины и определения", "terms"),
        ("### материалы и ссылки", "materials"),
    ):
        _check(section_of(line) == expected, "%r -> %r" % (line, section_of(line)))
    for line in ("### Выводы", "Темы и решения обсудили позже", "#темы", "", "### Сроки"):
        _check(section_of(line) is None, "%r сошло за заголовок раздела" % line)

    _check(demote_heading("# Саммари") == "#### Саммари", "заголовок # не опущен")
    _check(demote_heading("## Темы") == "#### Темы", "заголовок ## не опущен")
    _check(demote_heading("### Выводы") == "#### Выводы", "заголовок ### не опущен")
    _check(demote_heading("#### Мелкий") == "#### Мелкий", "заголовок #### изменён")
    _check(demote_heading("##### Ещё") == "##### Ещё", "заголовок ##### изменён")
    _check(demote_heading("просто строка") == "просто строка", "строка без решёток изменена")


def _test_good_answer() -> None:
    """Успешный ответ: восемь разделов, один вызов claude, промпт со строками."""
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        stub = _answering_stub(workdir, "ок", _good_answer())
        # Кириллица и пробелы в пути, каталога ещё нет.
        out = os.path.join(workdir, "итог прогона", "саммари.md")

        code, report, stderr = _run(["--punctuated", source, "--out", out], stub)
        _check(code == 0, "успешный ответ вернул код %d (stderr: %s)" % (code, stderr.strip()))
        _check(os.path.isfile(out), "фрагмент не записан: %s" % out)
        _check(_calls(stub) == 1, "claude позван %d раз, а должен ровно один" % _calls(stub))

        fragment = parse_fragment(_read(out))
        _check(len(fragment["overview"]) == 3, "абзацев %d" % len(fragment["overview"]))
        _check(
            fragment["overview"][0].startswith("Созвон был посвящён"),
            "текст изложения изменился: %r" % fragment["overview"][0],
        )
        # Перенос строк внутри абзаца снят, слова на месте.
        _check(
            "сначала разобрали, как выглядит расшифровка" in fragment["overview"][0],
            "абзац не склеен в одну строку: %r" % fragment["overview"][0],
        )
        # Состав всех семи списочных разделов.
        for key, expected in (
            ("key_points", 3),
            ("topics", 3),
            ("homework", 2),
            ("deadlines", 2),
            ("actions", 2),
            ("terms", 2),
            ("materials", 1),
        ):
            _check(
                len(fragment[key]) == expected,
                "в разделе %s %d строк, ожидалось %d: %r"
                % (key, len(fragment[key]), expected, fragment[key]),
            )
        # Каждая строка каждого списочного раздела начинается с таймкода.
        for key in LIST_KEYS:
            for line in fragment[key]:
                _check(
                    ITEM_RE.match(line) is not None,
                    "раздел %s: строка не того вида: %r" % (key, line),
                )
        _check("- [02:35:29] Задачи разошлись по исполнителям" in fragment["actions"],
               "таймкод 02:35:29 потерялся: %r" % fragment["actions"])

        # Домашние задания: срок стоит в строке задания, названный срок повторён
        # строкой в «Сроках и дедлайнах», неназванный — это «срок: не назван».
        for line in fragment["homework"]:
            _check(HOMEWORK_LINE_RE.match(line) is not None, "строка ДЗ не того вида: %r" % line)
        _check(
            fragment["homework"][0]
            == "- [01:02:03] Прочитать главу про нарезку текста — срок: к следующему занятию",
            "срок потерялся из строки ДЗ: %r" % fragment["homework"][0],
        )
        _check(
            fragment["homework"][1] == "- [02:35:29] Прислать примеры своих записей — срок: %s"
            % DEADLINE_UNKNOWN,
            "ДЗ без срока записано не так: %r" % fragment["homework"][1],
        )
        _check(
            any("к следующему занятию" in line for line in fragment["deadlines"]),
            "срок ДЗ не повторён в «Сроках и дедлайнах»: %r" % fragment["deadlines"],
        )
        # Термин и материал — в выбранном формате, определение в той же строке.
        _check(
            fragment["terms"][0].endswith("Таймкод — время от начала записи в виде ЧЧ:ММ:СС"),
            "определение термина изменилось: %r" % fragment["terms"][0],
        )

        # Промпт: строки «[ЧЧ:ММ:СС] предложение» и требование про таймкоды.
        prompt = _prompt_of(stub)
        for needle in (
            "[00:00:00] Добрый день, начинаем разбор формата итогового документа.",
            "[00:10:00] Таймкод у каждого предложения, это решение уже принято.",
            "[02:35:29] На этом закончили, задачи разошлись по исполнителям.",
        ):
            _check(needle in prompt, "в промпте нет строки %r" % needle)
        for needle in (
            "ТОЛЬКО из начала строк расшифровки",
            "Своих таймкодов не выдумывай",
            "Отвечай по-русски",
            "ровно из восьми разделов",
            "- [ЧЧ:ММ:СС] задание — срок: как сказано в записи",
            "срок: %s" % DEADLINE_UNKNOWN,
            "Срок цитируй теми же словами",
            "к следующему занятию",
            "через две недели",
            "до 15 октября",
            "сроки в даты запрещено",
            "повтори здесь отдельной строкой",
            "- [ЧЧ:ММ:СС] термин — определение, как прозвучало",
            "- [ЧЧ:ММ:СС] название или ссылка, как прозвучала",
            "- не прозвучало",
        ):
            _check(needle in prompt, "в промпте нет требования %r" % needle)
        # Все восемь заголовков — образцом, по одному разу каждый.
        for section in SECTIONS:
            seen = prompt.count(section.heading)
            _check(seen == 1, "в промпте %d образцов раздела %r" % (seen, section.heading))

        # Отчёт в stdout: пунктов по каждому из семи списочных разделов.
        _check("Файл:" in report, "в отчёте нет строки «Файл:»")
        for label, value in (
            ("Абзацев", 3),
            ("Ключевых моментов", 3),
            ("Тем", 3),
            ("Домашних заданий", 2),
            ("Сроков", 2),
            ("Решений", 2),
            ("Терминов", 2),
            ("Материалов", 1),
        ):
            row = report_row(label, "%d" % value)
            _check(row in report, "в отчёте нет строки %r:\n%s" % (row.rstrip(), report))
        _check(not stderr.strip(), "при успешном ответе есть stderr: %s" % stderr.strip())

        # Кроме --out на диске ничего нового: текст лекции мимо флагов не пишется.
        _check(
            sorted(os.listdir(workdir)) == ["punctuated.json", "итог прогона", "ок"],
            "в рабочем каталоге появилось лишнее: %r" % sorted(os.listdir(workdir)),
        )


def _test_untidy_answer() -> None:
    """Ответ с теми же разделами, но в другой разметке: формат всё равно чинится."""
    answer = (
        "Вот саммари записи.\n"
        "\n"
        "## Изложение\n"
        "Запись про формат документа.\n"
        "\n"
        "Во второй части обсудили таймкоды.\n"
        "\n"
        "**Ключевые моменты**\n"
        "- [00:00:00] формат документа решён\n"
        "\n"
        "**Темы**\n"
        "Основные темы:\n"
        "* 00:00 — Формат документа\n"
        "* [00:10:00]: таймкоды\n"
        "- тема без таймкода\n"
        "\n"
        "Домашние задания\n"
        "1. 01:02:03 прочитать главу, срок — к следующему занятию\n"
        "2. [02:35:29] прислать примеры\n"
        "\n"
        "### Сроки и дедлайны\n"
        "- [01:02:03] прочитать главу — срок: к следующему занятию\n"
        "\n"
        "# Решения, задачи, договоренности\n"
        "1. [02:35:29] задачи разошлись\n"
        "\n"
        "#### Термины и определения\n"
        "- [00:10:00] таймкод — время от начала записи\n"
        "\n"
        "**Материалы и ссылки**\n"
        "нет\n"
    )
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "p.json"), _fixture_punctuated())
        stub = _answering_stub(workdir, "ок", answer)
        out = os.path.join(workdir, "саммари.md")

        code, _, stderr = _run(["--punctuated", source, "--out", out], stub)
        _check(code == 0, "неряшливый ответ вернул код %d" % code)

        fragment = parse_fragment(_read(out))
        _check(len(fragment["overview"]) == 2, "абзацев %d" % len(fragment["overview"]))
        _check(fragment["topics"] == ["- [00:00:00] Формат документа", "- [00:10:00] таймкоды"],
               "темы не привелись к виду: %r" % fragment["topics"])
        _check(fragment["actions"] == ["- [02:35:29] задачи разошлись"],
               "решения не привелись к виду: %r" % fragment["actions"])
        _check(fragment["key_points"] == ["- [00:00:00] формат документа решён"],
               "ключевые моменты не привелись к виду: %r" % fragment["key_points"])
        _check(fragment["terms"] == ["- [00:10:00] таймкод — время от начала записи"],
               "термины не привелись к виду: %r" % fragment["terms"])
        # «нет» вместо списка — это «перечислять нечего», а не пункт списка.
        _check(fragment["materials"] == [NOTHING],
               "материалы не помечены: %r" % fragment["materials"])
        # ДЗ: срок из строки «срок — к следующему занятию» перенесён в свою
        # позицию, у задания без срока там стоит «срок: не назван».
        _check(
            fragment["homework"] == [
                "- [01:02:03] прочитать главу — срок: к следующему занятию",
                "- [02:35:29] прислать примеры — срок: %s" % DEADLINE_UNKNOWN,
            ],
            "домашние задания не привелись к виду: %r" % fragment["homework"],
        )
        # Про отброшенные строки и короткое изложение сказано в stderr.
        _check("до первого раздела" in stderr, "про преамбулу не предупредили: %s" % stderr)
        _check("без таймкода" in stderr, "про строки без таймкода не предупредили: %s" % stderr)
        _check("Темы:" in stderr, "в предупреждении не назван раздел: %s" % stderr)
        _check("абзацев" in stderr, "про число абзацев не предупредили: %s" % stderr)

        # Таймкод за концом записи — признак выдуманного: строка остаётся (слов
        # claude инструмент не трогает), но о ней сказано в stderr.
        beyond = _answering_stub(
            workdir, "мимо", _answer(topics="- [05:00:00] Тема из будущего")
        )
        out2 = os.path.join(workdir, "мимо.md")
        code, _, stderr = _run(["--punctuated", source, "--out", out2], beyond)
        _check(code == 0, "таймкод за концом записи вернул код %d" % code)
        _check(
            parse_fragment(_read(out2))["topics"] == ["- [05:00:00] Тема из будущего"],
            "строку с таймкодом за концом записи переписали",
        )
        _check(
            "за пределами записи" in stderr and "02:35:40" in stderr,
            "про таймкод за концом записи не предупредили: %s" % stderr,
        )


def _test_no_sections() -> None:
    """Ответ без восьми разделов: сохраняется как есть, предупреждение, код 0."""
    answer = (
        "# Саммари созвона\n"
        "\n"
        "Обсудили формат документа, таймкоды и нарезку текста на окна.\n"
        "\n"
        "## Что дальше\n"
        "\n"
        "Задачи разошлись по исполнителям.\n"
    )
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "p.json"), _fixture_punctuated())
        stub = _answering_stub(workdir, "ок", answer)
        out = os.path.join(workdir, "саммари.md")

        code, report, stderr = _run(["--punctuated", source, "--out", out], stub)
        _check(code == 0, "ответ без разделов вернул код %d, ожидался 0" % code)

        text = _read(out)
        fragment = parse_fragment(text)
        # Текст ответа на месте целиком, но заголовки «#» и «##» опущены до «###».
        for needle in (
            "Обсудили формат документа, таймкоды и нарезку текста на окна.",
            "Задачи разошлись по исполнителям.",
            "#### Саммари созвона",
            "#### Что дальше",
        ):
            _check(needle in text, "в фрагменте нет %r" % needle)
        # Все семь списочных разделов получили прежнюю пометку «не выделено…».
        for key in LIST_KEYS:
            _check(
                fragment[key] == [NOT_PARSED],
                "раздел %s не помечен: %r" % (key, fragment[key]),
            )
        _check("восемь разделов в ответе claude не найдены" in stderr,
               "нет предупреждения о ненайденных разделах: %s" % stderr)
        _check("восемь разделов в ответе не найдены" in report, "в отчёте нет пометки: %s" % report)

        # Часть заголовков — тот же случай: формату верить нельзя. Проверяется на
        # каждом разделе по очереди: пропущен любой из восьми — откат тот же.
        for number, skipped in enumerate(SECTIONS):
            partial = "\n\n".join(
                "%s\n\n%s" % (section.heading, _OVERVIEW_DEFAULT if section.key == OVERVIEW
                               else "- [00:00:00] Пункт")
                for section in SECTIONS
                if section.key != skipped.key
            ) + "\n"
            half = _answering_stub(workdir, "без-%d" % number, partial)
            out2 = os.path.join(workdir, "без-%d.md" % number)
            code, _, stderr = _run(["--punctuated", source, "--out", out2], half)
            _check(code == 0, "ответ без раздела %s вернул код %d" % (skipped.key, code))
            fragment = parse_fragment(_read(out2))
            _check(
                fragment["topics"] == [NOT_PARSED],
                "без раздела %s списки не помечены: %r" % (skipped.key, fragment["topics"]),
            )
            _check(
                "восемь разделов в ответе claude не найдены" in stderr,
                "без раздела %s нет предупреждения: %s" % (skipped.key, stderr),
            )


def _test_nothing_said() -> None:
    """Пустой список в любом из семи разделов — ровно строка «не прозвучало».

    Проверяется на всех семи по очереди и во всех трёх видах, какими claude
    говорит «перечислять нечего»: домашних заданий, сроков, терминов и
    материалов в обычном созвоне и правда не бывает, так что это не редкий
    случай, а повседневный.
    """
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "p.json"), _fixture_punctuated())
        empty_bodies = [
            ("claude сказал словами", NOTHING),
            ("claude сказал без маркера", "не прозвучало"),
            ("claude оставил раздел пустым", ""),
            ("claude ответил «нет»", "нет"),
        ]
        for number, section in enumerate(LIST_SECTIONS):
            for order, (name, body) in enumerate(empty_bodies):
                # Остальные шесть разделов заполнены: пустеет ровно один.
                bodies = {key: "- [00:00:00] Пункт" for key in LIST_KEYS}
                bodies[section.key] = body
                stub = _answering_stub(workdir, "пусто-%d-%d" % (number, order), _answer(**bodies))
                out = os.path.join(workdir, "саммари-%d-%d.md" % (number, order))
                code, report, _ = _run(["--punctuated", source, "--out", out], stub)
                where = "раздел %s, случай %r" % (section.key, name)
                _check(code == 0, "%s: код %d" % (where, code))
                fragment = parse_fragment(_read(out))
                _check(
                    fragment[section.key] == [NOTHING],
                    "%s: в разделе %r, ожидалось [%r]" % (where, fragment[section.key], NOTHING),
                )
                _check(report_row(section.label, "1") in report, "%s: не та цифра в отчёте" % where)
                # Соседние разделы от этого не пострадали.
                for other in LIST_SECTIONS:
                    if other.key != section.key:
                        _check(
                            len(fragment[other.key]) == 1,
                            "%s: пострадал раздел %s: %r" % (where, other.key, fragment[other.key]),
                        )


def _test_not_received() -> None:
    """Саммари не получено: пометка с причиной во всех восьми разделах и код 1."""
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "p.json"), _fixture_punctuated())
        cases = [
            ("пустой ответ", _stub(workdir, "пусто", "exit 0\n"),
             ["--attempts", "1"], "пустой ответ"),
            ("ненулевой код", _stub(workdir, "упал", "printf 'сломалось\\n' >&2\nexit 3\n"),
             ["--attempts", "1"], "код 3"),
            ("не авторизован", _stub(workdir, "логин", "printf 'Invalid API key\\n' >&2\nexit 1\n"),
             ["--attempts", "1"], "не авторизован"),
            ("нет бинарника", os.path.join(workdir, "нет такого", "claude"), [], "файл не найден"),
            ("таймаут", _stub(workdir, "спит", "sleep 30\n"),
             ["--attempts", "1", "--timeout", "0.4"], "таймаут"),
        ]
        for name, binary, extra, needle in cases:
            out = os.path.join(workdir, "саммари-%s.md" % name.replace(" ", "-"))
            code, report, stderr = _run(
                ["--punctuated", source, "--out", out] + extra, binary
            )
            _check(code == 1, "случай %r вернул код %d, ожидался 1" % (name, code))
            _check(os.path.isfile(out), "случай %r: фрагмент не записан" % name)

            fragment = parse_fragment(_read(out))
            note = fragment["overview"][0]
            _check(
                note.startswith("Саммари не получено:") and note.endswith("."),
                "случай %r: не та пометка в изложении: %r" % (name, note),
            )
            _check(needle in note, "случай %r: в пометке нет причины %r: %r" % (name, needle, note))
            _check(len(fragment["overview"]) == 1, "случай %r: в изложении лишнее" % name)
            for key in LIST_KEYS:
                _check(
                    fragment[key] == ["- %s" % note],
                    "случай %r: раздел %s без пометки: %r" % (name, key, fragment[key]),
                )
            # Причина названа и в stderr, и в отчёте.
            _check(note in stderr, "случай %r: причины нет в stderr: %s" % (name, stderr))
            _check("не получено" in report, "случай %r: в отчёте нет пометки" % name)

        # Слишком длинный промпт — тот же исход, причём claude не вызывается.
        stub = _answering_stub(workdir, "длинный", _good_answer())
        saved = claude_cli.MAX_PROMPT_BYTES
        claude_cli.MAX_PROMPT_BYTES = 100
        try:
            out = os.path.join(workdir, "длинный.md")
            code, _, stderr = _run(["--punctuated", source, "--out", out], stub)
        finally:
            claude_cli.MAX_PROMPT_BYTES = saved
        _check(code == 1, "длинный промпт вернул код %d" % code)
        _check(_calls(stub) == 0, "длинный промпт всё-таки ушёл в claude")
        note = parse_fragment(_read(out))["overview"][0]
        _check("слишком длинный" in note, "в пометке нет причины про длину: %r" % note)


def _test_homework() -> None:
    """Строка ДЗ: срок на своём месте, «не назван» — когда его не было."""
    cases = [
        # Срок назван и отделён по-разному — текст срока не переписывается.
        ("Прочитать главу — срок: к следующему занятию",
         "Прочитать главу — срок: к следующему занятию"),
        ("Прочитать главу, срок — к следующему занятию",
         "Прочитать главу — срок: к следующему занятию"),
        ("Сделать задание дедлайн пятница", "Сделать задание — срок: пятница"),
        ("Сдать работу — срок сдачи: до 15 октября", "Сдать работу — срок: до 15 октября"),
        ("Сдать работу — срок: к следующему занятию (до 15 октября)",
         "Сдать работу — срок: к следующему занятию (до 15 октября)"),
        # Срок не назван вовсе или назван словами «не указан».
        ("Прислать примеры", "Прислать примеры — срок: %s" % DEADLINE_UNKNOWN),
        ("Прислать примеры, срок не указан", "Прислать примеры — срок: %s" % DEADLINE_UNKNOWN),
        ("Прислать примеры — срок:", "Прислать примеры — срок: %s" % DEADLINE_UNKNOWN),
        ("Прислать примеры — срок: не назван", "Прислать примеры — срок: %s" % DEADLINE_UNKNOWN),
        # Слово «срок» внутри самого задания за метку не сходит.
        ("Разобраться со сроками проектов",
         "Разобраться со сроками проектов — срок: %s" % DEADLINE_UNKNOWN),
        ("Прочитать главу про срок сдачи — срок: к пятнице",
         "Прочитать главу про срок сдачи — срок: к пятнице"),
    ]
    for body, expected in cases:
        got = homework_body(body)
        _check(got == expected, "%r -> %r, ожидалось %r" % (body, got, expected))
    # Срок без задания: вписывать в список нечего, строка выбрасывается.
    for body in ("срок: к следующему занятию", "— срок: пятница", "срок"):
        _check(homework_body(body) is None, "%r сошло за задание: %r" % (body, homework_body(body)))

    # Названный срок, не повторённый в «Сроках и дедлайнах», — предупреждение.
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "p.json"), _fixture_punctuated())
        stub = _answering_stub(
            workdir,
            "без-сроков",
            _answer(
                homework=(
                    "- [00:10:00] Прочитать главу — срок: к следующему занятию\n"
                    "- [00:10:00] Прислать примеры"
                ),
                deadlines=NOTHING,
            ),
        )
        out = os.path.join(workdir, "саммари.md")
        code, _, stderr = _run(["--punctuated", source, "--out", out], stub)
        _check(code == 0, "неповторённый срок вернул код %d" % code)
        _check(
            "к следующему занятию" in stderr and "не повторён строкой" in stderr,
            "про неповторённый срок не предупредили: %s" % stderr,
        )
        # Неназванный срок в это предупреждение не попадает.
        _check(
            DEADLINE_UNKNOWN not in stderr,
            "неназванный срок попал в предупреждение: %s" % stderr,
        )


def _test_transcript_mode() -> None:
    """Режим --transcript и приоритет --punctuated при обоих флагах."""
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        transcript = _write_json(os.path.join(workdir, "transcript.json"), _fixture_transcript())
        punctuated = _write_json(os.path.join(workdir, "punctuated.json"), _fixture_punctuated())
        stub = _answering_stub(workdir, "ок", _good_answer())
        out = os.path.join(workdir, "саммари.md")

        code, report, stderr = _run(["--transcript", transcript, "--out", out], stub)
        _check(code == 0, "режим транскрипта вернул код %d" % code)
        prompt = _prompt_of(stub)
        _check(
            "[00:00:00] добрый день начинаем разбор формата итогового документа" in prompt,
            "сегмент транскрипта не попал в промпт",
        )
        _check("[02:35:29] на этом закончили" in prompt, "последний сегмент не в промпте")
        _check(TRANSCRIPT_SCHEMA in report, "в отчёте не названа схема входа")
        _check("без пунктуации" in stderr, "про дословный текст не предупредили: %s" % stderr)

        # Оба флага: берётся punctuated, в промпте текст с пунктуацией.
        out2 = os.path.join(workdir, "оба.md")
        code, _, stderr = _run(
            ["--transcript", transcript, "--punctuated", punctuated, "--out", out2], stub
        )
        _check(code == 0, "прогон с обоими флагами вернул код %d" % code)
        _check("взят --punctuated" in stderr, "про приоритет не сказано: %s" % stderr)
        prompt = _prompt_of(stub)
        _check(
            "[00:00:00] Добрый день, начинаем разбор формата итогового документа." in prompt,
            "при обоих флагах в промпт ушёл транскрипт",
        )


def _test_broken_input() -> None:
    """Непригодный вход: код 2, понятное сообщение, --out не создаётся."""
    bad = [
        ("не объект", []),
        ("нет schema", {"sentences": []}),
        ("чужая схема", {"schema": "punctuated/2", "sentences": []}),
        ("схема транскрипта", {"schema": TRANSCRIPT_SCHEMA, "sentences": []}),
        ("нет sentences", {"schema": PUNCTUATED_SCHEMA, "paragraphs": []}),
        ("sentences не список", {"schema": PUNCTUATED_SCHEMA, "sentences": {}}),
        ("sentences пуст", {"schema": PUNCTUATED_SCHEMA, "sentences": []}),
        (
            "предложение не объект",
            {"schema": PUNCTUATED_SCHEMA, "sentences": ["текст"]},
        ),
        (
            "нет start",
            {"schema": PUNCTUATED_SCHEMA, "sentences": [{"index": 0, "text": "а"}]},
        ),
        (
            "start не число",
            {"schema": PUNCTUATED_SCHEMA, "sentences": [{"start": "ноль", "text": "а"}]},
        ),
        (
            "start меньше нуля",
            {"schema": PUNCTUATED_SCHEMA, "sentences": [{"start": -1.0, "text": "а"}]},
        ),
        ("нет text", {"schema": PUNCTUATED_SCHEMA, "sentences": [{"start": 0.0}]}),
        (
            "все предложения пустые",
            {"schema": PUNCTUATED_SCHEMA, "sentences": [{"start": 0.0, "text": "  "}]},
        ),
    ]
    for name, payload in bad:
        try:
            with redirect_stderr(io.StringIO()):
                parse_punctuated(payload, "<память>")
        except InputError as exc:
            _check(str(exc).strip() != "", "случай %r: пустое сообщение об ошибке" % name)
        else:
            raise AssertionError("случай %r прошёл проверку структуры" % name)

    # Транскрипт проверяется той же меркой.
    for name, payload in (
        ("не та схема", {"schema": PUNCTUATED_SCHEMA, "segments": []}),
        ("нет segments", {"schema": TRANSCRIPT_SCHEMA}),
        ("segments пуст", {"schema": TRANSCRIPT_SCHEMA, "segments": []}),
    ):
        try:
            parse_transcript(payload, "<память>")
        except InputError:
            pass
        else:
            raise AssertionError("транскрипт, случай %r, прошёл проверку" % name)

    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        stub = _answering_stub(workdir, "ок", _good_answer())
        good = _write_json(os.path.join(workdir, "good.json"), _fixture_punctuated())
        empty = _write_json(
            os.path.join(workdir, "empty.json"),
            dict(_fixture_punctuated(), sentences=[], paragraphs=[]),
        )
        broken = os.path.join(workdir, "broken.json")
        write_text_file(broken, '{"schema": "punctuated/1", "sentences": [')
        out = os.path.join(workdir, "не должен появиться", "саммари.md")

        for argv, what in (
            (["--punctuated", empty, "--out", out], "пустой список предложений"),
            (["--punctuated", broken, "--out", out], "битый JSON"),
            (["--punctuated", os.path.join(workdir, "нет.json"), "--out", out], "нет файла"),
            (["--punctuated", workdir, "--out", out], "каталог вместо файла"),
            (["--transcript", good, "--out", out], "punctuated под видом транскрипта"),
            (["--out", out], "без входа"),
            (["--punctuated", good], "без --out"),
            (["--punctuated", good, "--out", good], "--out поверх входа"),
            (["--punctuated", good, "--out", out, "--timeout", "0"], "нулевой таймаут"),
            (["--punctuated", good, "--out", out, "--timeout", "много"], "нечисловой таймаут"),
            (["--punctuated", good, "--out", out, "--attempts", "0"], "ноль попыток"),
            (["--punctuated", good, "--out", out, "--attempts", "два"], "нечисловые попытки"),
            (["--punctuated", good, "--out", ""], "пустое значение --out"),
            (["--punctuated", good, "--out"], "--out без значения"),
            (["--кривая-опция"], "неизвестная опция"),
            (["лишнее"], "лишний аргумент"),
            (["--selftest=1"], "--selftest со значением"),
        ):
            code, _, stderr = _run(argv, stub)
            _check(code == 2, "случай %r завершился кодом %d, ожидался 2" % (what, code))
            _check(stderr.startswith(PROG + ":"), "случай %r ничего не сказал в stderr" % what)
            _check(not os.path.exists(out), "случай %r оставил файл --out на диске" % what)
        _check(_calls(stub) == 0, "при непригодном входе claude всё-таки вызывался")

        # Существующий фрагмент при непригодном входе не перетирается.
        kept = os.path.join(workdir, "kept.md")
        write_text_file(kept, "старое содержимое\n")
        code, _, _ = _run(["--punctuated", broken, "--out", kept], stub)
        _check(code == 2, "перетирание: код %d вместо 2" % code)
        _check(_read(kept) == "старое содержимое\n", "старый фрагмент перетёрт при ошибке входа")


def _test_log_dir() -> None:
    """--log-dir: промпт и ответ сохраняются через tools/claude_cli.py."""
    with tempfile.TemporaryDirectory(prefix="summarize-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "p.json"), _fixture_punctuated())
        stub = _answering_stub(workdir, "ок", _good_answer())
        out = os.path.join(workdir, "саммари.md")
        log_dir = os.path.join(workdir, "логи claude")

        code, _, _ = _run(
            ["--punctuated", source, "--out", out, "--log-dir", log_dir], stub
        )
        _check(code == 0, "прогон с --log-dir вернул код %d" % code)
        names = sorted(os.listdir(log_dir))
        _check(
            names == ["%s-001.prompt.txt" % LOG_TAG, "%s-001.response.txt" % LOG_TAG],
            "в каталоге логов не те файлы: %r" % names,
        )
        prompt_log = _read(os.path.join(log_dir, "%s-001.prompt.txt" % LOG_TAG))
        _check("[02:35:29] На этом закончили" in prompt_log, "в логе промпта нет строк записи")
        response_log = _read(os.path.join(log_dir, "%s-001.response.txt" % LOG_TAG))
        _check(SECTION_TOPICS in response_log, "в логе ответа нет ответа claude")

        # Неудачный вызов тоже оставляет оба файла — иначе непонятно, что ушло.
        broken = _stub(workdir, "упал", "exit 5\n")
        code, _, _ = _run(
            ["--punctuated", source, "--out", out, "--log-dir", log_dir, "--attempts", "1"],
            broken,
        )
        _check(code == 1, "неудачный вызов с --log-dir вернул код %d" % code)
        failure = _read(os.path.join(log_dir, "%s-002.response.txt" % LOG_TAG))
        _check(failure.startswith("НЕ ПОЛУЧЕНО"), "в логе неудачи нет пометки: %r" % failure)


def _test_help() -> None:
    """Справка — на русском, со всеми флагами, код 0."""
    for flag in ("-h", "--help"):
        code, out, _ = _run([flag])
        _check(code == 0, "%s вернул код %d" % (flag, code))
        for needle in (
            "--punctuated",
            "--transcript",
            "--out",
            "--timeout",
            "--attempts",
            "--log-dir",
            "--selftest",
            "-h, --help",
            "Использование",
            "Коды возврата",
            # Новый состав разделов и формат строк ДЗ и сроков.
            "восемью разделами",
            "задание — срок:",
            "срок: %s" % DEADLINE_UNKNOWN,
            "повторяется здесь строкой",
            "как прозвучал в записи",
            "Пересчёт относительных сроков в даты промпт запрещает",
            "не прозвучало",
        ):
            _check(needle in out, "в справке (%s) нет %r" % (flag, needle))
        # Все восемь разделов названы в справке.
        for section in SECTIONS:
            _check(
                "«%s»" % section.name in out,
                "в справке (%s) не назван раздел «%s»" % (flag, section.name),
            )
        _check(re.search(r"[а-яА-Я]", out) is not None, "справка не на русском")


def _test_only_stdlib() -> None:
    """Импорты — только стандартная библиотека и соседние tools/*.py."""
    allowed = {
        "__future__",
        "io",
        "json",
        "math",
        "os",
        "re",
        "sys",
        "tempfile",
        "contextlib",
        "dataclasses",
        "claude_cli",
        "render_doc",
    }
    with open(os.path.abspath(__file__), "r", encoding="utf-8") as handle:
        source = handle.read()
    imported = set(re.findall(r"^(?:import|from)\s+([A-Za-z_][\w]*)", source, re.MULTILINE))
    _check(imported, "импортов не нашлось — проверка сломана")
    _check(imported <= allowed, "лишние импорты: %r" % sorted(imported - allowed))
    # Свой subprocess здесь не пишется: claude запускается только через обёртку.
    _check("subprocess" not in imported, "появился свой вызов subprocess")


def selftest() -> int:
    """Прогоняет встроенные проверки. Код возврата 0 — все прошли."""
    tests = (
        ("восемь разделов: состав и порядок", _test_section_order),
        ("формат и разбор таймкодов", _test_timecodes),
        ("распознавание заголовков разделов", _test_sections),
        ("успешный ответ с восемью разделами", _test_good_answer),
        ("ответ в другой разметке приводится к формату", _test_untidy_answer),
        ("ответ без восьми разделов сохраняется как есть", _test_no_sections),
        ("пустой список в любом разделе — «не прозвучало»", _test_nothing_said),
        ("домашние задания: срок в строке и в разделе сроков", _test_homework),
        ("саммари не получено: пометка и код 1", _test_not_received),
        ("режим --transcript и приоритет --punctuated", _test_transcript_mode),
        ("непригодный вход и коды возврата", _test_broken_input),
        ("--log-dir: промпт и ответ на диске", _test_log_dir),
        ("справка", _test_help),
        ("импорты только из стандартной библиотеки", _test_only_stdlib),
    )
    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ summarize.py ================\n")
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
