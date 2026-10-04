#!/usr/bin/env python3
"""roles.py — claude превращает «Говорящий N» в роли «Преподаватель» и «Студент N».

На вход — файл схемы speakers/1 (его пишет tools/speakers.py: фразы с именами
говорящих от диаризации). Диаризация делит речь по голосам и больше ничего о
людях не знает: имена у неё обезличенные — «Говорящий 1», «Говорящий 2». Кто из
них ведёт занятие, а кто учится, видно только по смыслу разговора, и это
определяет claude одним вызовом по показательным фразам каждого говорящего.

На выход — JSON схемы roles/1: соответствие «Говорящий N → роль», где роли
называются «Преподаватель» (он один) и «Студент 1», «Студент 2», … Этот файл
читают tools/render_doc.py и tools/summarize.py, чтобы поставить ярлык в строку
расшифровки, поэтому состав полей здесь фиксирован и менять его нельзя.

Главное свойство: файл пишется всегда. Если claude недоступен, не авторизован,
не ответил или ответил негодно, в файле остаётся то же соответствие, но роли в
нём равны именам говорящих, стоит status «fallback» с причиной, а код возврата
становится 1. Расшифровка стоит двух часов распознавания, и терять ярлыки совсем
из-за одного неудачного вызова нельзя: «Говорящий 1» в документе хуже, чем
«Преподаватель», но лучше, чем строка без ярлыка. Код 2 оставлен для непригодного
входа: там --out не создаётся и существующий файл не перетирается.

Два случая решаются без claude. Говорящий всего один — он и есть ведущий
занятие, звать claude не за чем (status «ok», source «default», код 0).
Говорящих нет вовсе — соответствие пустое, status «skip», код 0: диаризация не
дала ничего, но это не ошибка этого инструмента.

Почему ответ claude не принимается дословно. Он может назвать говорящего,
которого в записи нет, пропустить говорящего, дописать пояснение в роль или
назвать «Преподавателем» сразу двоих — а документ собирается по этому файлу, и
две роли «Преподаватель» в нём означают, что ярлыки врут. Поэтому роль чистится
до одной строки не длиннее 40 символов, лишние строки ответа игнорируются,
неназванный говорящий получает «Студент N», а при споре за «Преподавателя»
роль остаётся у того, кто больше говорил: ведущий занятие говорит больше всех
не по определению, но почти всегда, а выбрать кого-то всё равно нужно.

Что выбрано здесь из оставленного на усмотрение. Показательные фразы — это 15
самых длинных фраз говорящего (в них видно, объясняет человек или спрашивает),
поданные в порядке времени; длинная фраза обрезается, чтобы промпт двухчасовой
записи влезал в предел длины аргумента командной строки. Роли, которых нет в
промпте, не отбрасываются, а чистятся и записываются как есть: ярлык с чужим
словом читателю понятен, а потерянный ярлык — нет.

Зависимости: только стандартная библиотека. Сам claude запускается через
tools/claude_cli.py — это единственное место в проекте, где собирается его
командная строка; своего subprocess здесь нет. Формат таймкода повторяет
tools/render_doc.py (импортировать его здесь нельзя — список зависимостей этого
инструмента ограничен claude_cli), чтобы «[01:02:03]» в промпте и в документе
означало одно и то же.

Проверки живут в самом файле (--selftest): в окружении проекта нет pytest.
Установленный claude для них не нужен и намеренно не используется — все вызовы
идут на подставных sh-скриптах.

Использование:
  python tools/roles.py --speakers <speakers.json> --out <roles.json>
  python tools/roles.py --selftest
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

PROG = "roles.py"

SPEAKERS_SCHEMA = "speakers/1"
ROLES_SCHEMA = "roles/1"

# Названия ролей. Других в промпте нет — это решение владельца, и расширять
# список без нового решения нельзя.
TEACHER = "Преподаватель"
STUDENT = "Студент %d"

# Статусы файла roles/1.
STATUS_OK = "ok"
STATUS_FALLBACK = "fallback"
STATUS_SKIP = "skip"

# Откуда взялась роль: claude назвал её, инструмент поставил «Студент N» вместо
# неназванной или роли нет вовсе и вместо неё стоит имя говорящего.
SOURCE_CLAUDE = "claude"
SOURCE_DEFAULT = "default"
SOURCE_FALLBACK = "fallback"

# Предел длины роли из критериев приёмки: ярлык стоит в начале каждой строки
# расшифровки, и длинный превращает документ в кашу.
MAX_ROLE_CHARS = 40

# Сколько показательных фраз на говорящего уходит в промпт и насколько длинной
# может быть одна: промпт передаётся аргументом командной строки, и у него есть
# предел (см. claude_cli.MAX_PROMPT_BYTES).
MAX_PHRASES = 15
MAX_PHRASE_CHARS = 300

# Префикс имён файлов лога вызова (tools/claude_cli.py кладёт <tag>-NNN.*.txt).
LOG_TAG = "роли"

# Причины, которые попадают в поле reason, когда claude не звали.
REASON_SINGLE = (
    "говорящий один: он и ведёт занятие, роль «%s» присвоена без вызова claude" % TEACHER
)
REASON_NO_SPEAKERS = "в speakers/1 нет ни одного говорящего — соответствие пустое"

HELP = """Использование:
  python tools/roles.py --speakers <speakers.json> --out <roles.json> [опции]
  python tools/roles.py --selftest

Определяет одним вызовом claude, кто из говорящих кто по смыслу разговора, и
пишет JSON схемы roles/1 — соответствие «Говорящий N → роль». Роли называются
«{teacher}» (он один) и «Студент 1», «Студент 2», … с нумерацией по убыванию
доли речи.

В промпт по каждому говорящему уходят имя, доля речи, число фраз и до {phrases}
показательных фраз с таймкодами; ответ требуется строками строго вида
«Говорящий N -> Роль», без пояснений. Ответ разбирается терпимо: лишние строки
игнорируются, неизвестные имена говорящих дают предупреждение в stderr, роль
приводится к одной строке не длиннее {chars} символов. Говорящий, которого
claude не назвал, получает «Студент N»; если роль «{teacher}» досталась
нескольким, она остаётся у того, кто больше говорил, а остальные становятся
«Студент N».

Говорящий всего один — роль «{teacher}» присваивается без вызова claude.
Говорящих нет вовсе — файл пишется с пустым соответствием и статусом «{skip}».

Сам claude запускается через tools/claude_cli.py как `claude -p <промпт>`;
модель не задаётся, вызов ровно один на прогон.

Опции:
  --speakers <файл>    JSON схемы speakers/1 от tools/speakers.py (обязателен)
  --out <файл>         куда записать JSON схемы roles/1 (обязателен)
  --timeout <сек>      сколько ждать вызов claude (по умолчанию {timeout:g})
  --attempts <N>       сколько попыток вызова делать (по умолчанию {attempts};
                       повтор бывает только на таймауте и ненулевом коде)
  --log-dir <каталог>  куда tools/claude_cli.py положит промпт и ответ
                       ({tag}-NNN.prompt.txt и {tag}-NNN.response.txt)
  --selftest           прогнать встроенные проверки и выйти (claude не нужен)
  -h, --help           показать эту справку

Каталог для --out создаётся при необходимости, файл пишется в UTF-8 с
ensure_ascii=False и отступом 2.

Коды возврата:
  0  роли получены (в том числе когда говорящий один или говорящих нет)
  1  claude не ответил или ответ негоден — файл записан, но роли в нём равны
     именам говорящих и стоит status «{fallback}» с причиной
  2  вход непригоден (нет файла, не та схема, негодный список говорящих) или
     ошибка в аргументах — файл --out не создаётся и не перетирается

Примеры:
  python tools/roles.py --speakers data/output/лекция/speakers.json \\
      --out data/output/лекция/roles.json --log-dir data/output/лекция/claude
  python tools/roles.py --speakers data/output/лекция/speakers.json \\
      --out data/output/лекция/roles.json --timeout 600 --attempts 1
""".format(
    teacher=TEACHER,
    phrases=MAX_PHRASES,
    chars=MAX_ROLE_CHARS,
    skip=STATUS_SKIP,
    fallback=STATUS_FALLBACK,
    timeout=claude_cli.DEFAULT_TIMEOUT,
    attempts=claude_cli.DEFAULT_ATTEMPTS,
    tag=LOG_TAG,
)


class UsageError(Exception):
    """Ошибка в аргументах командной строки."""


class InputError(Exception):
    """Вход не читается, не той схемы или список говорящих негоден."""


def die(message: str, code: int = 2) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main().

    По умолчанию 2: и негодные аргументы, и непригодный вход означают «роли не
    определялись вовсе». Код 1 в этом инструменте занят другим смыслом — файл
    записан, но ролей в нём нет.
    """
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return code


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но пользователь должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


def format_timecode(seconds: float) -> str:
    """Секунды от начала записи -> «ЧЧ:ММ:СС».

    Тот же формат, что в итоговом документе (tools/render_doc.py): дробная часть
    отбрасывается, а не округляется, части всегда двузначные. Импортировать
    функцию оттуда нельзя — зависимости этого инструмента ограничены claude_cli,
    — поэтому она повторена здесь.
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
class Phrase:
    """Одна фраза говорящего: время начала и текст с пунктуацией."""

    start: float
    text: str


@dataclass
class Speaker:
    """Говорящий из speakers/1 — всё, что о нём нужно знать для промпта.

    name    — имя от диаризации, «Говорящий N»: им же claude отвечает;
    share   — доля речи от всей речи записи, 0..1;
    phrases — сколько у него фраз всего (а не сколько ушло в промпт);
    examples — показательные фразы, уже отобранные и упорядоченные по времени.
    """

    name: str
    share: float
    phrases: int
    examples: list[Phrase] = field(default_factory=list)


@dataclass
class Record:
    """Входной файл: путь и говорящие по убыванию доли речи."""

    path: str
    speakers: list[Speaker]


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


def _number(value: object) -> float | None:
    """Конечное неотрицательное число или None, если это не оно."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _check_schema(data: dict) -> None:
    if "schema" not in data:
        raise InputError("в файле нет поля schema (ожидается %r)" % SPEAKERS_SCHEMA)
    if data["schema"] != SPEAKERS_SCHEMA:
        raise InputError(
            "не та схема: ожидается %r, в файле %r" % (SPEAKERS_SCHEMA, data["schema"])
        )


def _speaker_order_key(name: str) -> tuple[int, str]:
    """Ключ порядка для имён «Говорящий N»: по числу, а не по строке.

    Иначе при равных долях «Говорящий 10» встал бы перед «Говорящим 2», и
    нумерация студентов зависела бы от того, сколько их было.
    """
    found = re.search(r"(\d+)\s*$", name)
    return (int(found.group(1)) if found else 10**9, name)


def _collect_phrases(data: dict, names: set[str]) -> dict[str, list[Phrase]]:
    """Фразы из speakers/1, разложенные по именам говорящих.

    Фраза без говорящего, без текста или без времени в промпт не попадает: имя
    говорящего — это и есть то, по чему фраза здесь нужна, а выдумывать его
    инструмент не вправе. Фраза с незнакомым именем тоже отбрасывается — но о
    ней говорится в stderr: это рассогласование входного файла.
    """
    if "phrases" not in data:
        return {}
    items = data["phrases"]
    if not isinstance(items, list):
        raise InputError(
            "поле phrases должно быть списком, получено: %s" % _typename(items)
        )

    collected: dict[str, list[Phrase]] = {name: [] for name in names}
    skipped = 0
    unknown: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            skipped += 1
            continue
        speaker = item.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            # Фраза без атрибуции: такие в speakers/1 законны (диаризация не
            # покрыла кусок), но для определения ролей они бесполезны.
            continue
        speaker = speaker.strip()
        if speaker not in collected:
            unknown.add(speaker)
            continue
        start = _number(item.get("start"))
        text = item.get("text")
        if start is None or not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        collected[speaker].append(Phrase(start=start, text=" ".join(text.split())))
    if skipped:
        warn("в промпт не попало %d фраз без текста или без времени" % skipped)
    if unknown:
        warn(
            "в phrases есть говорящие, которых нет в списке speakers: %s"
            % ", ".join(sorted(unknown))
        )
    return collected


def pick_examples(phrases: list[Phrase]) -> list[Phrase]:
    """Отбирает показательные фразы: самые длинные, в порядке времени.

    Почему самые длинные. По коротким фразам («да», «угу», «понятно») роль не
    определить — они одинаковы у всех. Объяснение, постановка задачи и вопрос
    по теме длиннее, и именно в них видно, кто ведёт занятие. Порядок остаётся
    временной: claude читает их как куски разговора, а не как рейтинг.
    """
    chosen = sorted(phrases, key=lambda phrase: (-len(phrase.text), phrase.start))
    chosen = chosen[:MAX_PHRASES]
    chosen.sort(key=lambda phrase: phrase.start)
    return chosen


def _shares(entries: list[dict], counts: list[int]) -> list[float]:
    """Доли речи говорящих: из поля share, иначе — выведенные.

    Поле share пишет tools/speakers.py, но этот инструмент должен работать и с
    файлом, где его не оказалось: доля нужна только для порядка и нумерации
    студентов, и вывести её из времени речи (а если и его нет — из числа фраз)
    лучше, чем объявить вход непригодным.
    """
    shares: list[float | None] = []
    for entry in entries:
        value = _number(entry.get("share"))
        shares.append(value if value is not None and value <= 1.0 else None)
    if all(share is not None for share in shares):
        return [float(share) for share in shares]

    warn("у части говорящих нет поля share — доля речи выведена из speech_seconds")
    weights = [_number(entry.get("speech_seconds")) for entry in entries]
    if any(weight is None for weight in weights):
        weights = [float(count) for count in counts]
    total = sum(weight or 0.0 for weight in weights)
    if total <= 0:
        return [0.0 for _ in entries]
    return [
        share if share is not None else (weight or 0.0) / total
        for share, weight in zip(shares, weights)
    ]


def parse_speakers(data: object, path: str) -> Record:
    """Разбирает JSON схемы speakers/1 в список говорящих для промпта."""
    if not isinstance(data, dict):
        raise InputError(
            "на верхнем уровне ожидается объект JSON, получено: %s" % _typename(data)
        )
    _check_schema(data)

    if "speakers" not in data:
        raise InputError("в файле нет поля speakers")
    entries = data["speakers"]
    if not isinstance(entries, list):
        raise InputError(
            "поле speakers должно быть списком, получено: %s" % _typename(entries)
        )

    names: list[str] = []
    for position, entry in enumerate(entries):
        where = "говорящий №%d" % (position + 1)
        if not isinstance(entry, dict):
            raise InputError(
                "%s должен быть объектом JSON, получено: %s" % (where, _typename(entry))
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InputError("%s: поле name должно быть непустой строкой" % where)
        name = " ".join(name.split())
        if name in names:
            # Два говорящих с одним именем делают соответствие бессмысленным:
            # непонятно, кому из них достанется роль из ответа claude.
            raise InputError("имя говорящего встречается дважды: %s" % name)
        names.append(name)

    phrases = _collect_phrases(data, set(names))
    counts: list[int] = []
    for name, entry in zip(names, entries):
        declared = entry.get("phrases")
        if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0:
            counts.append(declared)
        else:
            counts.append(len(phrases.get(name, [])))

    shares = _shares(list(entries), counts)
    speakers = [
        Speaker(
            name=name,
            share=share,
            phrases=count,
            examples=pick_examples(phrases.get(name, [])),
        )
        for name, share, count in zip(names, shares, counts)
    ]
    # Порядок по убыванию доли речи: по нему нумеруются студенты и он же
    # определяет, кому достанется «Преподаватель» при споре.
    speakers.sort(key=lambda speaker: (-speaker.share, _speaker_order_key(speaker.name)))
    return Record(path=path, speakers=speakers)


def load_record(path: str) -> Record:
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
    return parse_speakers(data, path)


# ----------------------------------------------------------------- промпт ---

PROMPT_HEAD = """Определи, кто из говорящих в записи занятия какую роль играет.

Запись — занятие (лекция, семинар, разбор задач) на русском языке. Речь уже
разделена по голосам, и говорящие названы обезличенно: «Говорящий 1»,
«Говорящий 2» и так далее. Кто из них ведёт занятие, а кто учится, видно только
по смыслу речи — это и нужно определить.

Ниже по каждому говорящему даны имя, доля речи, число фраз и несколько
показательных фраз с таймкодами.

Требования к ответу:

1. Ответ состоит только из строк вида

Говорящий N -> Роль

   по одной строке на каждого говорящего из списка ниже. Больше в ответе не
   должно быть ничего: ни вступления, ни пояснений, ни рассуждений о том, как
   ты решал, ни разметки.
2. Имя слева от «->» пиши ровно так, как оно стоит в списке ниже.
3. Занятие ведёт один человек, его роль — «{teacher}». «{teacher}» в ответе
   ровно один.
4. Все остальные — «Студент 1», «Студент 2», «Студент 3» и так далее. Номера
   студентов идут по убыванию доли речи: тот из студентов, кто говорит больше
   всех, — «Студент 1».
5. Других названий ролей нет. Не придумывай своих, не добавляй пояснений в
   скобках и не пиши имён людей, даже если они прозвучали в записи.

Говорящие:

""".format(teacher=TEACHER)


def render_phrase(phrase: Phrase) -> str:
    """Фраза для промпта: «[ЧЧ:ММ:СС] текст», длинная — обрезанная."""
    text = phrase.text
    if len(text) > MAX_PHRASE_CHARS:
        text = text[:MAX_PHRASE_CHARS].rstrip() + "…"
    return "[%s] %s" % (format_timecode(phrase.start), text)


def render_speaker(speaker: Speaker) -> str:
    """Блок одного говорящего: имя, доля речи, число фраз и примеры."""
    head = "%s: доля речи %.1f %%, фраз %d" % (
        speaker.name,
        speaker.share * 100,
        speaker.phrases,
    )
    if not speaker.examples:
        return head + "\nПоказательных фраз нет."
    rows = "\n".join(render_phrase(phrase) for phrase in speaker.examples)
    return "%s\nПоказательные фразы:\n%s" % (head, rows)


def build_prompt(speakers: list[Speaker]) -> str:
    """Собирает промпт единственного вызова: требования и все говорящие."""
    body = "\n\n".join(render_speaker(speaker) for speaker in speakers)
    return PROMPT_HEAD + body + "\n"


# ------------------------------------------------------------ разбор ответа ---

# Стрелка между именем и ролью. Промпт требует «->», но claude пишет и «→», и
# «=>», и длинное тире со стрелкой: выбрасывать из-за этого всё соответствие
# жалко, а спутать такую строку с прозой нельзя.
ARROW_RE = re.compile(r"\s*(?:-+>|=+>|→|—>|–>)\s*")

# Маркер списка и разметка вокруг строки: claude иногда отвечает списком или
# ставит имя в обратные кавычки, хотя промпт этого не просил.
BULLET_RE = re.compile(r"^\s*(?:[-*•+]|\d{1,3}[.)])\s+")
MARKUP_CHARS = "*_`«»\"'"

# Роль «Студент N», как её пишет claude: по ней считаются занятые номера.
STUDENT_RE = re.compile(r"^студент\s*(\d{1,3})$")


def _key(text: str) -> str:
    """Текст -> ключ сравнения: строчные буквы и цифры, ё -> е, без знаков."""
    lowered = text.strip().lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", lowered)


def clean_role(text: str) -> str:
    """Роль из ответа claude -> одна строка не длиннее MAX_ROLE_CHARS.

    Переносы и лишние пробелы снимаются, разметка и знаки препинания по краям
    отбрасываются, длинная роль обрезается. Известные названия приводятся к
    каноническому виду («преподаватель» -> «Преподаватель», «студент  2» ->
    «Студент 2»), а незнакомое название остаётся как есть: ярлык с чужим словом
    читателю понятен, а потерянный ярлык — нет.
    """
    role = " ".join(str(text).split())
    role = role.strip(MARKUP_CHARS + " .,;:—–-")
    role = " ".join(role.split())
    if not role:
        return ""

    lowered = role.lower().replace("ё", "е")
    if lowered == TEACHER.lower():
        return TEACHER
    student = STUDENT_RE.match(lowered)
    if student is not None:
        return STUDENT % int(student.group(1))

    if len(role) > MAX_ROLE_CHARS:
        role = role[:MAX_ROLE_CHARS].rstrip()
    return role


def parse_answer(answer: str, speakers: list[Speaker]) -> dict[str, str]:
    """Ответ claude -> соответствие «имя говорящего → роль».

    Разбирается построчно и терпимо: строка без стрелки — это пояснение или
    вступление, которых промпт не просил, и она просто пропускается. Строка со
    стрелкой, слева от которой не имя из списка, — это уже рассогласование, о
    нём говорится в stderr. Повторную строку про того же говорящего побеждает
    первая: выбирать между двумя ответами claude инструменту не из чего, а
    молча брать последнюю — значит зависеть от порядка строк.
    """
    known = {_key(speaker.name): speaker.name for speaker in speakers}
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    empty = 0

    for line in answer.splitlines():
        text = BULLET_RE.sub("", line.strip(), count=1).strip()
        if not text:
            continue
        parts = ARROW_RE.split(text, maxsplit=1)
        if len(parts) != 2:
            # Без стрелки строка считается мапой только если слева ровно имя
            # говорящего: «Говорящий 1: Преподаватель» разобрать можно, а
            # «Вывод: роли распределились так» — нельзя.
            name, separator, role = text.partition(":")
            if not separator or _key(name) not in known:
                continue
            parts = [name, role]
        name, role = parts[0].strip(MARKUP_CHARS + " "), clean_role(parts[1])
        key = _key(name)
        if key not in known:
            if name:
                unknown.append(name)
            continue
        if not role:
            empty += 1
            continue
        mapping.setdefault(known[key], role)

    if unknown:
        warn(
            "в ответе claude названы говорящие, которых нет в записи: %s"
            % ", ".join(sorted(set(unknown)))
        )
    if empty:
        warn("в ответе claude %d строк без роли справа от стрелки" % empty)
    return mapping


# --------------------------------------------------------------- результат ---


@dataclass
class Assignment:
    """Роль одного говорящего — строка будущего файла roles/1."""

    speaker: str
    role: str
    share: float
    source: str


@dataclass
class Result:
    """Итог прогона: что записывать в файл и с каким кодом выходить."""

    status: str
    reason: str
    assignments: list[Assignment]
    called: bool = False

    @property
    def from_claude(self) -> int:
        return sum(1 for item in self.assignments if item.source == SOURCE_CLAUDE)

    @property
    def defaulted(self) -> int:
        return sum(1 for item in self.assignments if item.source == SOURCE_DEFAULT)


def fallback_result(speakers: list[Speaker], reason: str, called: bool) -> Result:
    """Роли не получены: в файле остаются имена говорящих и причина."""
    text = " ".join(str(reason).split()).rstrip(".") or "причина не названа"
    return Result(
        status=STATUS_FALLBACK,
        reason=text,
        assignments=[
            Assignment(
                speaker=speaker.name,
                role=speaker.name,
                share=speaker.share,
                source=SOURCE_FALLBACK,
            )
            for speaker in speakers
        ],
        called=called,
    )


def resolve_teacher(speakers: list[Speaker], mapping: dict[str, str]) -> dict[str, str]:
    """Оставляет «Преподавателя» одного: им становится тот, кто больше говорил.

    Говорящие приходят по убыванию доли речи, поэтому побеждает первый из них.
    Остальные теряют роль и получают «Студент N» дальше, вместе с теми, кого
    claude не назвал вовсе.
    """
    claimed = [
        speaker.name
        for speaker in speakers
        if _key(mapping.get(speaker.name, "")) == _key(TEACHER)
    ]
    if len(claimed) <= 1:
        return mapping
    warn(
        "claude назвал «%s» нескольких (%s) — роль осталась у «%s», он говорил больше"
        % (TEACHER, ", ".join(claimed), claimed[0])
    )
    return {
        name: role
        for name, role in mapping.items()
        if name == claimed[0] or _key(role) != _key(TEACHER)
    }


def assign_roles(speakers: list[Speaker], mapping: dict[str, str]) -> Result:
    """Собирает итоговое соответствие из ответа claude и умолчаний.

    Говорящий, роли которому не досталось, получает «Студент N» со свободным
    номером: занятыми считаются номера студентов, которых назвал claude, —
    иначе в документе оказалось бы два «Студента 1».
    """
    mapping = resolve_teacher(speakers, mapping)

    taken: set[int] = set()
    for role in mapping.values():
        student = STUDENT_RE.match(role.lower().replace("ё", "е"))
        if student is not None:
            taken.add(int(student.group(1)))

    assignments: list[Assignment] = []
    duplicates: list[str] = []
    seen: dict[str, str] = {}
    number = 0
    for speaker in speakers:
        role = mapping.get(speaker.name)
        source = SOURCE_CLAUDE
        if role is None:
            number += 1
            while number in taken:
                number += 1
            taken.add(number)
            role = STUDENT % number
            source = SOURCE_DEFAULT
        if role in seen:
            duplicates.append(role)
        seen[role] = speaker.name
        assignments.append(
            Assignment(
                speaker=speaker.name,
                role=role,
                share=speaker.share,
                source=source,
            )
        )

    if duplicates:
        # Переименовывать за claude инструмент не станет: роль — это ярлык в
        # документе, и подмена его молча хуже, чем видимое предупреждение.
        warn(
            "одна роль досталась нескольким говорящим: %s"
            % ", ".join("«%s»" % role for role in sorted(set(duplicates)))
        )
    if not any(_key(item.role) == _key(TEACHER) for item in assignments):
        warn("claude не назвал «%s» никого — в документе не будет этого ярлыка" % TEACHER)

    return Result(status=STATUS_OK, reason="", assignments=assignments, called=True)


def decide(record: Record, opts: dict) -> Result:
    """Главное решение: кто какую роль получает. Вызов claude — ровно один."""
    speakers = record.speakers
    if not speakers:
        return Result(status=STATUS_SKIP, reason=REASON_NO_SPEAKERS, assignments=[])
    if len(speakers) == 1:
        # Звать claude не за чем: выбирать не из кого, а единственный говорящий
        # на занятии — это тот, кто его ведёт.
        speaker = speakers[0]
        return Result(
            status=STATUS_OK,
            reason=REASON_SINGLE,
            assignments=[
                Assignment(
                    speaker=speaker.name,
                    role=TEACHER,
                    share=speaker.share,
                    source=SOURCE_DEFAULT,
                )
            ],
        )

    prompt = build_prompt(speakers)
    try:
        answer = claude_cli.ask(
            prompt,
            timeout=opts["timeout"],
            attempts=opts["attempts"],
            log_dir=opts["log_dir"],
            tag=LOG_TAG,
        )
    except claude_cli.ClaudeUnavailable as exc:
        return fallback_result(speakers, "вызывать claude нечем: %s" % exc, called=False)
    except claude_cli.ClaudeFailed as exc:
        return fallback_result(speakers, "claude не ответил: %s" % exc, called=True)

    mapping = parse_answer(answer, speakers)
    if not mapping:
        return fallback_result(
            speakers,
            "ответ claude негоден: в нём нет ни одной строки вида «Говорящий N -> Роль»",
            called=True,
        )
    return assign_roles(speakers, mapping)


# ------------------------------------------------------------------ запись ---


def build_document(record: Record, result: Result) -> dict:
    """Собирает объект схемы roles/1 — поля ровно те, что ждут читатели файла."""
    return {
        "schema": ROLES_SCHEMA,
        "speakers": record.path,
        "status": result.status,
        "reason": result.reason,
        "roles": [
            {
                "speaker": item.speaker,
                "role": item.role,
                "share": round(item.share, 3),
                "source": item.source,
            }
            for item in result.assignments
        ],
        "stats": {
            "speakers": len(result.assignments),
            "from_claude": result.from_claude,
            "defaulted": result.defaulted,
        },
    }


def write_json_file(path: str, payload: dict) -> None:
    """Пишет JSON в UTF-8: ensure_ascii=False, отступ 2, перевод строки в конце."""
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


REPORT_WIDTH = len("По умолчанию") + 2


def report_row(label: str, value: str) -> str:
    """Строка отчёта: ярлык с двоеточием, выровненный по колонке, и значение."""
    return "%-*s%s\n" % (REPORT_WIDTH, label + ":", value)


def print_report(path: str, size: int, record: Record, result: Result) -> None:
    """Печатает итоги — то, что потом уходит в отчёт о прогоне."""
    sys.stdout.write("================ РОЛИ ================\n")
    sys.stdout.write(report_row("Файл", path))
    sys.stdout.write(report_row("Размер", "%d байт" % size))
    sys.stdout.write(report_row("Вход", "%s (%s)" % (record.path, SPEAKERS_SCHEMA)))
    sys.stdout.write(report_row("Статус", result.status))
    sys.stdout.write(report_row("Говорящих", "%d" % len(result.assignments)))
    sys.stdout.write(report_row("От claude", "%d" % result.from_claude))
    sys.stdout.write(report_row("По умолчанию", "%d" % result.defaulted))
    for item in result.assignments:
        sys.stdout.write(
            "  %s -> %s (доля речи %.1f %%, %s)\n"
            % (item.speaker, item.role, item.share * 100, item.source)
        )
    if result.reason:
        sys.stdout.write("  * %s\n" % result.reason)


# ------------------------------------------------------- разбор аргументов ---

_FLAGS_VALUE = {
    "--speakers": "speakers",
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
        "speakers": None,
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


def check_args(opts: dict) -> str:
    """Проверяет набор аргументов целиком и возвращает путь ко входу."""
    if not opts["speakers"]:
        raise UsageError("не указан --speakers (см. --help)")
    if not opts["out"]:
        raise UsageError("не указан --out (см. --help)")
    # Перетереть собственный вход — это потерять фразы с говорящими: заново их
    # строить нечем, а роли определяются в секунды.
    if os.path.abspath(opts["out"]) == os.path.abspath(opts["speakers"]):
        raise UsageError("--out указывает на тот же файл, что и вход: %s" % opts["out"])
    return opts["speakers"]


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
        path = check_args(opts)
        record = load_record(path)
    except (UsageError, InputError) as exc:
        # Вход непригоден: файл --out не создаётся и не перетирается.
        return die(str(exc))

    result = decide(record, opts)
    try:
        write_json_file(opts["out"], build_document(record, result))
    except InputError as exc:
        # Вход был годен и роли, возможно, получены — не записался сам файл.
        # Это не код 2: вход тут ни при чём, а результата всё равно нет.
        return die("%s — роли не записаны" % exc, 1)

    try:
        size = os.path.getsize(opts["out"])
    except OSError:  # pragma: no cover — файл только что записан
        size = 0
    print_report(opts["out"], size, record, result)

    if result.status == STATUS_FALLBACK:
        sys.stderr.write("%s: роли не получены: %s\n" % (PROG, result.reason))
        return 1
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


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_json(path: str, payload: object) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


# Фразы фикстуры: (говорящий, начало, текст). У «Говорящего 1» их заведомо
# больше пятнадцати — на нём проверяется предел числа примеров в промпте.
_LONG = "Разберём, как устроен этот механизм и почему он работает именно так"


def _fixture_phrases() -> list[tuple[str, float, str]]:
    phrases: list[tuple[str, float, str]] = []
    for number in range(20):
        phrases.append(("Говорящий 1", 10.0 * number, "%s, случай %d." % (_LONG, number)))
    phrases.append(("Говорящий 2", 305.0, "А если данных не хватает, что тогда делать?"))
    phrases.append(("Говорящий 2", 905.0, "Понял, спасибо, теперь стало яснее."))
    phrases.append(("Говорящий 3", 1205.0, "У меня вопрос про срок сдачи работы."))
    phrases.append(("Говорящий 3", 3725.0, "Да."))
    return phrases


def _fixture(names: int = 3) -> dict:
    """Файл схемы speakers/1 для проверок: собирается в памяти, без claude."""
    shares = {"Говорящий 1": 0.62, "Говорящий 2": 0.25, "Говорящий 3": 0.13}
    chosen = ["Говорящий %d" % (number + 1) for number in range(names)]
    phrases = [item for item in _fixture_phrases() if item[0] in chosen]
    return {
        "schema": SPEAKERS_SCHEMA,
        "transcript": "data/output/лекция/transcript.json",
        "punctuated": "data/output/лекция/punctuated.json",
        "diarization": "data/output/лекция/diarization.json",
        "pause": 1.0,
        "speakers": [
            {
                "name": name,
                "label": name,
                "phrases": len([item for item in phrases if item[0] == name]),
                "speech_seconds": round(shares[name] * 3600, 1),
                "share": shares[name],
            }
            for name in chosen
        ],
        "phrases": [
            {
                "index": number,
                "start": start,
                "end": start + 5.0,
                "text": text,
                "speaker": name,
                "sentences": [number, number],
                "paragraph": 0,
                "fallback": False,
                "reason": "speaker",
            }
            for number, (name, start, text) in enumerate(phrases)
        ],
        "stats": {"speakers": len(chosen), "phrases": len(phrases)},
    }


def _good_answer() -> str:
    """Ответ claude, какого требует промпт: по строке на говорящего."""
    return (
        "Говорящий 1 -> Преподаватель\n"
        "Говорящий 2 -> Студент 1\n"
        "Говорящий 3 -> Студент 2\n"
    )


def _test_prompt() -> None:
    """Промпт: имя, доля речи, число фраз, до 15 фраз с таймкодами, формат ответа."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        out = os.path.join(workdir, "roles.json")
        stub = _answering_stub(workdir, "ок", _good_answer())
        code, _, _ = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "удачный прогон вернул код %d" % code)
        _check(_calls(stub) == 1, "вызовов claude %d вместо одного" % _calls(stub))

        prompt = _prompt_of(stub)
        _check("Говорящий N -> Роль" in prompt, "в промпте нет требуемого вида строк")
        _check(TEACHER in prompt, "в промпте не назван «%s»" % TEACHER)
        _check("Студент 1" in prompt, "в промпте не названы студенты")
        _check("по убыванию доли речи" in prompt, "в промпте нет правила нумерации")
        _check(
            "по одной строке на каждого говорящего" in prompt,
            "в промпте не сказано, сколько строк должно быть в ответе",
        )
        _check("ни пояснений" in prompt, "в промпте нет запрета на пояснения")
        for name, share, count in (
            ("Говорящий 1", "62.0 %", 20),
            ("Говорящий 2", "25.0 %", 2),
            ("Говорящий 3", "13.0 %", 2),
        ):
            head = "%s: доля речи %s, фраз %d" % (name, share, count)
            _check(head in prompt, "в промпте нет строки %r" % head)
        _check("[00:00:00] Разберём" in prompt, "в промпте нет фразы с таймкодом")
        _check("[00:05:05] А если данных" in prompt, "в промпте нет фразы второго говорящего")

        # Фраз у первого говорящего двадцать, в промпт ушло пятнадцать.
        block = prompt.split("Говорящий 2:")[0]
        rows = [row for row in block.splitlines() if row.startswith("[")]
        _check(
            len(rows) == MAX_PHRASES,
            "в промпте %d фраз первого говорящего вместо %d" % (len(rows), MAX_PHRASES),
        )
        # Отобраны длинные, а порядок — временной.
        times = [row[1:9] for row in rows]
        _check(times == sorted(times), "фразы в промпте не по времени: %r" % times)
        _check("[00:01:00] Да." not in prompt, "короткая фраза вытеснила длинную")


def _test_good_answer() -> None:
    """Удачный разбор: состав полей файла, статус, stats, код 0."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        out = os.path.join(workdir, "roles.json")
        stub = _answering_stub(workdir, "ок", _good_answer())
        code, report, stderr = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)
        _check("предупреждение" not in stderr, "лишние предупреждения: %r" % stderr)

        data = _read_json(out)
        _check(
            list(data) == ["schema", "speakers", "status", "reason", "roles", "stats"],
            "не тот состав полей: %r" % list(data),
        )
        _check(data["schema"] == ROLES_SCHEMA, "не та схема: %r" % data["schema"])
        _check(data["speakers"] == source, "в поле speakers не путь входа: %r" % data["speakers"])
        _check(data["status"] == STATUS_OK, "статус %r вместо ok" % data["status"])
        _check(data["reason"] == "", "в удачном прогоне есть причина: %r" % data["reason"])
        for item in data["roles"]:
            _check(
                list(item) == ["speaker", "role", "share", "source"],
                "не тот состав полей роли: %r" % list(item),
            )
        _check(
            [(item["speaker"], item["role"], item["source"]) for item in data["roles"]]
            == [
                ("Говорящий 1", TEACHER, SOURCE_CLAUDE),
                ("Говорящий 2", "Студент 1", SOURCE_CLAUDE),
                ("Говорящий 3", "Студент 2", SOURCE_CLAUDE),
            ],
            "не то соответствие: %r" % data["roles"],
        )
        _check(data["roles"][0]["share"] == 0.62, "доля речи потерялась: %r" % data["roles"][0])
        _check(
            data["stats"] == {"speakers": 3, "from_claude": 3, "defaulted": 0},
            "не та статистика: %r" % data["stats"],
        )
        # Файл читаемый человеком: кириллица не экранирована, отступ два пробела.
        text = _read(out)
        _check("Преподаватель" in text, "в файле экранированная кириллица")
        _check('\n  "schema"' in text, "в файле не тот отступ")
        _check(text.endswith("\n"), "в конце файла нет перевода строки")
        _check("Говорящий 1 -> Преподаватель" in report, "в отчёте нет соответствия")


def _test_untidy_answer() -> None:
    """Лишние строки, разметка, чужие имена и длинная роль: разбор устойчив."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        out = os.path.join(workdir, "roles.json")
        answer = (
            "Разобрал запись, вот что получилось:\n"
            "\n"
            "- **Говорящий 1** → преподаватель\n"
            "* `Говорящий 2`: студент 1\n"
            "Говорящий 3 -> Студент, который спрашивает про сроки сдачи работы и молчит\n"
            "Говорящий 7 -> Студент 3\n"
            "\n"
            "Если нужно, могу расписать подробнее.\n"
        )
        stub = _answering_stub(workdir, "мусор", answer)
        code, _, stderr = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)
        _check("Говорящий 7" in stderr, "нет предупреждения о чужом имени: %r" % stderr)

        roles = {item["speaker"]: item["role"] for item in _read_json(out)["roles"]}
        _check(roles["Говорящий 1"] == TEACHER, "роль не приведена к канону: %r" % roles)
        _check(roles["Говорящий 2"] == "Студент 1", "роль из строки с «:» потеряна: %r" % roles)
        long_role = roles["Говорящий 3"]
        _check(
            len(long_role) <= MAX_ROLE_CHARS,
            "роль длиной %d символов: %r" % (len(long_role), long_role),
        )
        _check("\n" not in long_role, "в роли остался перенос строки")
        _check(
            _read_json(out)["stats"]["from_claude"] == 3,
            "не все роли засчитаны как полученные от claude",
        )


def _test_clean_role() -> None:
    """Роль приводится к одной строке не длиннее предела."""
    _check(clean_role("  Преподаватель  ") == TEACHER, "пробелы не сняты")
    _check(clean_role("**преподаватель**") == TEACHER, "разметка не снята")
    _check(clean_role("ПРЕПОДАВАТЕЛЬ.") == TEACHER, "точка и регистр не сняты")
    _check(clean_role("студент  2") == "Студент 2", "студент не приведён к канону")
    _check(clean_role("Студент\n3") == "Студент 3", "перенос строки не снят")
    _check(clean_role("Ведущий\nвстречи") == "Ведущий встречи", "перенос строки не снят")
    _check(clean_role("   ") == "", "пустая роль не распознана")
    long_role = clean_role("Ведущий " + "очень " * 20 + "занятия")
    _check(len(long_role) == MAX_ROLE_CHARS, "длина роли %d" % len(long_role))
    _check("\n" not in long_role, "в обрезанной роли перенос строки")


def _test_defaulted() -> None:
    """Говорящий, которого claude не назвал, получает «Студент N»."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        out = os.path.join(workdir, "roles.json")
        stub = _answering_stub(workdir, "неполный", "Говорящий 1 -> Преподаватель\n")
        code, _, _ = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)

        data = _read_json(out)
        _check(
            [(item["role"], item["source"]) for item in data["roles"]]
            == [
                (TEACHER, SOURCE_CLAUDE),
                ("Студент 1", SOURCE_DEFAULT),
                ("Студент 2", SOURCE_DEFAULT),
            ],
            "не те умолчания: %r" % data["roles"],
        )
        _check(
            data["stats"] == {"speakers": 3, "from_claude": 1, "defaulted": 2},
            "не та статистика: %r" % data["stats"],
        )

        # Номер, который claude уже занял, умолчанию не достаётся.
        taken = _answering_stub(
            workdir, "занятый номер", "Говорящий 1 -> Преподаватель\nГоворящий 3 -> Студент 1\n"
        )
        code, _, _ = _run(["--speakers", source, "--out", out], taken)
        _check(code == 0, "код %d вместо 0" % code)
        roles = {item["speaker"]: item["role"] for item in _read_json(out)["roles"]}
        _check(roles["Говорящий 2"] == "Студент 2", "номер студента занят дважды: %r" % roles)
        _check(roles["Говорящий 3"] == "Студент 1", "роль от claude потеряна: %r" % roles)


def _test_two_teachers() -> None:
    """«Преподавателем» назвали нескольких: роль у того, кто больше говорил."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        out = os.path.join(workdir, "roles.json")
        answer = (
            "Говорящий 2 -> Преподаватель\n"
            "Говорящий 1 -> Преподаватель\n"
            "Говорящий 3 -> Студент 1\n"
        )
        stub = _answering_stub(workdir, "два преподавателя", answer)
        code, _, stderr = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)
        _check(TEACHER in stderr, "нет предупреждения о споре за роль: %r" % stderr)

        data = _read_json(out)
        roles = {item["speaker"]: item["role"] for item in data["roles"]}
        _check(roles["Говорящий 1"] == TEACHER, "роль ушла не тому: %r" % roles)
        _check(roles["Говорящий 2"] == "Студент 2", "проигравший не стал студентом: %r" % roles)
        _check(roles["Говорящий 3"] == "Студент 1", "роль от claude потеряна: %r" % roles)
        _check(
            sum(1 for role in roles.values() if role == TEACHER) == 1,
            "«%s» в файле не один: %r" % (TEACHER, roles),
        )
        source_of = {item["speaker"]: item["source"] for item in data["roles"]}
        _check(
            source_of["Говорящий 2"] == SOURCE_DEFAULT,
            "роль, отобранная у claude, осталась его: %r" % source_of,
        )


def _test_single_speaker() -> None:
    """Говорящий один: «Преподаватель» без вызова claude, код 0."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture(names=1))
        out = os.path.join(workdir, "roles.json")
        # Заглушка, которая падает: если её позовут, это будет видно.
        stub = _stub(workdir, "не звать", "exit 5\n")
        code, report, _ = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)
        _check(_calls(stub) == 0, "claude позвали %d раз вместо нуля" % _calls(stub))

        data = _read_json(out)
        _check(data["status"] == STATUS_OK, "статус %r вместо ok" % data["status"])
        _check("без вызова claude" in data["reason"], "в причине не сказано, почему: %r" % data["reason"])
        _check(
            data["roles"] == [
                {"speaker": "Говорящий 1", "role": TEACHER, "share": 0.62, "source": SOURCE_DEFAULT}
            ],
            "не та роль: %r" % data["roles"],
        )
        _check(
            data["stats"] == {"speakers": 1, "from_claude": 0, "defaulted": 1},
            "не та статистика: %r" % data["stats"],
        )
        _check("без вызова claude" in report, "в отчёте не сказано, почему claude не звали")


def _test_no_speakers() -> None:
    """Говорящих нет: пустое соответствие, статус skip, код 0."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        payload = _fixture()
        payload["speakers"] = []
        payload["phrases"] = []
        source = _write_json(os.path.join(workdir, "speakers.json"), payload)
        out = os.path.join(workdir, "roles.json")
        stub = _stub(workdir, "не звать", "exit 5\n")
        code, report, _ = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)
        _check(_calls(stub) == 0, "claude позвали %d раз вместо нуля" % _calls(stub))

        data = _read_json(out)
        _check(data["status"] == STATUS_SKIP, "статус %r вместо skip" % data["status"])
        _check(data["roles"] == [], "соответствие не пустое: %r" % data["roles"])
        _check(data["reason"], "у статуса skip нет причины")
        _check(
            data["stats"] == {"speakers": 0, "from_claude": 0, "defaulted": 0},
            "не та статистика: %r" % data["stats"],
        )
        _check("РОЛИ" in report, "отчёт не напечатан")


def _test_fallback() -> None:
    """Claude не ответил или ответил негодно: роли = имена, статус fallback, код 1."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        out = os.path.join(workdir, "roles.json")

        cases = (
            ("упал", _stub(workdir, "упал", "exit 5\n"), "claude не ответил"),
            ("пустой ответ", _stub(workdir, "пустой", "exit 0\n"), "claude не ответил"),
            (
                "ответ не по форме",
                _answering_stub(workdir, "проза", "Я не могу определить роли по этой записи.\n"),
                "ответ claude негоден",
            ),
            (
                "нет бинарника",
                os.path.join(workdir, "нет такого файла", "claude"),
                "вызывать claude нечем",
            ),
        )
        for name, binary, needle in cases:
            code, report, stderr = _run(
                ["--speakers", source, "--out", out, "--attempts", "1"], binary
            )
            _check(code == 1, "%s: код %d вместо 1" % (name, code))
            data = _read_json(out)
            _check(data["status"] == STATUS_FALLBACK, "%s: статус %r" % (name, data["status"]))
            _check(needle in data["reason"], "%s: не та причина: %r" % (name, data["reason"]))
            _check(
                all(item["role"] == item["speaker"] for item in data["roles"]),
                "%s: роли не равны именам говорящих: %r" % (name, data["roles"]),
            )
            _check(
                all(item["source"] == SOURCE_FALLBACK for item in data["roles"]),
                "%s: не тот источник роли: %r" % (name, data["roles"]),
            )
            _check(
                data["stats"] == {"speakers": 3, "from_claude": 0, "defaulted": 0},
                "%s: не та статистика: %r" % (name, data["stats"]),
            )
            _check(data["reason"] in stderr, "%s: причина не продублирована в stderr" % name)
            _check("fallback" in report, "%s: в отчёте нет статуса" % name)


def _test_broken_input() -> None:
    """Непригодный вход — код 2, и --out не создаётся и не перетирается."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        out = os.path.join(workdir, "roles.json")
        guard = "уже записанный файл\n"
        broken = os.path.join(workdir, "битый.json")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write("{не json")

        alien = _write_json(os.path.join(workdir, "чужая.json"), {"schema": "transcript/1"})
        no_field = _write_json(os.path.join(workdir, "без поля.json"), {"schema": SPEAKERS_SCHEMA})
        not_list = _write_json(
            os.path.join(workdir, "не список.json"),
            {"schema": SPEAKERS_SCHEMA, "speakers": {"name": "Говорящий 1"}},
        )
        no_name = _write_json(
            os.path.join(workdir, "без имени.json"),
            {"schema": SPEAKERS_SCHEMA, "speakers": [{"share": 1.0}]},
        )
        twice = _write_json(
            os.path.join(workdir, "дважды.json"),
            {
                "schema": SPEAKERS_SCHEMA,
                "speakers": [{"name": "Говорящий 1"}, {"name": "Говорящий 1"}],
            },
        )
        good = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        stub = _answering_stub(workdir, "ок2", _good_answer())

        cases = (
            (["--speakers", os.path.join(workdir, "нет.json"), "--out", out], "нет файла"),
            (["--speakers", broken, "--out", out], "битый JSON"),
            (["--speakers", alien, "--out", out], "чужая схема"),
            (["--speakers", no_field, "--out", out], "нет поля speakers"),
            (["--speakers", not_list, "--out", out], "speakers не список"),
            (["--speakers", no_name, "--out", out], "говорящий без имени"),
            (["--speakers", twice, "--out", out], "имя дважды"),
            (["--speakers", workdir, "--out", out], "вместо файла каталог"),
            (["--out", out], "нет --speakers"),
            (["--speakers", good], "нет --out"),
            (["--speakers", good, "--out", good], "--out поверх входа"),
            (["--speakers", good, "--out", out, "--timeout", "0"], "нулевой таймаут"),
            (["--speakers", good, "--out", out, "--attempts", "нет"], "негодные попытки"),
            (["--speakers", good, "--out", out, "--лишнее"], "неизвестная опция"),
            (["--speakers", good, "--out", out, "лишний"], "лишний аргумент"),
            (["--selftest=1"], "--selftest со значением"),
            (["--speakers"], "--speakers без значения"),
        )
        for argv, name in cases:
            with open(out, "w", encoding="utf-8") as handle:
                handle.write(guard)
            code, _, stderr = _run(argv, stub)
            _check(code == 2, "%s: код %d вместо 2" % (name, code))
            _check(stderr.startswith("%s: " % PROG), "%s: не то сообщение: %r" % (name, stderr))
            _check(_read(out) == guard, "%s: существующий файл перетёрт" % name)

        # Тот же набор, но файла --out нет вовсе: он не должен появиться.
        os.remove(out)
        code, _, _ = _run(["--speakers", broken, "--out", out], stub)
        _check(code == 2, "код %d вместо 2" % code)
        _check(not os.path.exists(out), "при коде 2 файл --out создан")
        _check(_calls(stub) == 0, "при непригодном входе звали claude")


def _test_shares_and_order() -> None:
    """Порядок по доле речи и выведенная доля, когда поля share нет."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        payload = _fixture()
        # Доли перепутаны местами относительно порядка в файле.
        payload["speakers"] = list(reversed(payload["speakers"]))
        source = _write_json(os.path.join(workdir, "speakers.json"), payload)
        out = os.path.join(workdir, "roles.json")
        stub = _answering_stub(workdir, "ок", "Говорящий 1 -> Преподаватель\n")
        code, _, _ = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)
        names = [item["speaker"] for item in _read_json(out)["roles"]]
        _check(
            names == ["Говорящий 1", "Говорящий 2", "Говорящий 3"],
            "порядок не по убыванию доли речи: %r" % names,
        )

        # Без поля share доля выводится из speech_seconds, и порядок тот же.
        payload = _fixture()
        for entry in payload["speakers"]:
            del entry["share"]
        source = _write_json(os.path.join(workdir, "без доли.json"), payload)
        code, _, stderr = _run(["--speakers", source, "--out", out], stub)
        _check(code == 0, "код %d вместо 0" % code)
        _check("share" in stderr, "о выведенной доле речи не сказано: %r" % stderr)
        shares = [item["share"] for item in _read_json(out)["roles"]]
        _check(shares == sorted(shares, reverse=True), "доли не по убыванию: %r" % shares)
        _check(abs(shares[0] - 0.62) < 0.001, "доля выведена неверно: %r" % shares)


def _test_log_dir() -> None:
    """--log-dir: промпт и ответ claude остаются на диске."""
    with tempfile.TemporaryDirectory(prefix="roles-selftest-") as workdir:
        source = _write_json(os.path.join(workdir, "speakers.json"), _fixture())
        out = os.path.join(workdir, "roles.json")
        log_dir = os.path.join(workdir, "claude")
        stub = _answering_stub(workdir, "ок", _good_answer())
        code, _, _ = _run(["--speakers", source, "--out", out, "--log-dir", log_dir], stub)
        _check(code == 0, "код %d вместо 0" % code)
        names = sorted(os.listdir(log_dir))
        _check(
            names == ["%s-001.prompt.txt" % LOG_TAG, "%s-001.response.txt" % LOG_TAG],
            "в каталоге логов не те файлы: %r" % names,
        )
        prompt_log = _read(os.path.join(log_dir, "%s-001.prompt.txt" % LOG_TAG))
        _check("Говорящий 1: доля речи" in prompt_log, "в логе промпта нет говорящих")
        response_log = _read(os.path.join(log_dir, "%s-001.response.txt" % LOG_TAG))
        _check(TEACHER in response_log, "в логе ответа нет ответа claude")

        # Неудачный вызов тоже оставляет оба файла — иначе непонятно, что ушло.
        broken = _stub(workdir, "упал2", "exit 5\n")
        code, _, _ = _run(
            ["--speakers", source, "--out", out, "--log-dir", log_dir, "--attempts", "1"],
            broken,
        )
        _check(code == 1, "неудачный вызов вернул код %d" % code)
        failure = _read(os.path.join(log_dir, "%s-002.response.txt" % LOG_TAG))
        _check(failure.startswith("НЕ ПОЛУЧЕНО"), "в логе неудачи нет пометки: %r" % failure)


def _test_help() -> None:
    """Справка — на русском, со всеми флагами, код 0."""
    for flag in ("-h", "--help"):
        code, out, _ = _run([flag])
        _check(code == 0, "%s вернул код %d" % (flag, code))
        for needle in (
            "--speakers",
            "--out",
            "--timeout",
            "--attempts",
            "--log-dir",
            "--selftest",
            "-h, --help",
            "Использование",
            "Коды возврата",
            TEACHER,
            "Студент 1",
            "по убыванию",
            "roles/1",
            "speakers/1",
        ):
            _check(needle in out, "в справке (%s) нет %r" % (flag, needle))
        _check(re.search(r"[а-яА-Я]", out) is not None, "справка не на русском")


def _test_only_stdlib() -> None:
    """Импорты — только стандартная библиотека и claude_cli."""
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
        ("промпт: говорящие, доли, фразы с таймкодами", _test_prompt),
        ("удачный ответ: состав файла roles/1", _test_good_answer),
        ("неряшливый ответ разбирается устойчиво", _test_untidy_answer),
        ("чистка роли до одной короткой строки", _test_clean_role),
        ("неназванный говорящий получает «Студент N»", _test_defaulted),
        ("несколько «%s»: роль у того, кто больше говорил" % TEACHER, _test_two_teachers),
        ("один говорящий — без вызова claude", _test_single_speaker),
        ("говорящих нет — статус skip", _test_no_speakers),
        ("claude не ответил или ответ негоден — откат и код 1", _test_fallback),
        ("непригодный вход и коды возврата", _test_broken_input),
        ("доли речи и порядок говорящих", _test_shares_and_order),
        ("--log-dir: промпт и ответ на диске", _test_log_dir),
        ("справка", _test_help),
        ("импорты только из стандартной библиотеки", _test_only_stdlib),
    )
    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ roles.py ================\n")
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
