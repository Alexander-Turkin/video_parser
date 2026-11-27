#!/usr/bin/env python3
"""cleanup_runs.py — удаление лишних файлов из рабочих подкаталогов прогонов.

Прогон scripts/transcribe_lecture.sh оставляет рядом с итоговым документом
рабочий подкаталог, в котором лежит всё, что понадобилось по пути: audio.wav
(часто гигабайты), chunks/ с сотнями кусков, plan.json, transcript.txt,
transcript.srt, checks.json, checks.txt, ollama/ с промптами и ответами,
run.log. Из этого стоит хранить ровно шесть файлов:

  transcript.json   дословное распознавание с таймкодами — его не воспроизвести
                    без двух часов работы модели;
  diarization.json  интервалы говорящих — pyannote на CPU считает примерно в
                    реальном времени, на лекции это десятки минут;
  punctuated.json   распознанное с пунктуацией и предложениями — цена
                    восстановления это десятки вызовов ollama;
  summary.md        фрагмент саммари — один вызов ollama, но по живой записи;
  roles.json        роли говорящих («Преподаватель», «Студент N») — ещё один
                    вызов ollama;
  speakers.json     расшифровка, разбитая на фразы с говорящими — считается
                    локально за секунды, но только из первых трёх файлов.

Всё остальное либо воспроизводится из этих шести файлов за секунды, либо вообще
не нужно после сборки документа. Инструмент удаляет всё остальное.

Решения, которые стоит знать, прежде чем читать код:

* Белый список один и тот же для любого каталога. Проверки «похож ли каталог на
  прогон» здесь сознательно нет: имён-хранителей достаточно, а предупреждения
  про «чужой каталог» только мешают. Отсюда же --run, который чистит ровно то,
  что ему дали.
* По символическим ссылкам инструмент не переходит нигде. audio.wav у прогона —
  это обычно ссылка на исходную запись пользователя (её делает
  scripts/transcribe_lecture.sh, когда вход уже 16 кГц моно WAV), и удалить по
  ней настоящую лекцию было бы невосполнимой потерей. Поэтому вместо
  shutil.rmtree здесь свой обход: он различает «каталог» и «ссылка на каталог»
  по os.scandir(follow_symlinks=False) и ссылки только отвязывает.
* Каталог без единого файла-хранителя удаляется целиком — это оборвавшийся
  прогон, в котором остались только аудио, чанки и лог.
* Файлы, лежащие прямо в data/output (итоговые «<имя>-<дата>.md»), не
  обрабатываются никогда: в работу идут только непосредственные подкаталоги.
* Размер ссылки — это размер самой ссылки, а не записи, на которую она
  показывает: в освобождённое место чужие гигабайты не записываются.
* Список удалённого печатается по верхнеуровневым записям каталога, а вложенный
  каталог — одной строкой с числом файлов внутри. Иначе отчёт по одному прогону
  это 470 строк имён чанков, в которых не видно ничего полезного.

Зависимостей нет: только стандартная библиотека. Проверки живут в самом файле
(--selftest): pytest в окружении проекта нет. Селф-тест работает исключительно
на каталогах из tempfile и отдельно убеждается, что настоящий data/output после
всех проверок остался байт в байт таким же.

Использование:
  python tools/cleanup_runs.py [--dry-run]
  python tools/cleanup_runs.py --run <каталог> [--dry-run]
  python tools/cleanup_runs.py --selftest
"""

from __future__ import annotations

import io
import os
import re
import stat
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

PROG = "cleanup_runs.py"

# Файлы-хранители: остаются в каталоге, всё прочее удаляется. Порядок задаёт
# порядок в отчёте — от самого дорогого к самому дешёвому по стоимости
# повторного получения: часы работы модели, десятки минут pyannote на CPU,
# десятки вызовов ollama, один вызов ollama, один вызов ollama, секунды локально.
KEEP_NAMES = (
    "transcript.json",
    "diarization.json",
    "punctuated.json",
    "summary.md",
    "roles.json",
    "speakers.json",
)

# Корень берётся от расположения файла, а не от текущего каталога: инструмент
# чистит data/output этого проекта, и результат не должен зависеть от того,
# откуда его позвали. Флага выбора другого корня нет сознательно — для «другого
# каталога» есть --run.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(REPO_ROOT, "data", "output")

# Хранители в справке — по три имени в строке: шесть имён одной строкой не влезают
# в 80 столбцов и в терминале переносятся как попало.
KEEP_NAMES_HELP = "\n".join(
    "  " + ", ".join(KEEP_NAMES[start:start + 3]) for start in range(0, len(KEEP_NAMES), 3)
)

HELP = f"""Использование:
  python tools/cleanup_runs.py [--dry-run]
  python tools/cleanup_runs.py --run <каталог> [--dry-run]
  python tools/cleanup_runs.py --selftest

Чистит рабочие подкаталоги прогонов: в каталоге остаются только файлы-хранители

{KEEP_NAMES_HELP}

а всё остальное удаляется — audio.wav, plan.json, chunks/, transcript.txt,
transcript.srt, checks.json, checks.txt, ollama/, run.log и любые другие файлы и
подкаталоги (вложенные — рекурсивно). Каталог, в котором не осталось ни одного
из этих шести файлов, удаляется целиком вместе с содержимым.

ВНИМАНИЕ: удаление необратимо. Файлы не переносятся в корзину и не
архивируются, восстановить их нечем. Посмотрите сначала, что будет удалено,
флагом --dry-run.

Без аргументов обрабатываются все непосредственные подкаталоги каталога прогонов

  {DEFAULT_ROOT}

Файлы, лежащие прямо там (итоговые документы «<имя>-<дата>.md»), не удаляются и
не изменяются никогда, вглубь подкаталогов поиск прогонов не идёт.

Опции:
  --run <каталог>  обработать ровно один указанный каталог — любой, проверки
                   «похож ли он на прогон» нет: белого списка из шести файлов
                   достаточно
  --dry-run        только показать, что было бы удалено, и не удалять ничего
  --selftest       прогнать встроенные проверки и выйти
  -h, --help       показать эту справку

audio.wav у прогона обычно символическая ссылка на исходную запись: удаляется
сама ссылка, по ссылкам инструмент не переходит, исходная запись за пределами
каталога не затрагивается ни при каком раскладе.

Коды возврата:
  0  отработано, в том числе когда чистить нечего и когда каталога нет
  1  что-то удалить не удалось (нет прав, файл занят)
  2  ошибка в аргументах или --run указывает на несуществующий путь либо не на
     каталог

Примеры:
  python tools/cleanup_runs.py --dry-run
  python tools/cleanup_runs.py
  python tools/cleanup_runs.py --run "data/output/Лекция 1-2026-10-03" --dry-run
"""


class UsageError(Exception):
    """Ошибка в аргументах командной строки или в указанном пути."""


def die(message: str, code: int = 2) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main().

    По умолчанию 2: все ошибки вызова означают «ничего не чистили», а код 1 в
    этом инструменте занят другим смыслом — чистили, но что-то не удалилось.
    """
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return code


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но пользователь должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


# ------------------------------------------------------------ форматирование ---


def fmt_bytes(size: int) -> str:
    """Байты -> человеческая строка. Единицы те же, что у fmt_bytes в common.sh.

    Байты печатаются целыми, остальное — с двумя знаками: «284.86 МиБ» читается
    глазами, а «298548990» нет.
    """
    value = float(size)
    units = ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ")
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    if index == 0:
        return "%d %s" % (int(value), units[index])
    return "%.2f %s" % (value, units[index])


def plural_files(count: int) -> str:
    """«1 файл», «2 файла», «5 файлов» — отчёт читает человек, а не машина."""
    tail = count % 100
    if 11 <= tail <= 14:
        word = "файлов"
    elif count % 10 == 1:
        word = "файл"
    elif 2 <= count % 10 <= 4:
        word = "файла"
    else:
        word = "файлов"
    return "%d %s" % (count, word)


# ------------------------------------------------------- обход и удаление ---


def _lstat_size(path: str) -> int:
    """Размер самой записи, без перехода по ссылке. Недоступна — считаем нулём.

    Ноль, а не ошибка: размер нужен только для отчёта, и терять из-за него
    удаление файла нельзя.
    """
    try:
        return os.lstat(path).st_size
    except OSError:
        return 0


def _scandir(path: str, errors: list[str]) -> list[os.DirEntry]:
    """Содержимое каталога одним списком. Беда с чтением — в errors, не трейсбек.

    Список, а не итератор: дальше содержимое удаляется, а менять каталог под
    открытым os.scandir — способ получить неопределённый обход.
    """
    try:
        with os.scandir(path) as entries:
            return sorted(entries, key=lambda entry: entry.name)
    except OSError as exc:
        errors.append("не удалось прочитать каталог %s (%s)" % (path, exc.strerror or exc))
        return []


def tree_stats(path: str, errors: list[str]) -> tuple[int, int]:
    """Сколько файлов и байт в каталоге. Ссылки считаются собой, переходов нет.

    Нужна для --dry-run: там удалять нечего, а сказать, сколько освободится,
    надо. Сам каталог в число файлов не входит — считаются только файлы.
    """
    files = 0
    size = 0
    for entry in _scandir(path, errors):
        if entry.is_dir(follow_symlinks=False):
            sub_files, sub_size = tree_stats(entry.path, errors)
            files += sub_files
            size += sub_size
        else:
            files += 1
            size += _lstat_size(entry.path)
    return files, size


def remove_file(path: str, errors: list[str]) -> tuple[int, int]:
    """Удаляет файл или символическую ссылку. Возвращает (сколько, сколько байт).

    Размер снимается до удаления и только с самой записи: для ссылки это длина
    пути, на который она показывает, а не вес исходной лекции.

    Открытый кем-то файл (например, run.log, в который пишет tee из работающего
    прогона) на macOS отвязывается без ошибки — место освободится, когда
    владелец закроет дескриптор. Падать на таком случае нельзя: инструмент
    может быть вызван из самого прогона.
    """
    size = _lstat_size(path)
    try:
        os.unlink(path)
    except OSError as exc:
        errors.append("не удалось удалить %s (%s)" % (path, exc.strerror or exc))
        return 0, 0
    return 1, size


def remove_tree(path: str, errors: list[str]) -> tuple[int, int]:
    """Удаляет каталог со всем содержимым, не переходя по символическим ссылкам.

    Своя рекурсия вместо shutil.rmtree: здесь видно, что ссылка на каталог
    только отвязывается (is_dir(follow_symlinks=False) для неё ложь, и она
    уходит в remove_file), а значит, содержимое по ту сторону ссылки не
    затрагивается. Возвращает, сколько файлов и байт действительно удалено:
    то, что не удалилось, в отчёт как освобождённое не попадает.
    """
    files = 0
    size = 0
    for entry in _scandir(path, errors):
        if entry.is_dir(follow_symlinks=False):
            sub_files, sub_size = remove_tree(entry.path, errors)
        else:
            sub_files, sub_size = remove_file(entry.path, errors)
        files += sub_files
        size += sub_size

    try:
        os.rmdir(path)
    except OSError as exc:
        errors.append("не удалось удалить каталог %s (%s)" % (path, exc.strerror or exc))
    return files, size


# -------------------------------------------------------------- один каталог ---


@dataclass
class Removed:
    """Верхнеуровневая запись каталога, которая удалена (или была бы удалена)."""

    name: str
    is_dir: bool
    is_link: bool
    files: int
    size: int


@dataclass
class DirResult:
    """Чем закончилась чистка одного каталога — это же уходит в отчёт."""

    path: str
    kept: list[str] = field(default_factory=list)
    removed: list[Removed] = field(default_factory=list)
    files: int = 0
    size: int = 0
    errors: list[str] = field(default_factory=list)
    dir_removed: bool = False


def _is_keeper(entry: os.DirEntry) -> bool:
    """Хранитель — это обычный файл с нужным именем, и только он.

    Не ссылка и не каталог: ссылку с именем transcript.json пришлось бы либо
    оставить болтаться (а инструмент по ссылкам не ходит и не может сказать,
    куда она ведёт и жива ли), либо перейти по ней, чего делать нельзя. Каталог
    с таким именем — тем более не файл-хранитель.
    """
    if entry.name not in KEEP_NAMES:
        return False
    return entry.is_file(follow_symlinks=False)


def clean_dir(path: str, dry_run: bool = False) -> DirResult:
    """Чистит один каталог: оставляет файлы-хранители, удаляет всё остальное.

    Каталог, в котором не нашлось ни одного хранителя, удаляется целиком — так
    решил владелец: это оборвавшийся прогон, и держать в data/output аудио с
    чанками от него незачем. Если при чистке что-то не удалилось, сам каталог не
    трогается: пустым он всё равно не стал, а внятная ошибка полезнее второй
    ошибки от os.rmdir.
    """
    result = DirResult(path=path)
    entries = _scandir(path, result.errors)
    if result.errors:
        # Каталог не прочитался: удалять наугад нечего, и решение «нет
        # хранителей, значит удалить целиком» на пустом списке было бы ложным.
        return result

    for entry in entries:
        if _is_keeper(entry):
            result.kept.append(entry.name)
            continue

        is_link = entry.is_symlink()
        is_dir = entry.is_dir(follow_symlinks=False)
        if is_dir:
            if dry_run:
                files, size = tree_stats(entry.path, result.errors)
            else:
                files, size = remove_tree(entry.path, result.errors)
        elif dry_run:
            files, size = 1, _lstat_size(entry.path)
        else:
            files, size = remove_file(entry.path, result.errors)

        result.removed.append(
            Removed(name=entry.name, is_dir=is_dir, is_link=is_link, files=files, size=size)
        )
        result.files += files
        result.size += size

    result.kept.sort(key=KEEP_NAMES.index)

    if not result.kept and not result.errors:
        if dry_run:
            result.dir_removed = True
        else:
            try:
                os.rmdir(path)
            except OSError as exc:
                result.errors.append(
                    "не удалось удалить каталог %s (%s)" % (path, exc.strerror or exc)
                )
            else:
                result.dir_removed = True
    return result


def list_run_dirs(root: str) -> list[str]:
    """Непосредственные подкаталоги корня — и ничего больше.

    Файлы (итоговые «<имя>-<дата>.md») не попадают сюда вообще: их имена даже не
    доходят до чистки. Ссылка на каталог пропускается с предупреждением —
    перейти по ней значило бы чистить неизвестно что неизвестно где.
    """
    errors: list[str] = []
    dirs: list[str] = []
    for entry in _scandir(root, errors):
        if entry.is_symlink():
            if entry.is_dir():
                warn("пропущена символическая ссылка на каталог: %s" % entry.path)
            continue
        if entry.is_dir(follow_symlinks=False):
            dirs.append(entry.path)
    for message in errors:
        raise UsageError(message)
    return dirs


# ------------------------------------------------------------------- отчёт ---


def describe(item: Removed) -> str:
    """Строка списка удалённого: «chunks/ — 470 файлов, 280.00 МиБ»."""
    name = item.name + ("/" if item.is_dir else "")
    parts: list[str] = []
    if item.is_link:
        # У ссылки свой размер (длина пути) уже посчитан, но главное сказать,
        # что это ссылка: читатель должен видеть, что исходник не тронут.
        parts.append("символическая ссылка")
    elif item.is_dir:
        parts.append(plural_files(item.files))
    parts.append(fmt_bytes(item.size))
    return "%s — %s" % (name, ", ".join(parts))


def print_header(root: str, target: str | None, dry_run: bool) -> None:
    """Шапка отчёта: что чистим, где и в каком режиме."""
    sys.stdout.write("================ ОЧИСТКА ПРОГОНОВ ================\n")
    if target is None:
        sys.stdout.write("Что чистим:  все подкаталоги каталога прогонов\n")
        sys.stdout.write("Путь:        %s\n" % root)
    else:
        sys.stdout.write("Что чистим:  один каталог, указанный через --run\n")
        sys.stdout.write("Путь:        %s\n" % target)
    sys.stdout.write(
        "Режим:       %s\n"
        % (
            "--dry-run, только показать: ничего не удаляется"
            if dry_run
            else "удаление, необратимо"
        )
    )
    sys.stdout.write("Хранители:   %s\n" % ", ".join(KEEP_NAMES))


def print_dir_report(result: DirResult, dry_run: bool) -> None:
    """Отчёт по одному каталогу: что удалено (сколько файлов и байт) и что осталось."""
    sys.stdout.write("\nКаталог: %s\n" % result.path)

    if result.removed:
        sys.stdout.write(
            "  %s: %s, %s\n"
            % (
                "будет удалено" if dry_run else "удалено",
                plural_files(result.files),
                fmt_bytes(result.size),
            )
        )
        for item in result.removed:
            sys.stdout.write("    %s\n" % describe(item))
    else:
        sys.stdout.write("  удалять нечего: лишних файлов в каталоге нет\n")

    if result.kept:
        sys.stdout.write(
            "  %s: %s\n" % ("останется" if dry_run else "оставлено", ", ".join(result.kept))
        )
    elif result.dir_removed:
        sys.stdout.write(
            "  файлов-хранителей нет: каталог %s целиком\n"
            % ("был бы удалён" if dry_run else "удалён")
        )
    else:
        sys.stdout.write("  файлов-хранителей нет, но каталог оставлен: удалилось не всё\n")

    for message in result.errors:
        sys.stdout.write("  ОШИБКА: %s\n" % message)


def print_total(results: list[DirResult], dry_run: bool) -> None:
    """Общий итог в конце — числа по всем обработанным каталогам сразу."""
    files = sum(result.files for result in results)
    size = sum(result.size for result in results)
    dirs_removed = sum(1 for result in results if result.dir_removed)
    errors = sum(len(result.errors) for result in results)

    sys.stdout.write("\n---------------- ИТОГ ----------------\n")
    sys.stdout.write("Обработано каталогов:      %d\n" % len(results))
    sys.stdout.write(
        "%s %s, %s\n"
        % (
            "Было бы удалено:          " if dry_run else "Удалено:                  ",
            plural_files(files),
            fmt_bytes(size),
        )
    )
    sys.stdout.write(
        "%s %d\n"
        % (
            "Каталогов целиком:        " if dry_run else "Удалено каталогов целиком:",
            dirs_removed,
        )
    )
    if errors:
        sys.stdout.write("Не удалось удалить:        %d объектов\n" % errors)
    if dry_run:
        sys.stdout.write("\nЭто был --dry-run: на диске ничего не изменилось.\n")


# ------------------------------------------------------------------ чистка ---


def check_target(target: str) -> None:
    """Проверяет путь из --run. Любая беда — UsageError, то есть код 2.

    Проверяется ровно одно: это существующий каталог, а не файл и не ссылка.
    «Похож ли он на прогон» не проверяется сознательно. Ссылка отвергается
    потому, что чистка пошла бы по ней в чужое место, а инструмент по ссылкам не
    ходит нигде.
    """
    if not os.path.lexists(target):
        raise UsageError("--run указывает на несуществующий путь: %s" % target)
    if os.path.islink(target):
        raise UsageError(
            "--run указывает на символическую ссылку, а не на каталог: %s "
            "(по ссылкам инструмент не переходит)" % target
        )
    if not os.path.isdir(target):
        raise UsageError("--run указывает не на каталог: %s" % target)


def cleanup(root: str, target: str | None, dry_run: bool) -> int:
    """Делает всю работу и печатает отчёт. Возвращает код возврата для main().

    Корень и цель передаются аргументами, а не берутся из констант: так
    селф-тест прогоняет и ветку «без аргументов» на временном каталоге, ни разу
    не задев настоящий data/output.
    """
    if target is not None:
        check_target(target)
        targets = [target]
    else:
        if not os.path.exists(root):
            print_header(root, target, dry_run)
            sys.stdout.write("\nКаталога прогонов нет — чистить нечего.\n")
            return 0
        if not os.path.isdir(root):
            raise UsageError("каталог прогонов оказался файлом, а не каталогом: %s" % root)
        targets = list_run_dirs(root)

    print_header(root, target, dry_run)
    if not targets:
        sys.stdout.write("\nПодкаталогов прогонов нет — чистить нечего.\n")
        return 0

    results = [clean_dir(path, dry_run) for path in targets]
    for result in results:
        print_dir_report(result, dry_run)
    print_total(results, dry_run)

    failed = sum(len(result.errors) for result in results)
    if failed:
        sys.stderr.write(
            "%s: не удалось удалить %d объектов — подробности выше\n" % (PROG, failed)
        )
        return 1
    return 0


# --------------------------------------------------------- разбор аргументов ---

# Флаги без значения: имя в командной строке -> ключ в opts.
_FLAGS_PLAIN = {
    "-h": "help",
    "--help": "help",
    "--selftest": "selftest",
    "--dry-run": "dry_run",
}


def parse_args(argv: list[str]) -> dict:
    """Разбирает аргументы так же, как это делают остальные инструменты проекта.

    Своим разбором, а не argparse: тот печатает usage и свои ошибки
    по-английски, а весь остальной вывод инструментов проекта — на русском.
    """
    opts: dict = {"run": None, "dry_run": False, "help": False, "selftest": False}

    position = 0
    while position < len(argv):
        token = argv[position]
        inline: str | None = None
        if token.startswith("--") and "=" in token:
            token, _, inline = token.partition("=")

        if token in _FLAGS_PLAIN:
            if inline is not None:
                raise UsageError("%s не принимает значение" % token)
            opts[_FLAGS_PLAIN[token]] = True
            position += 1
            continue

        if token == "--run":
            if inline is None:
                if position + 1 >= len(argv):
                    raise UsageError("--run требует каталог")
                value = argv[position + 1]
                position += 2
            else:
                value = inline
                position += 1
            if not value:
                raise UsageError("--run требует непустое значение")
            opts["run"] = value
            continue

        if token.startswith("-"):
            raise UsageError("неизвестная опция: %s (см. --help)" % token)
        raise UsageError("лишний аргумент: %s (см. --help)" % token)

    return opts


# ---------------------------------------------------------------- селф-тест ---


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Сквозной прогон main() с перехватом вывода.

    В селф-тесте main() вызывается только с --run, --help, --selftest-подобными
    и заведомо негодными аргументами. main([]) не вызывается нигде и никогда:
    он чистил бы настоящий data/output. Ветка «без аргументов» проверяется
    через cleanup() на временном корне.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _write(path: str, content: str = "x") -> str:
    """Файл с содержимым — чтобы у него был ненулевой размер и было что сверять."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def _make_run(
    parent: str,
    name: str,
    *,
    keepers: tuple[str, ...] = KEEP_NAMES,
    audio_link: str | None = None,
) -> str:
    """Рабочий подкаталог прогона в том виде, в каком его пишет прогонный скрипт.

    audio_link — путь к «исходной записи» вне каталога: тогда audio.wav станет
    символической ссылкой на неё, как при уже готовом 16 кГц моно WAV на входе.
    """
    workdir = os.path.join(parent, name)
    os.makedirs(workdir, exist_ok=True)

    for keeper in keepers:
        _write(os.path.join(workdir, keeper), "содержимое %s\n" % keeper)

    _write(os.path.join(workdir, "plan.json"), '{"schema": "chunk-plan/1"}\n')
    _write(os.path.join(workdir, "transcript.txt"), "дословный текст\n")
    _write(os.path.join(workdir, "transcript.srt"), "1\n00:00:00,000 --> 00:00:01,000\nа\n")
    _write(os.path.join(workdir, "checks.json"), '{"schema": "timecheck/1"}\n')
    _write(os.path.join(workdir, "checks.txt"), "9 из 9\n")
    _write(os.path.join(workdir, "run.log"), "лог прогона\n")
    for number in range(3):
        _write(os.path.join(workdir, "chunks", "chunk_%03d.wav" % number), "wav%d" % number)
    _write(os.path.join(workdir, "ollama", "punctuate-1.prompt.txt"), "промпт\n")
    _write(os.path.join(workdir, "ollama", "punctuate-1.response.txt"), "ответ\n")

    if audio_link is None:
        _write(os.path.join(workdir, "audio.wav"), "RIFF" + "0" * 64)
    else:
        os.symlink(audio_link, os.path.join(workdir, "audio.wav"))
    return workdir


def _names(path: str) -> list[str]:
    """Отсортированный список имён в каталоге — без перехода по ссылкам."""
    return sorted(os.listdir(path))


def _snapshot(path: str) -> list[tuple] | None:
    """Снимок дерева: путь, тип, размер, время правки. None — пути нет.

    Нужен там, где надо доказать «не тронули»: для файлов прямо в data/output,
    для исходной записи за ссылкой и для настоящего data/output целиком. Размеры
    и mtime берутся через lstat, по ссылкам обхода нет.
    """
    if not os.path.lexists(path):
        return None
    items: list[tuple] = []
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs.sort()
        for name in sorted(dirs) + sorted(files):
            full = os.path.join(root, name)
            try:
                info = os.lstat(full)
            except OSError:
                items.append((os.path.relpath(full, path), "нет доступа", 0, 0))
                continue
            kind = "ссылка" if stat.S_ISLNK(info.st_mode) else (
                "каталог" if stat.S_ISDIR(info.st_mode) else "файл"
            )
            items.append((os.path.relpath(full, path), kind, info.st_size, info.st_mtime_ns))
    items.sort()
    return items


def _tempdir(suffix: str = "") -> tempfile.TemporaryDirectory:
    """Временный каталог с проверкой, что он заведомо вне настоящего data/output."""
    holder = tempfile.TemporaryDirectory(prefix="cleanup-runs-selftest-%s" % suffix)
    real = os.path.realpath(holder.name)
    if real == os.path.realpath(DEFAULT_ROOT) or real.startswith(
        os.path.realpath(DEFAULT_ROOT) + os.sep
    ):
        raise AssertionError("временный каталог оказался внутри %s" % DEFAULT_ROOT)
    return holder


def _test_formatting() -> None:
    """Размеры и склонения в отчёте."""
    cases = [
        (0, "0 Б"),
        (1, "1 Б"),
        (1023, "1023 Б"),
        (1024, "1.00 КиБ"),
        (1536, "1.50 КиБ"),
        (1024 * 1024, "1.00 МиБ"),
        (298_548_990, "284.72 МиБ"),
        (1024 ** 3, "1.00 ГиБ"),
        (1024 ** 4, "1.00 ТиБ"),
        (1024 ** 5, "1024.00 ТиБ"),
    ]
    for size, expected in cases:
        got = fmt_bytes(size)
        _check(got == expected, "%d байт -> %r, ожидалось %r" % (size, got, expected))

    for count, expected in (
        (0, "0 файлов"),
        (1, "1 файл"),
        (2, "2 файла"),
        (4, "4 файла"),
        (5, "5 файлов"),
        (11, "11 файлов"),
        (12, "12 файлов"),
        (21, "21 файл"),
        (22, "22 файла"),
        (111, "111 файлов"),
        (470, "470 файлов"),
    ):
        got = plural_files(count)
        _check(got == expected, "%d -> %r, ожидалось %r" % (count, got, expected))


def _test_default_root() -> None:
    """Ветка «без аргументов» указывает на data/output проекта — и только на него."""
    opts = parse_args([])
    _check(opts["run"] is None, "без аргументов --run оказался заполнен: %r" % opts["run"])
    _check(not opts["dry_run"], "без аргументов включился --dry-run")
    _check(not opts["help"] and not opts["selftest"], "без аргументов включились флаги")
    _check(
        DEFAULT_ROOT == os.path.join(REPO_ROOT, "data", "output"),
        "корень прогонов не data/output проекта: %s" % DEFAULT_ROOT,
    )
    _check(os.path.isabs(DEFAULT_ROOT), "корень прогонов не абсолютный путь: %s" % DEFAULT_ROOT)
    # Корень считается от файла инструмента, а не от текущего каталога: иначе
    # вызов из любого другого места чистил бы чужой data/output.
    _check(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) == REPO_ROOT,
        "корень проекта определяется не от расположения файла",
    )


def _test_root_cleanup() -> None:
    """Обычная чистка всего корня: хранители на месте, лишнее удалено."""
    with _tempdir("root") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)

        # Два полноценных прогона, один с неполным набором хранителей и один
        # оборвавшийся — без хранителей вообще.
        full = _make_run(root, "Лекция 1-2026-10-03-210023")
        partial = _make_run(root, "Лекция 2-2026-10-04", keepers=("transcript.json",))
        broken = _make_run(root, "Лекция 3-2026-10-04", keepers=())
        empty = os.path.join(root, "совсем пустой")
        os.makedirs(empty)

        # Итоговые документы лежат прямо в корне: их не трогают никогда.
        docs = [
            _write(os.path.join(root, "Лекция 1-2026-10-03-210023.md"), "# Лекция 1\n"),
            _write(os.path.join(root, "Лекция 2-2026-10-04.md"), "# Лекция 2\n"),
        ]
        docs_before = {path: _snapshot(path) for path in docs}

        code = _capture(lambda: cleanup(root, None, False))[0]
        _check(code == 0, "чистка корня вернула код %d" % code)

        _check(_names(full) == sorted(KEEP_NAMES), "в полном прогоне осталось: %r" % _names(full))
        _check(
            _names(partial) == ["transcript.json"],
            "в неполном прогоне осталось: %r" % _names(partial),
        )
        _check(not os.path.exists(broken), "каталог без хранителей не удалён: %s" % broken)
        _check(not os.path.exists(empty), "пустой каталог не удалён: %s" % empty)

        # Содержимое хранителей не переписано.
        with open(os.path.join(full, "summary.md"), "r", encoding="utf-8") as handle:
            _check(handle.read() == "содержимое summary.md\n", "содержимое хранителя изменилось")

        # Итоговые документы в корне — байт в байт те же, и сам корень на месте.
        for path, before in docs_before.items():
            _check(os.path.isfile(path), "итоговый документ исчез: %s" % path)
            _check(_snapshot(path) == before, "итоговый документ изменился: %s" % path)
        _check(
            _names(root) == sorted([os.path.basename(path) for path in docs] + [
                os.path.basename(full), os.path.basename(partial)
            ]),
            "в корне осталось не то: %r" % _names(root),
        )

        # Повторный прогон по уже чистому корню ничего не ломает и даёт 0.
        code, report, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "повторная чистка вернула код %d" % code)
        _check("удалять нечего" in report, "повторная чистка не сказала, что чистить нечего")
        _check(_names(full) == sorted(KEEP_NAMES), "повторная чистка тронула хранители")


def _test_report() -> None:
    """Отчёт: по каталогу — удалённое и оставленное, в конце общий итог."""
    with _tempdir("report") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)
        full = _make_run(root, "прогон")
        _make_run(root, "оборвался", keepers=())

        code, report, stderr = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "чистка вернула код %d (stderr: %s)" % (code, stderr.strip()))

        for needle in (
            "ОЧИСТКА ПРОГОНОВ",
            "Хранители:   %s" % ", ".join(KEEP_NAMES),
            "удаление, необратимо",
            "Каталог: %s" % full,
            "удалено:",
            "оставлено: %s" % ", ".join(KEEP_NAMES),
            "chunks/ — 3 файла,",
            "ollama/ — 2 файла,",
            "audio.wav —",
            "run.log —",
            "файлов-хранителей нет: каталог удалён целиком",
            "ИТОГ",
            "Обработано каталогов:      2",
            "Удалено:",
            "Удалено каталогов целиком: 1",
        ):
            _check(needle in report, "в отчёте нет %r\n%s" % (needle, report))

        # Отчёт на русском, латиницей только имена файлов.
        _check(re.search(r"[а-яА-Я]", report) is not None, "отчёт не на русском")
        # В обычном режиме нет формулировок будущего времени из --dry-run.
        for forbidden in ("будет удалено", "останется", "было бы удалено", "--dry-run"):
            _check(forbidden not in report, "в обычном отчёте формулировка dry-run: %r" % forbidden)
        _check(not stderr.strip(), "при успешной чистке есть вывод в stderr: %s" % stderr)


def _test_symlink_audio() -> None:
    """audio.wav — ссылка: удаляется ссылка, исходная запись не затрагивается."""
    with _tempdir("symlink") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)

        # «Исходная лекция» лежит вне корня прогонов — как у пользователя.
        source = _write(os.path.join(workdir, "записи", "Лекция 1.wav"), "RIFF" * 100)
        source_dir = os.path.dirname(source)
        before = _snapshot(source_dir)

        workrun = _make_run(root, "прогон со ссылкой", audio_link=source)
        _check(os.path.islink(os.path.join(workrun, "audio.wav")), "audio.wav не ссылка")

        # Ссылка на каталог внутри прогона — тоже не повод уйти за его пределы.
        os.symlink(source_dir, os.path.join(workrun, "записи-ссылка"))

        code, report, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "чистка со ссылками вернула код %d" % code)

        _check(
            not os.path.lexists(os.path.join(workrun, "audio.wav")),
            "ссылка audio.wav осталась в каталоге",
        )
        _check(
            not os.path.lexists(os.path.join(workrun, "записи-ссылка")),
            "ссылка на каталог осталась в каталоге",
        )
        _check(os.path.isfile(source), "исходная запись удалена вместе со ссылкой!")
        _check(_snapshot(source_dir) == before, "каталог с исходной записью изменился")
        with open(source, "r", encoding="utf-8") as handle:
            _check(handle.read() == "RIFF" * 100, "исходная запись изменилась")
        _check("символическая ссылка" in report, "в отчёте не сказано, что это ссылка")
        # Вес исходника в освобождённое место не записан: у ссылки свой размер.
        _check(
            "audio.wav — символическая ссылка, %s" % fmt_bytes(len(source.encode("utf-8")))
            in report,
            "в отчёте не размер самой ссылки:\n%s" % report,
        )

        # То же в --dry-run: ссылка не разыменовывается и остаётся на месте.
        second = _make_run(root, "второй со ссылкой", audio_link=source)
        code, _, _ = _capture(lambda: cleanup(root, None, True))
        _check(code == 0, "--dry-run со ссылкой вернул код %d" % code)
        _check(os.path.islink(os.path.join(second, "audio.wav")), "--dry-run удалил ссылку")
        _check(os.path.isfile(source), "--dry-run тронул исходную запись")


def _test_dry_run() -> None:
    """--dry-run: полный список того, что было бы удалено, и ни одного удаления."""
    with _tempdir("dry") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)
        full = _make_run(root, "прогон")
        broken = _make_run(root, "оборвался", keepers=())
        _write(os.path.join(root, "прогон.md"), "# документ\n")
        before = _snapshot(root)

        code, report, stderr = _capture(lambda: cleanup(root, None, True))
        _check(code == 0, "--dry-run вернул код %d" % code)
        _check(_snapshot(root) == before, "--dry-run изменил что-то на диске")
        _check(os.path.isdir(broken), "--dry-run удалил каталог без хранителей")
        # Девять лишних записей из _make_run плюс все хранители — ни одной меньше.
        _check(
            len(_names(full)) == 9 + len(KEEP_NAMES),
            "--dry-run тронул содержимое: %r" % _names(full),
        )

        # Полный список удаляемого: каждая верхнеуровневая запись названа.
        for name in (
            "audio.wav", "plan.json", "transcript.txt", "transcript.srt",
            "checks.json", "checks.txt", "run.log", "chunks/", "ollama/",
        ):
            _check(name in report, "в списке --dry-run нет %r" % name)
        for needle in (
            "--dry-run, только показать: ничего не удаляется",
            "будет удалено:",
            "останется: %s" % ", ".join(KEEP_NAMES),
            "файлов-хранителей нет: каталог был бы удалён целиком",
            "Было бы удалено:",
            "Это был --dry-run: на диске ничего не изменилось.",
        ):
            _check(needle in report, "в отчёте --dry-run нет %r\n%s" % (needle, report))
        # Прошедшего времени в отчёте --dry-run нет: ни строки каталога
        # «  удалено:», ни строки итога «Удалено:». Сравнение по началу строки —
        # «будет удалено:» содержит «удалено:» как подстроку.
        for forbidden in ("\n  удалено:", "\nУдалено:", "\nУдалено каталогов целиком:"):
            _check(forbidden not in report, "в --dry-run есть строка %r" % forbidden)
        _check(not stderr.strip(), "--dry-run написал в stderr: %s" % stderr)

        # Объём и число файлов те же, что потом удалятся на самом деле.
        planned = re.search(r"Было бы удалено:\s+(\d+) файл\S*, (.+)\n", report)
        _check(planned is not None, "в итоге --dry-run нет числа файлов и объёма")
        code, real, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "чистка после --dry-run вернула код %d" % code)
        done = re.search(r"Удалено:\s+(\d+) файл\S*, (.+)\n", real)
        _check(done is not None, "в итоге чистки нет числа файлов и объёма")
        _check(
            planned.groups() == done.groups(),
            "--dry-run обещал %r, удалено %r" % (planned.groups(), done.groups()),
        )

        # Через main() --dry-run ведёт себя так же и даёт код 0.
        again = _make_run(root, "ещё один")
        snapshot = _snapshot(again)
        code, report, _ = _run(["--run", again, "--dry-run"])
        _check(code == 0, "main с --dry-run вернул код %d" % code)
        _check(_snapshot(again) == snapshot, "main с --dry-run удалил файлы")
        code, report, _ = _run(["--dry-run", "--run=%s" % again])
        _check(code == 0, "--run= со значением вернул код %d" % code)
        _check(_snapshot(again) == snapshot, "--run= со значением удалил файлы")


def _test_run_flag() -> None:
    """--run обрабатывает ровно один указанный каталог, любой."""
    with _tempdir("run") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)
        first = _make_run(root, "первый")
        second = _make_run(root, "второй")
        untouched = _snapshot(second)

        code, report, stderr = _run(["--run", first])
        _check(code == 0, "--run вернул код %d (stderr: %s)" % (code, stderr.strip()))
        _check(
            _names(first) == sorted(KEEP_NAMES),
            "в указанном каталоге осталось: %r" % _names(first),
        )
        _check(_snapshot(second) == untouched, "--run тронул соседний каталог")
        _check("один каталог, указанный через --run" in report, "в шапке не сказано про --run")
        _check("Обработано каталогов:      1" in report, "обработан не один каталог")

        # Каталог вне data/output и без признаков прогона — чистится тоже:
        # проверки «похож ли на прогон» нет сознательно.
        alien = os.path.join(workdir, "чужой каталог")
        _write(os.path.join(alien, "заметки.txt"), "текст\n")
        _write(os.path.join(alien, "summary.md"), "саммари\n")
        code, _, _ = _run(["--run", alien])
        _check(code == 0, "--run на чужом каталоге вернул код %d" % code)
        _check(_names(alien) == ["summary.md"], "в чужом каталоге осталось: %r" % _names(alien))

        # Каталог без хранителей по --run удаляется целиком.
        hopeless = os.path.join(workdir, "без хранителей")
        _write(os.path.join(hopeless, "мусор.bin"), "0" * 10)
        code, _, _ = _run(["--run", hopeless])
        _check(code == 0, "--run на каталоге без хранителей вернул код %d" % code)
        _check(not os.path.exists(hopeless), "каталог без хранителей по --run не удалён")

        # Ссылка на каталог: чистить через неё нельзя, это код 2.
        link = os.path.join(workdir, "ссылка на прогон")
        os.symlink(second, link)
        snapshot = _snapshot(second)
        code, _, stderr = _run(["--run", link])
        _check(code == 2, "--run на ссылке вернул код %d, ожидался 2" % code)
        _check(_snapshot(second) == snapshot, "--run по ссылке почистил каталог за ней")
        _check("ссылк" in stderr, "про ссылку в --run ничего не сказано: %s" % stderr)


def _test_bad_args() -> None:
    """Негодные аргументы и пути: код 2, сообщение в stderr, ни одного удаления."""
    with _tempdir("args") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)
        workrun = _make_run(root, "прогон")
        snapshot = _snapshot(root)

        cases = [
            (["--run", os.path.join(workdir, "нет такого")], "несуществующий путь"),
            (["--run", os.path.join(workrun, "transcript.json")], "файл вместо каталога"),
            (["--run"], "--run без значения"),
            (["--run="], "--run с пустым значением"),
            (["--dry-run=1"], "--dry-run со значением"),
            (["--selftest=1"], "--selftest со значением"),
            (["--кривая-опция"], "неизвестная опция"),
            (["-x"], "короткая неизвестная опция"),
            ([workrun], "путь без флага"),
            (["--run", workrun, "лишнее"], "лишний аргумент"),
        ]
        for argv, what in cases:
            code, _, stderr = _run(argv)
            _check(code == 2, "случай %r завершился кодом %d, ожидался 2" % (what, code))
            _check(
                stderr.startswith(PROG + ":"),
                "случай %r ничего не написал в stderr: %r" % (what, stderr),
            )
            _check(_snapshot(root) == snapshot, "случай %r что-то удалил" % what)


def _test_missing_root() -> None:
    """Корня нет или он пуст — код 0 и внятное «чистить нечего»."""
    with _tempdir("empty") as workdir:
        absent = os.path.join(workdir, "нет каталога")
        code, report, stderr = _capture(lambda: cleanup(absent, None, False))
        _check(code == 0, "отсутствующий корень вернул код %d" % code)
        _check("чистить нечего" in report, "про отсутствующий корень не сказано: %s" % report)
        _check(not os.path.exists(absent), "отсутствующий корень был создан")
        _check(not stderr.strip(), "отсутствующий корень дал вывод в stderr: %s" % stderr)

        root = os.path.join(workdir, "output")
        os.makedirs(root)
        code, report, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "пустой корень вернул код %d" % code)
        _check("чистить нечего" in report, "про пустой корень не сказано: %s" % report)
        _check(os.path.isdir(root), "пустой корень удалён — а он не подкаталог прогона")

        # В корне только итоговые документы: ни один не тронут, код 0.
        doc = _write(os.path.join(root, "Лекция-2026-10-04.md"), "# документ\n")
        before = _snapshot(root)
        code, report, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "корень из одних документов вернул код %d" % code)
        _check(_snapshot(root) == before, "файл прямо в корне изменён или удалён")
        _check(os.path.isfile(doc), "итоговый документ удалён")
        _check("Подкаталогов прогонов нет" in report, "про отсутствие подкаталогов не сказано")

        # Корень оказался файлом — это код 2, а не молчаливое «нечего делать».
        as_file = _write(os.path.join(workdir, "output-файл"), "не каталог\n")
        try:
            with redirect_stdout(io.StringIO()):
                cleanup(as_file, None, False)
        except UsageError:
            pass
        else:
            raise AssertionError("корень-файл не вызвал ошибки")

        # Ссылка на каталог внутри корня пропускается, а не чистится через неё.
        other = os.path.join(workdir, "чужое")
        _write(os.path.join(other, "важное.txt"), "данные\n")
        os.symlink(other, os.path.join(root, "ссылка-на-чужое"))
        snapshot = _snapshot(other)
        code, _, stderr = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "корень со ссылкой вернул код %d" % code)
        _check(_snapshot(other) == snapshot, "чистка ушла по ссылке из корня")
        _check(os.path.islink(os.path.join(root, "ссылка-на-чужое")), "ссылка в корне удалена")
        _check("ссылк" in stderr, "про пропущенную ссылку не предупредили: %s" % stderr)


def _test_keeper_edges() -> None:
    """Хранитель — только обычный файл с таким именем, и ничего другого."""
    with _tempdir("keepers") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)

        # Каталог с именем хранителя — не хранитель: удаляется рекурсивно,
        # и каталог прогона уходит целиком, потому что хранителей в нём нет.
        tricky = os.path.join(root, "каталог-вместо-файла")
        _write(os.path.join(tricky, "summary.md", "внутри.txt"), "вложенное\n")
        _write(os.path.join(tricky, "diarization.json", "внутри.txt"), "вложенное\n")
        code, _, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "каталог с именем хранителя вернул код %d" % code)
        _check(not os.path.exists(tricky), "каталог с именем хранителя остался: %s" % tricky)

        # Ссылка с именем хранителя — тоже не хранитель: по ней не ходят, значит
        # оставлять её незачем. Цель ссылки при этом не затрагивается.
        target = _write(os.path.join(workdir, "настоящий summary.md"), "саммари\n")
        diar = _write(os.path.join(workdir, "настоящий diarization.json"), '{"schema": "x"}\n')
        linked = os.path.join(root, "прогон со ссылкой-хранителем")
        os.makedirs(linked)
        os.symlink(target, os.path.join(linked, "summary.md"))
        os.symlink(diar, os.path.join(linked, "diarization.json"))
        _write(os.path.join(linked, "transcript.json"), "{}\n")
        code, _, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "ссылка с именем хранителя вернула код %d" % code)
        _check(_names(linked) == ["transcript.json"], "осталось: %r" % _names(linked))
        _check(os.path.isfile(target), "цель ссылки-хранителя удалена")
        _check(os.path.isfile(diar), "цель ссылки diarization.json удалена")

        # Каталог, где только хранители и ничего больше, остаётся как был.
        clean = _make_run(root, "уже чистый", keepers=KEEP_NAMES)
        for name in _names(clean):
            if name not in KEEP_NAMES:
                path = os.path.join(clean, name)
                if os.path.isdir(path) and not os.path.islink(path):
                    remove_tree(path, [])
                else:
                    os.unlink(path)
        before = _snapshot(clean)
        code, report, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "чистый каталог вернул код %d" % code)
        _check(_snapshot(clean) == before, "чистый каталог изменён")
        _check("удалять нечего" in report, "про чистый каталог не сказано, что чистить нечего")


def _test_new_keepers() -> None:
    """Диаризация, фразы и роли остаются после чистки, и каждая спасает каталог.

    Отдельной проверкой, а не строчкой в «чистке всего корня»: diarization.json
    стоит десятки минут pyannote на CPU, speakers.json и roles.json без него не
    пересобрать, и потерять их при чистке — значит оплатить всё это второй раз.
    """
    expected = (
        "transcript.json",
        "diarization.json",
        "punctuated.json",
        "summary.md",
        "roles.json",
        "speakers.json",
    )
    _check(
        KEEP_NAMES == expected,
        "список хранителей %r, ожидался %r" % (KEEP_NAMES, expected),
    )

    with _tempdir("new-keepers") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)

        # Полный прогон с диаризацией: все шесть файлов на месте и не переписаны.
        full = _make_run(root, "прогон с диаризацией")
        code, report, stderr = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "чистка прогона с диаризацией вернула код %d" % code)
        _check(not stderr.strip(), "чистка с диаризацией написала в stderr: %s" % stderr)
        for name in ("diarization.json", "speakers.json", "roles.json"):
            path = os.path.join(full, name)
            _check(os.path.isfile(path), "хранитель %s удалён при чистке" % name)
            with open(path, "r", encoding="utf-8") as handle:
                _check(
                    handle.read() == "содержимое %s\n" % name,
                    "содержимое хранителя %s изменилось" % name,
                )
        _check(_names(full) == sorted(KEEP_NAMES), "в прогоне осталось: %r" % _names(full))
        _check(
            "оставлено: %s" % ", ".join(KEEP_NAMES) in report,
            "в отчёте нет всех шести оставленных:\n%s" % report,
        )

        # Каждый из трёх новых хранителей сам по себе спасает каталог: из шести
        # файлов в каталоге есть только он, и «нет хранителей» не срабатывает.
        singles = {
            name: _make_run(root, "только %s" % name, keepers=(name,))
            for name in ("diarization.json", "speakers.json", "roles.json")
        }
        code, report, _ = _capture(lambda: cleanup(root, None, False))
        _check(code == 0, "чистка каталогов с одним хранителем вернула код %d" % code)
        for name, path in singles.items():
            _check(os.path.isdir(path), "каталог, где из шести файлов только %s, удалён" % name)
            _check(
                _names(path) == [name],
                "в каталоге с одним %s осталось: %r" % (name, _names(path)),
            )
            _check(
                "оставлено: %s\n" % name in report,
                "в отчёте нет «оставлено: %s»:\n%s" % (name, report),
            )
        _check(
            "каталог удалён целиком" not in report,
            "каталог с одним хранителем удалён целиком:\n%s" % report,
        )


def _test_failed_removal() -> None:
    """Что-то не удалилось — код 1, каталог не удаляется, причина в отчёте."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        # От root права на запись не ограничивают, и смоделировать отказ нечем.
        sys.stdout.write("         (проверка отказа в доступе пропущена: запуск от root)\n")
        return

    with _tempdir("denied") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)
        workrun = _make_run(root, "прогон без прав", keepers=())
        locked = os.path.join(workrun, "закрытый")
        inside = _write(os.path.join(locked, "файл.bin"), "данные\n")

        os.chmod(locked, 0o500)  # читать можно, удалять внутри — нет
        try:
            code, report, stderr = _capture(lambda: cleanup(root, None, False))
            _check(code == 1, "неудавшееся удаление вернуло код %d, ожидался 1" % code)
            _check(os.path.isfile(inside), "файл без прав всё-таки удалён")
            _check(os.path.isdir(workrun), "каталог удалён, хотя внутри осталось содержимое")
            _check("ОШИБКА:" in report, "в отчёте нет строки об ошибке:\n%s" % report)
            _check("Не удалось удалить:" in report, "в итоге нет числа неудач")
            _check(
                "каталог оставлен" in report,
                "не сказано, почему каталог остался:\n%s" % report,
            )
            _check(PROG in stderr and "не удалось удалить" in stderr, "stderr молчит: %r" % stderr)
            # Остальное из каталога при этом вычищено: одна беда не отменяет работу.
            _check(
                sorted(_names(workrun)) == ["закрытый"],
                "из каталога с отказом удалилось не всё: %r" % _names(workrun),
            )
            # --dry-run на том же каталоге не падает и остаётся кодом 0: он
            # ничего не удаляет, значит и отказать ему не в чем.
            code, _, _ = _capture(lambda: cleanup(root, None, True))
            _check(code == 0, "--dry-run на закрытом каталоге вернул код %d" % code)
        finally:
            os.chmod(locked, 0o700)


def _test_open_file() -> None:
    """run.log, открытый на запись, отвязывается без ошибки — как при вызове из прогона."""
    with _tempdir("open") as workdir:
        root = os.path.join(workdir, "output")
        os.makedirs(root)
        workrun = _make_run(root, "прогон на ходу")
        with open(os.path.join(workrun, "run.log"), "a", encoding="utf-8") as handle:
            handle.write("прогон ещё идёт\n")
            handle.flush()
            code, _, stderr = _capture(lambda: cleanup(root, None, False))
            _check(code == 0, "чистка при открытом run.log вернула код %d" % code)
            _check(not stderr.strip(), "открытый run.log дал ошибку: %s" % stderr)
            # Дескриптор остался рабочим: запись после unlink не падает.
            handle.write("и ещё строка\n")
        _check(_names(workrun) == sorted(KEEP_NAMES), "осталось: %r" % _names(workrun))


def _test_help() -> None:
    """Справка — на русском, с флагами, хранителями и предупреждением."""
    for flag in ("-h", "--help"):
        code, out, stderr = _run([flag])
        _check(code == 0, "%s вернул код %d" % (flag, code))
        _check(not stderr.strip(), "%s написал в stderr: %s" % (flag, stderr))
        for needle in ("--run", "--dry-run", "--selftest", "Использование", "Коды возврата"):
            _check(needle in out, "в справке (%s) нет %r" % (flag, needle))
        for keeper in KEEP_NAMES:
            _check(keeper in out, "в справке (%s) нет файла-хранителя %r" % (flag, keeper))
        _check("необратимо" in out, "в справке нет предупреждения о необратимости")
        _check(DEFAULT_ROOT in out, "в справке нет каталога прогонов по умолчанию")
        _check(re.search(r"[а-яА-Я]", out) is not None, "справка не на русском")


def _capture(action) -> tuple[int, str, str]:
    """Прогон функции с перехватом stdout и stderr: (код, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = action()
    return code, stdout.getvalue(), stderr.getvalue()


def selftest() -> int:
    """Прогоняет встроенные проверки. Код возврата 0 — все прошли.

    Перед проверками снимается слепок настоящего data/output, после — сверяется:
    так видно, что ни одна проверка туда не заглянула. Сами проверки работают
    только в каталогах из tempfile, а main() без аргументов не вызывается нигде.
    """
    tests = (
        ("форматирование чисел в отчёте", _test_formatting),
        ("каталог прогонов по умолчанию", _test_default_root),
        ("чистка всего корня", _test_root_cleanup),
        ("отчёт по каталогам и итог", _test_report),
        ("audio.wav как символическая ссылка", _test_symlink_audio),
        ("--dry-run ничего не удаляет", _test_dry_run),
        ("--run на одном каталоге", _test_run_flag),
        ("негодные аргументы и пути", _test_bad_args),
        ("корня нет, корень пуст, одни документы", _test_missing_root),
        ("пограничные случаи с хранителями", _test_keeper_edges),
        ("диаризация, фразы и роли остаются", _test_new_keepers),
        ("отказ в доступе при удалении", _test_failed_removal),
        ("открытый на запись run.log", _test_open_file),
        ("справка", _test_help),
    )

    real_before = _snapshot(DEFAULT_ROOT)

    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ cleanup_runs.py ================\n")
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

    # Главная проверка безопасности: настоящий каталог прогонов не тронут.
    name = "настоящий data/output не затронут"
    real_after = _snapshot(DEFAULT_ROOT)
    if real_after == real_before:
        sys.stdout.write(
            "[ок]     %s (%s)\n"
            % (name, "его нет" if real_before is None else "записей: %d" % len(real_before))
        )
    else:
        failed += 1
        sys.stdout.write("[ПРОВАЛ] %s: содержимое %s изменилось\n" % (name, DEFAULT_ROOT))

    total = len(tests) + 1
    passed = total - failed
    sys.stdout.write("\nПройдено %d из %d\n" % (passed, total))
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
        return cleanup(DEFAULT_ROOT, opts["run"], opts["dry_run"])
    except UsageError as exc:
        return die(str(exc))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
