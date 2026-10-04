#!/usr/bin/env python3
"""diarize.py — диаризация pyannote по локальному кэшу Hugging Face.

На вход — подготовленный 16 кГц моно WAV (его делает scripts/lib/common.sh,
функция to_wav). На выход — один JSON схемы diarization/1: кто и когда говорил,
то есть список интервалов речи с именем говорящего.

Говорящие нумеруются по времени первого появления: «Говорящий 1» начал говорить
раньше «Говорящего 2», и так далее. Это не номер метки pyannote — та нумерует
кластеры в своём порядке, и SPEAKER_03 вполне может заговорить первым. Исходная
метка никуда не девается, она лежит в speakers[].label. Ролей здесь нет совсем:
«Говорящий 1» — это голос, а не преподаватель и не студент. Роли по тексту
расставляет другой инструмент.

Перекрывающиеся интервалы сохраняются как есть. Если двое говорили одновременно,
в turns будут два интервала с пересекающимися границами, и склеивать их нельзя:
именно по ним дальше видно, где была перебивка, а где чистая реплика.

Веса не скачиваются никогда. Перед обращением к pyannote инструмент проверяет,
что в локальном кэше Hugging Face лежит снапшот репозитория модели, и без него
падает с кодом 2, назвав полный путь и repo-id — ровно как tools/transcribe.py
ведёт себя без весов GigaAM. Сверх проверки перед импортом pyannote выставляется
HF_HUB_OFFLINE=1, так что любая попытка библиотеки что-нибудь дотянуть
превращается в понятную ошибку, а не в молчаливую закачку гигабайтов. Там же
гасится телеметрия pyannote (PYANNOTE_METRICS_ENABLED=false): в версии 4.x она
включена по умолчанию и отправляет факт запуска на otel.pyannote.ai, а инструмент
обещает не ходить в сеть вообще.

Звук читается модулем wave из стандартной библиотеки, а не torchaudio и не
ffmpeg. Сначала заголовок — для длительности и ранней проверки «звук вообще
читается», которая отрабатывает до загрузки модели (та на CPU стоит десятки
секунд). Потом, уже перед прогоном, та же распаковка отдаёт pyannote готовый
сигнал в памяти вместо пути к файлу. Так сделано не из любви к тензорам: звук по
пути декодирует torchcodec, а он на этой машине не загружается без библиотек
ffmpeg@8, и сам pyannote в своём предупреждении советует ровно этот обход. Вход
по договорённости уже приведён к 16 кГц моно pcm_s16le, так что декодер не нужен
вовсе; файл другого формата уходит в pyannote путём, как раньше.

Прогон на CPU идёт примерно в реальном времени: на лекции 02:35:29 это десятки
минут. Поэтому pyannote получает hook, и каждый шаг с его прогрессом печатается
в stdout — иначе прогон выглядит зависшим.

Зависимости: pyannote.audio, torch и torchaudio, но они импортируются лениво,
внутри функции прогона. --help и --selftest работают без них, без весов, без
кэша, без сети и без ffmpeg. Проверки живут в самом файле (--selftest): pytest в
окружении проекта нет.

Использование:
  python tools/diarize.py --input <файл.wav> --out <файл.json> [опции]
  python tools/diarize.py --selftest
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import sys
import tempfile
import time
import wave
from contextlib import redirect_stderr, redirect_stdout

PROG = "diarize.py"

SCHEMA = "diarization/1"

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_DEVICE = "cpu"

# Кэш Hugging Face: $HF_HOME, иначе ~/.cache/huggingface. Внутри него библиотека
# huggingface_hub держит скачанные репозитории в подкаталоге hub/ — туда и
# смотрит преflight, туда же уходит cache_dir при загрузке пайплайна.
FALLBACK_CACHE_DIR = os.path.join("~", ".cache", "huggingface")
HUB_SUBDIR = "hub"

# Файл, по которому pyannote узнаёт пайплайн внутри снапшота репозитория.
# См. AssetFileName.Pipeline в pyannote/audio/utils/hf_hub.py.
PIPELINE_FILE = "config.yaml"

# Имя говорящего. Номер — порядок первого появления, а не номер метки pyannote.
SPEAKER_NAME = "Говорящий %d"

# Допуск на сравнение секунд: границы приходят из float-арифметики pyannote.
EPS = 1e-6

# Человеческие названия шагов пайплайна — их pyannote передаёт в hook.
STEP_NAMES = {
    "segmentation": "сегментация",
    "speaker_counting": "подсчёт говорящих",
    "embeddings": "векторы голосов",
    "clustering": "кластеризация голосов",
    "discrete_diarization": "сборка интервалов",
}

HELP = """Использование:
  python tools/diarize.py --input <файл.wav> --out <файл.json> [опции]
  python tools/diarize.py --selftest

Определяет по записи, кто и когда говорил, и пишет один JSON схемы %(schema)s:
интервалы речи с именами «Говорящий 1», «Говорящий 2», … Номер — это порядок
первого появления голоса, а не номер метки pyannote; исходная метка (SPEAKER_00
и подобные) сохраняется в speakers[].label.

Ролей здесь нет: инструмент слышит голоса, но не знает, кто преподаватель, а кто
студент. Роли по тексту расставляет другой инструмент.

Перекрывающиеся интервалы сохраняются как есть и НЕ склеиваются. Если двое
говорили одновременно, в turns будет два интервала с пересекающимися границами —
так видно перебивки. Суммарная длительность такой одновременной речи лежит в
stats.overlap_seconds.

Вход — подготовленный 16 кГц моно WAV (pcm_s16le), такой делает
scripts/lib/common.sh. Инструмент не вызывает ffmpeg и ничего не конвертирует.

Веса не скачиваются никогда. Перед загрузкой модели проверяется, что снапшот
репозитория уже лежит в локальном кэше Hugging Face; если его нет — код 2 с
полным путём и repo-id. Перед импортом pyannote выставляется HF_HUB_OFFLINE=1,
так что до сети дело не доходит ни при каком раскладе. Кэш готовится руками,
см. сообщение об ошибке — там написано, что именно сделать.

Опции:
  --input <файл>     WAV со звуком записи (обязателен для прогона)
  --out <файл>       куда писать JSON схемы %(schema)s (обязателен для прогона)
  --model <repo-id>  репозиторий модели на huggingface.co
                     (по умолчанию %(model)s)
  --cache-dir <кат>  каталог кэша Hugging Face: внутри него ищется
                     hub/models--<org>--<имя>/snapshots/<снапшот>/%(pipeline)s
                     (по умолчанию $HF_HOME, иначе %(cache)s)
  --device <устр>    устройство для torch: cpu, mps, cuda
                     (по умолчанию %(device)s)
  --min-speakers <N> нижняя граница числа говорящих, если она известна
  --max-speakers <N> верхняя граница числа говорящих, если она известна
  --selftest         прогнать встроенные проверки и выйти (без pyannote, torch,
                     весов, кэша и сети)
  -h, --help         показать эту справку

Коды возврата:
  0  файл записан, найден хотя бы один говорящий
  1  файл записан, но речи не нашлось: turns пуст, говорящих ноль
  2  вход непригоден, нет пакета pyannote.audio, нет кэша модели, pyannote упал
     или звук не читается — файл --out при этом не создаётся, а существующий
     не перетирается

Примеры:
  python tools/diarize.py --input data/output/прогон/audio.wav \\
      --out data/output/прогон/diarization.json
  python tools/diarize.py --input audio.wav --out diar.json --min-speakers 2
  python tools/diarize.py --selftest
""" % {
    "schema": SCHEMA,
    "model": DEFAULT_MODEL,
    "cache": FALLBACK_CACHE_DIR,
    "device": DEFAULT_DEVICE,
    "pipeline": PIPELINE_FILE,
}


class ToolError(Exception):
    """Ошибка, о которой говорят человеку, а не трейсбеком. Это всегда код 2."""


def die(message: str, code: int = 2) -> int:
    """Печатает сообщение в stderr и отдаёт код возврата для main()."""
    sys.stderr.write("%s: %s\n" % (PROG, message))
    return code


def warn(message: str) -> None:
    """Предупреждение: работа продолжается, но человек должен знать."""
    sys.stderr.write("%s: предупреждение: %s\n" % (PROG, message))


# ------------------------------------------------------------ форматирование ---


def format_hms(seconds: float) -> str:
    """Секунды в ЧЧ:ММ:СС.мсс — для отчёта и сообщений об ошибках."""
    total = max(0.0, float(seconds))
    hours, rest = divmod(total, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return "%02d:%02d:%06.3f" % (int(hours), int(minutes), secs)


def r3(value: float) -> float:
    """Округление до миллисекунд: в JSON проекта числа идут с тремя знаками."""
    return round(float(value), 3)


def plural(count: int, one: str, few: str, many: str) -> str:
    """«1 реплика», «2 реплики», «5 реплик» — отчёт читает человек."""
    tail = count % 100
    if 11 <= tail <= 14:
        word = many
    elif count % 10 == 1:
        word = one
    elif 2 <= count % 10 <= 4:
        word = few
    else:
        word = many
    return "%d %s" % (count, word)


# --------------------------------------------------------- кэш и зависимости ---


def default_cache_dir() -> str:
    """Каталог кэша Hugging Face по умолчанию: $HF_HOME, иначе ~/.cache/huggingface.

    Именно HF_HOME, а не HF_HUB_CACHE: владелец готовит кэш руками и думает о
    нём как о «каталоге хаггингфейса», а подкаталог hub/ внутри — деталь
    huggingface_hub, которую инструмент дописывает сам.
    """
    value = os.environ.get("HF_HOME")
    if value and value.strip():
        return os.path.expanduser(value.strip())
    return os.path.expanduser(FALLBACK_CACHE_DIR)


def repo_dir_name(repo_id: str) -> str:
    """«pyannote/speaker-diarization-community-1» -> «models--pyannote--speaker-...».

    Так huggingface_hub называет каталоги репозиториев внутри hub/, и преflight
    обязан смотреть ровно туда, куда потом пойдёт сама библиотека.
    """
    return "models--" + repo_id.strip("/").replace("/", "--")


def hub_dir(cache_dir: str) -> str:
    """Подкаталог кэша, в котором huggingface_hub держит репозитории."""
    return os.path.join(cache_dir, HUB_SUBDIR)


def repo_cache_dir(cache_dir: str, repo_id: str) -> str:
    """Полный путь к каталогу репозитория внутри кэша."""
    return os.path.join(hub_dir(cache_dir), repo_dir_name(repo_id))


def snapshots_dir(cache_dir: str, repo_id: str) -> str:
    """Полный путь к snapshots/ репозитория — там лежат версии по хэшу ревизии."""
    return os.path.join(repo_cache_dir(cache_dir, repo_id), "snapshots")


def _how_to_prepare_cache(cache_dir: str, repo_id: str) -> str:
    """Общий хвост всех сообщений про кэш: что именно сделать руками.

    Это читает владелец, у которого кэша нет вовсе, — значит в тексте должно
    быть всё: какой репозиторий принять условиями, чем качать, куда положить и
    почему инструмент не сделает этого сам.
    """
    return (
        "Инструмент ничего не скачивает: ни конфигурации пайплайна, ни весов. Перед\n"
        "импортом pyannote выставляется HF_HUB_OFFLINE=1, поэтому в сеть он не пойдёт\n"
        "даже случайно. Кэш готовится руками, один раз:\n"
        "\n"
        "  1. примите условия доступа к репозиторию — он gated, без этого отдадут 403:\n"
        "     https://hf.co/%(repo)s\n"
        "     (если пайплайн тянет модели из соседних репозиториев, условия надо\n"
        "     принять и у них — их имена назовёт сама закачка)\n"
        "  2. создайте токен чтения: https://hf.co/settings/tokens\n"
        "  3. скачайте репозиторий в кэш — токен нужен только на этом шаге, самому\n"
        "     инструменту он не нужен никогда:\n"
        "\n"
        "     HF_HOME=%(cache)s HF_TOKEN=hf_... \\\n"
        "         python -c \"from huggingface_hub import snapshot_download; \\\n"
        "snapshot_download('%(repo)s')\"\n"
        "\n"
        "Другой каталог кэша указывается флагом --cache-dir, другая модель — --model."
        % {"repo": repo_id, "cache": cache_dir}
    )


def check_cache(cache_dir: str, repo_id: str) -> str:
    """Преflight кэша. Возвращает путь к годному снапшоту либо бросает ToolError.

    Годный снапшот — это подкаталог snapshots/, в котором лежит config.yaml
    пайплайна. Проверять именно его, а не просто наличие каталога, важно:
    оборвавшаяся закачка оставляет каталог снапшота с symlink'ами на несуществующие
    блобы, и from_pretrained на таком кэше полез бы в сеть за недостающим файлом.
    """
    expected = os.path.join(
        snapshots_dir(cache_dir, repo_id), "<снапшот>", PIPELINE_FILE
    )
    head = (
        "нет локального кэша модели pyannote\n"
        "\n"
        "  репозиторий:    %s\n"
        "  каталог кэша:   %s\n"
        "  ожидался файл:  %s\n"
        "\n" % (repo_id, cache_dir, expected)
    )
    tail = "\n\n" + _how_to_prepare_cache(cache_dir, repo_id)

    if not os.path.isdir(cache_dir):
        raise ToolError(head + "  причина:        каталога кэша нет на диске" + tail)

    repo_path = repo_cache_dir(cache_dir, repo_id)
    if not os.path.isdir(repo_path):
        raise ToolError(
            head
            + "  причина:        в кэше нет репозитория %s" % repo_path
            + tail
        )

    snapshots = snapshots_dir(cache_dir, repo_id)
    if not os.path.isdir(snapshots):
        raise ToolError(
            head + "  причина:        в репозитории нет каталога %s" % snapshots + tail
        )

    try:
        names = sorted(os.listdir(snapshots))
    except OSError as exc:
        raise ToolError(
            head
            + "  причина:        не удалось прочитать %s (%s)"
            % (snapshots, exc.strerror or exc)
            + tail
        ) from None

    found = []
    for name in names:
        candidate = os.path.join(snapshots, name)
        if not os.path.isdir(candidate):
            continue
        found.append(candidate)
        # is_file() идёт по символической ссылке: так отсеивается снапшот, в
        # котором ссылка на блоб осталась, а сам блоб не докачался.
        if os.path.isfile(os.path.join(candidate, PIPELINE_FILE)):
            return candidate

    if not found:
        raise ToolError(
            head + "  причина:        в %s нет ни одного снапшота" % snapshots + tail
        )
    raise ToolError(
        head
        + "  причина:        снапшот есть, но в нём нет %s — закачка была неполной:\n"
        "                  %s" % (PIPELINE_FILE, ", ".join(found))
        + tail
    )


def check_package(module: str = "pyannote.audio") -> None:
    """Преflight пакета: без pyannote.audio прогон не состоится, и это код 2.

    find_spec, а не import: он находит пакет, не выполняя его код, то есть не
    тянет за собой torch на секунды и гигабайты. Проверка стоит до чтения WAV и
    до загрузки модели, чтобы «пакета нет» сообщалось сразу.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        return
    raise ToolError(
        "нет пакета %(module)s, а ставить что-либо инструмент не умеет\n"
        "\n"
        "Поставьте его в окружение проекта (.venv), а не глобально:\n"
        "\n"
        "  python -m pip install \"pyannote.audio>=4\"\n"
        "\n"
        "Вместе с ним приедут torch и torchaudio — они тоже нужны для прогона.\n"
        "Для --help и --selftest ничего из этого не требуется." % {"module": module}
    )


def prepare_env(cache_dir: str) -> dict:
    """Готовит окружение перед импортом pyannote. Возвращает выставленные значения.

    Всё это обязано случиться ДО импорта: huggingface_hub читает свои переменные
    один раз, при загрузке модуля, и выставлять их потом бессмысленно.

      HF_HUB_OFFLINE=1          любая попытка сходить в сеть превращается в
                                исключение, а не в закачку весов;
      HF_HOME, HF_HUB_CACHE     кэш берётся ровно тот, который проверил преflight,
                                даже если в окружении прописан другой;
      PYANNOTE_METRICS_ENABLED  в pyannote.audio 4.x телеметрия включена по
      =false                    умолчанию и шлёт факт запуска на otel.pyannote.ai
                                (см. pyannote/audio/telemetry/config.yaml).
                                Инструмент обещает не ходить в сеть — значит и
                                этого быть не должно.
    """
    values = {
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": cache_dir,
        "HF_HUB_CACHE": hub_dir(cache_dir),
        "PYANNOTE_METRICS_ENABLED": "false",
    }
    os.environ.update(values)
    return values


def package_version(name: str) -> str:
    """Версия установленного пакета без его импорта. Нет пакета — «неизвестно»."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 — версия нужна для отчёта, падать из-за неё нельзя
        return "неизвестно"


# ------------------------------------------------------------------- вход ---


def probe_wav(path: str) -> tuple[float, int, int]:
    """Длительность, частота и число каналов WAV. Возвращает (секунды, Гц, каналы).

    Читается только заголовок, модулем wave из стандартной библиотеки: ни
    torchaudio, ни ffmpeg здесь не нужны, а вход по договорённости уже приведён к
    16 кГц моно pcm_s16le. Непрочитанный заголовок — это код 2 до загрузки
    модели, которая на CPU стоит десятки секунд.
    """
    if not os.path.exists(path):
        raise ToolError("файл со звуком не найден: %s" % path)
    if os.path.isdir(path):
        raise ToolError("--input указывает на каталог, а не на файл: %s" % path)

    try:
        with contextlib.closing(wave.open(path, "rb")) as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (wave.Error, EOFError) as exc:
        raise ToolError(
            "не удалось прочитать WAV %s (%s)\n"
            "Ожидается подготовленный файл: 16 кГц, моно, pcm_s16le. Такой делает\n"
            "scripts/lib/common.sh (to_wav); сам инструмент ffmpeg не вызывает и\n"
            "ничего не конвертирует." % (path, exc)
        ) from None
    except OSError as exc:
        raise ToolError(
            "не удалось открыть %s (%s)" % (path, exc.strerror or exc)
        ) from None

    if rate <= 0:
        raise ToolError("в заголовке %s указана частота %d Гц" % (path, rate))
    duration = frames / float(rate)
    if duration <= 0:
        raise ToolError("в файле %s нет звука: длительность 0 с" % path)
    return duration, rate, channels


# -------------------------------------------------------- разбор результата ---


def clean_turns(raw, duration: float) -> tuple[list, int]:
    """Чистит сырые интервалы pyannote. Возвращает (отсортированные, отброшенные).

    На входе — последовательность (start, end, label), на выходе — список
    кортежей (start, end, label) и число отброшенных интервалов.

    Что здесь происходит и почему:

    * границы зажимаются в [0, source_duration] и округляются до миллисекунд.
      pyannote считает по кадрам и легко отдаёт end на доли миллисекунды дальше
      конца записи, а схема требует end <= source_duration;
    * интервал, у которого после округления start == end, отбрасывается: нулевая
      реплика ничего не значит, а схема требует start < end;
    * сортировка по (start, end) — порядок в файле обязан быть временным;
    * пересечения сохраняются как есть. Два человека, заговорившие одновременно,
      дают два пересекающихся интервала, и склеивать их нельзя.
    """
    limit = r3(duration)
    items = []
    dropped = 0
    for start, end, label in raw:
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if start != start or end != end:  # NaN
            dropped += 1
            continue
        start = r3(min(max(start, 0.0), limit))
        end = r3(min(max(end, 0.0), limit))
        if end - start <= EPS:
            dropped += 1
            continue
        items.append((start, end, str(label)))

    items.sort(key=lambda item: (item[0], item[1], item[2]))
    return items, dropped


def name_speakers(items) -> dict:
    """Соответствие «метка pyannote -> Говорящий N» по времени первого появления.

    Номер в имени — это порядок, в котором голос впервые зазвучал, а не номер
    метки: pyannote нумерует кластеры по-своему, и SPEAKER_03 вполне может
    заговорить первым. Считается по уже очищенному и отсортированному списку,
    иначе отброшенный нулевой интервал сдвинул бы нумерацию.
    """
    names: dict = {}
    for _start, _end, label in items:
        if label not in names:
            names[label] = SPEAKER_NAME % (len(names) + 1)
    return names


def normalize_turns(raw, duration: float) -> tuple[list, int]:
    """Сырые интервалы pyannote -> реплики схемы: {index, start, end, speaker}."""
    items, dropped = clean_turns(raw, duration)
    names = name_speakers(items)
    turns = [
        {"index": position, "start": start, "end": end, "speaker": names[label]}
        for position, (start, end, label) in enumerate(items)
    ]
    return turns, dropped


def merge_intervals(intervals) -> list:
    """Объединение пересекающихся и соприкасающихся интервалов. Для длительностей."""
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    merged: list = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + EPS:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def union_seconds(intervals) -> float:
    """Сколько всего времени покрыто интервалами — пересечения считаются один раз."""
    return sum(end - start for start, end in merge_intervals(intervals))


def overlap_seconds(intervals) -> float:
    """Сколько времени говорили двое и больше одновременно.

    Считается не суммой пересечений попарно (там тройное наложение посчиталось бы
    трижды), а разметкой оси: событиями +1 на начало и -1 на конец, и в зачёт
    идут отрезки, где одновременно активны хотя бы двое.
    """
    events = []
    for start, end in intervals:
        events.append((float(start), 1))
        events.append((float(end), -1))
    events.sort()

    total = 0.0
    layers = 0
    previous = None
    for moment, delta in events:
        if layers >= 2 and previous is not None:
            total += moment - previous
        layers += delta
        previous = moment
    return total


def summarize_speakers(turns, raw_labels: dict) -> list:
    """Сводка по говорящим: сколько реплик, сколько речи и какая доля.

    speech_seconds — объединение интервалов говорящего, а не их сумма: у одного
    голоса интервалы pyannote не пересекаются, но на случай странного вывода
    честнее считать покрытое время.

    share — доля от суммарной речи всех говорящих (сумма долей равна 1). Не от
    длительности записи: тогда доли не складывались бы в целое ни при паузах, ни
    при перебивках, и строка «55%» ничего бы не значила.
    """
    by_name: dict = {}
    for turn in turns:
        by_name.setdefault(turn["speaker"], []).append((turn["start"], turn["end"]))

    seconds = {name: union_seconds(items) for name, items in by_name.items()}
    total = sum(seconds.values())

    speakers = []
    for name in by_name:
        speech = seconds[name]
        speakers.append(
            {
                "name": name,
                "label": raw_labels.get(name, ""),
                "turns": len(by_name[name]),
                "speech_seconds": r3(speech),
                "share": r3(speech / total) if total > EPS else 0.0,
            }
        )
    # Порядок — по номеру в имени, то есть по времени первого появления.
    speakers.sort(key=lambda item: _speaker_number(item["name"]))
    return speakers


def _speaker_number(name: str) -> int:
    """Номер из «Говорящий 7». Имя всегда своё, но на чужом не падаем."""
    tail = name.rsplit(" ", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def labels_by_name(raw, duration: float) -> dict:
    """Соответствие «Говорящий N» -> исходная метка pyannote.

    Обратная сторона name_speakers: в turns лежит имя, а в speakers[].label
    должна остаться метка, которую дала модель.
    """
    items, _dropped = clean_turns(raw, duration)
    return {name: label for label, name in name_speakers(items).items()}


# ----------------------------------------------------------- сборка файла ---


def build_payload(
    *,
    source: str,
    duration: float,
    model: str,
    device: str,
    turns: list,
    speakers: list,
    elapsed_seconds: float,
    model_load_seconds: float,
    min_speakers=None,
    max_speakers=None,
) -> dict:
    """Собирает объект схемы diarization/1. Поля — ровно те, что в схеме."""
    intervals = [(turn["start"], turn["end"]) for turn in turns]
    stats = {
        "turns": len(turns),
        "speakers": len(speakers),
        "speech_seconds": r3(union_seconds(intervals)),
        "overlap_seconds": r3(overlap_seconds(intervals)),
        "elapsed_seconds": r3(elapsed_seconds),
        "model_load_seconds": r3(model_load_seconds),
        "device": device,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "pyannote_audio_version": package_version("pyannote.audio"),
        "torch_version": package_version("torch"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    return {
        "schema": SCHEMA,
        "source": source,
        "source_duration": r3(duration),
        "model": model,
        "device": device,
        "speakers": speakers,
        "turns": turns,
        "stats": stats,
    }


def write_payload(path: str, payload: dict) -> None:
    """Пишет JSON: UTF-8 без эскейпов, отступ 2, перевод строки в конце."""
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise ToolError(
            "не удалось записать файл %s (%s)" % (path, exc.strerror or exc)
        ) from None


def print_report(path: str, payload: dict, dropped: int) -> None:
    """Отчёт о прогоне в stdout — он же попадает в run.log прогона."""
    stats = payload["stats"]
    duration = payload["source_duration"]
    speech = stats["speech_seconds"]
    share = (speech / duration) if duration > EPS else 0.0

    sys.stdout.write("\n================ ДИАРИЗАЦИЯ ================\n")
    sys.stdout.write("Говорящих:        %d\n" % stats["speakers"])
    sys.stdout.write("Реплик:           %s\n" % plural(stats["turns"], "реплика", "реплики", "реплик"))
    if dropped:
        sys.stdout.write("Отброшено:        %d интервалов нулевой длины\n" % dropped)
    sys.stdout.write(
        "Речь:             %s из %s (%.1f%%)\n"
        % (format_hms(speech), format_hms(duration), 100.0 * share)
    )
    sys.stdout.write("Одновременно:     %s\n" % format_hms(stats["overlap_seconds"]))
    for speaker in payload["speakers"]:
        sys.stdout.write(
            "  %-12s (%s): %s, %s, %.1f%%\n"
            % (
                speaker["name"],
                speaker["label"] or "без метки",
                plural(speaker["turns"], "реплика", "реплики", "реплик"),
                format_hms(speaker["speech_seconds"]),
                100.0 * speaker["share"],
            )
        )
    sys.stdout.write("Загрузка модели:  %.2f с\n" % stats["model_load_seconds"])
    sys.stdout.write("Время прогона:    %.2f с\n" % stats["elapsed_seconds"])
    sys.stdout.write("Файл:             %s\n" % path)


# ------------------------------------------------------------- сам прогон ---


class ProgressHook:
    """Печатает прогресс пайплайна: иначе десятки минут выглядят зависанием.

    pyannote зовёт hook как hook(шаг, артефакт, file=..., completed=..., total=...)
    и на длинных шагах делает это много раз подряд. Печатать каждый вызов нельзя
    (run.log распухнет на тысячи строк), поэтому внутри шага выводится каждые
    10% и обязательно последняя строка.
    """

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.step = None
        self.bucket = -1

    def __call__(self, step_name, step_artefact=None, file=None, total=None,
                 completed=None, **kwargs):
        if step_name != self.step:
            self.step = step_name
            self.bucket = -1
            self.stream.write("  шаг: %s\n" % STEP_NAMES.get(step_name, step_name))
            self.stream.flush()
        if not total or completed is None:
            return
        percent = int(100.0 * completed / total)
        bucket = percent // 10
        if bucket > self.bucket or completed >= total:
            self.bucket = bucket
            self.stream.write("    %3d%% (%d из %d)\n" % (percent, completed, total))
            self.stream.flush()


def load_audio(path: str):
    """Читает подготовленный WAV в память и отдаёт его pyannote словарём.

    Зачем вообще: звук в pyannote декодирует torchcodec, а тот на этой машине не
    собирается — ему нужны библиотеки ffmpeg@8, которых рядом нет
    (libtorchcodec не находит libavutil). Сам pyannote в этом случае прямым
    текстом советует подать waveform в памяти, и это ровно наш случай: вход уже
    приведён к 16 кГц моно pcm_s16le, распаковывается модулем wave из
    стандартной библиотеки, и никакой декодер не нужен.

    Цена — запись целиком в оперативной памяти: 2,5 часа моно в float32 это
    примерно 0,6 ГБ плюс столько же на время преобразования. Для прогона, где
    сама модель занимает больше, это приемлемо.

    Возвращает None, если ширина сэмпла не 2 байта: такой файл не пойми какой,
    и честнее отдать pyannote путь и дать ему решать самому.
    """
    import numpy
    import torch

    with contextlib.closing(wave.open(path, "rb")) as handle:
        if handle.getsampwidth() != 2:
            return None
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    samples = numpy.frombuffer(raw, dtype="<i2")
    # Хвост в полкадра бывает у оборванной записи: лишние сэмплы отбрасываются,
    # иначе reshape упадёт на ровном месте.
    usable = (len(samples) // channels) * channels
    block = numpy.ascontiguousarray(samples[:usable].reshape(-1, channels).T)
    waveform = torch.from_numpy(block.astype(numpy.float32) / 32768.0)
    return {
        "waveform": waveform,
        "sample_rate": rate,
        "uri": os.path.splitext(os.path.basename(path))[0],
    }


def diarize_with_pyannote(
    source: str,
    *,
    model: str,
    cache_dir: str,
    device: str,
    min_speakers=None,
    max_speakers=None,
) -> tuple[list, float]:
    """Прогон pyannote. Возвращает (список (start, end, метка), время загрузки).

    Тяжёлые импорты живут здесь и только здесь: --help и --selftest обязаны
    работать без pyannote, torch и torchaudio.
    """
    prepare_env(cache_dir)

    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:  # noqa: BLE001 — сломанная установка это тоже код 2
        raise ToolError(
            "не удалось импортировать pyannote.audio (%s: %s)"
            % (type(exc).__name__, exc)
        ) from None

    sys.stdout.write("Загружаю модель из кэша (это десятки секунд)…\n")
    sys.stdout.flush()
    started = time.perf_counter()
    try:
        # cache_dir здесь — это hub/: huggingface_hub понимает под cache_dir
        # именно его, а не корень HF_HOME.
        pipeline = Pipeline.from_pretrained(model, cache_dir=hub_dir(cache_dir))
    except Exception as exc:  # noqa: BLE001 — любая беда загрузки это код 2
        raise ToolError(_load_failure(model, cache_dir, exc)) from None
    if pipeline is None:
        raise ToolError(
            "pyannote не смог собрать пайплайн %s из кэша %s — скорее всего, в\n"
            "снапшоте нет %s или он неполный" % (model, cache_dir, PIPELINE_FILE)
        )

    try:
        pipeline.to(torch.device(device))
    except Exception as exc:  # noqa: BLE001 — кривое устройство это код 2, а не трейсбек
        raise ToolError(
            "не удалось перенести модель на устройство %r (%s: %s)"
            % (device, type(exc).__name__, exc)
        ) from None
    model_load_seconds = time.perf_counter() - started
    sys.stdout.write("Модель загружена за %.2f с\n" % model_load_seconds)
    sys.stdout.flush()

    kwargs: dict = {"hook": ProgressHook()}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        audio = load_audio(source)
    except Exception as exc:  # noqa: BLE001 — не распаковался WAV, но путь ещё остался
        warn(
            "не удалось распаковать %s в память (%s: %s) — звук пойдёт в pyannote "
            "файлом" % (source, type(exc).__name__, exc)
        )
        audio = None

    try:
        output = pipeline(audio if audio is not None else source, **kwargs)
    except Exception as exc:  # noqa: BLE001 — падение модели это код 2
        raise ToolError(_apply_failure(source, exc)) from None

    return list(extract_turns(output)), model_load_seconds


def _apply_failure(source: str, exc: Exception) -> str:
    """Сообщение о падении прогона. Про сломанный torchcodec говорится отдельно.

    Если звук не удалось подать waveform'ом, pyannote полезет декодировать файл
    через torchcodec — а он без библиотек ffmpeg@8 не загружается, и ошибка
    выглядит как «Could not load libtorchcodec». Человеку надо сказать, что это
    не про модель и не про запись.
    """
    text = str(exc)
    message = "pyannote упал на прогоне %s (%s: %s)" % (source, type(exc).__name__, text)
    if "torchcodec" in text or "libavutil" in text:
        message += (
            "\n"
            "\n"
            "Это не про модель и не про запись: pyannote не смог декодировать звук,\n"
            "потому что torchcodec не находит библиотеки ffmpeg. Обычно инструмент\n"
            "обходит это сам — распаковывает подготовленный pcm_s16le модулем wave и\n"
            "подаёт готовый сигнал в память. Раз дело дошло до декодера, на входе\n"
            "что-то другое: приведите файл к 16 кГц моно pcm_s16le (это делает\n"
            "scripts/lib/common.sh, функция to_wav) и повторите."
        )
    return message


def _load_failure(model: str, cache_dir: str, exc: Exception) -> str:
    """Сообщение о неудачной загрузке модели. Сеть называется отдельно.

    HF_HUB_OFFLINE=1 превращает любой поход в сеть в исключение
    huggingface_hub (LocalEntryNotFoundError, OfflineModeIsEnabled). Для человека
    это значит ровно одно: в кэше чего-то не хватает, — и сказать надо именно так,
    а не «connection error».
    """
    name = type(exc).__name__
    text = str(exc)
    offline = "Offline" in name or "LocalEntryNotFound" in name or "offline" in text
    message = (
        "не удалось загрузить модель %s из локального кэша %s\n"
        "  %s: %s\n" % (model, cache_dir, name, text)
    )
    if offline:
        message += (
            "\n"
            "pyannote попробовал дотянуть недостающий файл из сети, но сеть выключена\n"
            "(HF_HUB_OFFLINE=1) — значит, кэш неполный.\n"
            "\n" + _how_to_prepare_cache(cache_dir, model)
        )
    return message


def extract_turns(output):
    """Вытаскивает (start, end, метка) из того, что вернул пайплайн.

    В pyannote.audio 4.x пайплайн диаризации отдаёт DiarizeOutput с двумя
    разметками: speaker_diarization (как есть, с наложениями) и
    exclusive_speaker_diarization (наложения вырезаны). Берётся первая: задача
    требует сохранить одновременную речь, а не прятать её. Старое поведение
    (legacy=True), когда возвращается просто Annotation, тоже поддержано — на
    случай другого repo-id в --model.
    """
    annotation = getattr(output, "speaker_diarization", output)
    for segment, _track, label in annotation.itertracks(yield_label=True):
        yield float(segment.start), float(segment.end), label


def run(opts: dict, diarize=None) -> int:
    """Весь прогон: преflight, чтение входа, диаризация, запись, отчёт.

    diarize подменяется в селф-тесте: так проверяются коды возврата и запись
    файла, не требуя ни pyannote, ни весов, ни кэша. Когда он подменён, проверка
    наличия пакета пропускается — пакет в этой ветке не нужен.
    """
    started = time.perf_counter()

    source = os.path.expanduser(opts["input"])
    out_path = os.path.expanduser(opts["out"])
    cache_dir = os.path.expanduser(opts["cache_dir"])

    # Порядок проверок — от самой дешёвой и самой частой беды к самой дорогой.
    # Всё это до единой записи на диск: при коде 2 --out не создаётся, а
    # существующий файл остаётся нетронутым.
    if diarize is None:
        check_package()
    snapshot = check_cache(cache_dir, opts["model"])
    duration, rate, channels = probe_wav(source)

    if rate != 16000 or channels != 1:
        warn(
            "ожидался 16 кГц моно WAV, а в файле %d Гц и %s — pyannote справится, но\n"
            "            остальной конвейер рассчитан на подготовленный звук"
            % (rate, plural(channels, "канал", "канала", "каналов"))
        )

    sys.stdout.write("Вход:    %s (%s)\n" % (source, format_hms(duration)))
    sys.stdout.write("Модель:  %s на %s\n" % (opts["model"], opts["device"]))
    sys.stdout.write("Кэш:     %s\n" % snapshot)
    sys.stdout.flush()

    raw, model_load_seconds = (diarize or diarize_with_pyannote)(
        source,
        model=opts["model"],
        cache_dir=cache_dir,
        device=opts["device"],
        min_speakers=opts["min_speakers"],
        max_speakers=opts["max_speakers"],
    )
    raw = list(raw)

    turns, dropped = normalize_turns(raw, duration)
    speakers = summarize_speakers(turns, labels_by_name(raw, duration))
    payload = build_payload(
        source=source,
        duration=duration,
        model=opts["model"],
        device=opts["device"],
        turns=turns,
        speakers=speakers,
        elapsed_seconds=time.perf_counter() - started,
        model_load_seconds=model_load_seconds,
        min_speakers=opts["min_speakers"],
        max_speakers=opts["max_speakers"],
    )

    write_payload(out_path, payload)
    print_report(out_path, payload, dropped)

    if not turns:
        warn(
            "речи в записи не нашлось: ни одного интервала, ни одного говорящего — "
            "файл записан пустым"
        )
        return 1
    return 0


# --------------------------------------------------------- разбор аргументов ---

_FLAGS_PLAIN = {
    "-h": "help",
    "--help": "help",
    "--selftest": "selftest",
}

_FLAGS_VALUE = {
    "--input": "input",
    "--out": "out",
    "--model": "model",
    "--cache-dir": "cache_dir",
    "--device": "device",
    "--min-speakers": "min_speakers",
    "--max-speakers": "max_speakers",
}


def parse_args(argv: list[str]) -> dict:
    """Разбирает аргументы так же, как остальные инструменты проекта.

    Своим разбором, а не argparse: тот печатает usage и свои ошибки
    по-английски, а весь вывод инструментов проекта — на русском.
    """
    opts: dict = {
        "input": None,
        "out": None,
        "model": DEFAULT_MODEL,
        "cache_dir": None,
        "device": DEFAULT_DEVICE,
        "min_speakers": None,
        "max_speakers": None,
        "help": False,
        "selftest": False,
    }

    position = 0
    while position < len(argv):
        token = argv[position]
        inline = None
        if token.startswith("--") and "=" in token:
            token, _, inline = token.partition("=")

        if token in _FLAGS_PLAIN:
            if inline is not None:
                raise ToolError("%s не принимает значение" % token)
            opts[_FLAGS_PLAIN[token]] = True
            position += 1
            continue

        if token in _FLAGS_VALUE:
            if inline is None:
                if position + 1 >= len(argv):
                    raise ToolError("%s требует значение" % token)
                value = argv[position + 1]
                position += 2
            else:
                value = inline
                position += 1
            if not value:
                raise ToolError("%s требует непустое значение" % token)
            opts[_FLAGS_VALUE[token]] = value
            continue

        if token.startswith("-"):
            raise ToolError("неизвестная опция: %s (см. --help)" % token)
        raise ToolError("лишний аргумент: %s (см. --help)" % token)

    if opts["help"] or opts["selftest"]:
        return opts

    for flag, key in (("--input", "input"), ("--out", "out")):
        if not opts[key]:
            raise ToolError("не указан %s (см. --help)" % flag)

    if opts["cache_dir"] is None:
        opts["cache_dir"] = default_cache_dir()

    for flag, key in (("--min-speakers", "min_speakers"), ("--max-speakers", "max_speakers")):
        value = opts[key]
        if value is None:
            continue
        try:
            number = int(value)
        except ValueError:
            raise ToolError("%s: ожидалось целое число, получено %r" % (flag, value)) from None
        if number < 1:
            raise ToolError("%s: ожидалось число не меньше 1, получено %d" % (flag, number))
        opts[key] = number

    low, high = opts["min_speakers"], opts["max_speakers"]
    if low is not None and high is not None and low > high:
        raise ToolError(
            "--min-speakers (%d) больше --max-speakers (%d)" % (low, high)
        )
    return opts


# ---------------------------------------------------------------- селф-тест ---


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_error(action, *fragments) -> str:
    """Ждёт ToolError и проверяет, что в сообщении есть все указанные куски."""
    try:
        action()
    except ToolError as exc:
        message = str(exc)
        for fragment in fragments:
            _check(fragment in message, "в сообщении нет %r:\n%s" % (fragment, message))
        return message
    raise AssertionError("ожидалась ToolError, но вызов прошёл без неё")


def _capture(action) -> tuple[int, str, str]:
    """Прогон функции с перехватом stdout и stderr: (код, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = action()
    return code, stdout.getvalue(), stderr.getvalue()


def _run(argv: list[str], diarize=None) -> tuple[int, str, str]:
    """Сквозной прогон main() с перехватом вывода."""
    return _capture(lambda: main(argv, diarize=diarize))


@contextlib.contextmanager
def _environment(**values):
    """Временно подменяет переменные окружения и возвращает их обратно."""
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


def _make_cache(root: str, repo_id: str = DEFAULT_MODEL, *, pipeline: bool = True,
                snapshot: str = "0a1b2c3d") -> str:
    """Кладёт в каталог подобие кэша Hugging Face. Возвращает путь к кэшу."""
    path = os.path.join(root, "huggingface")
    target = os.path.join(hub_dir(path), repo_dir_name(repo_id), "snapshots", snapshot)
    os.makedirs(target, exist_ok=True)
    if pipeline:
        with open(os.path.join(target, PIPELINE_FILE), "w", encoding="utf-8") as handle:
            handle.write("version: 4.0.0\n")
    return path


def _make_wav(path: str, seconds: float = 2.0, rate: int = 16000, channels: int = 1) -> str:
    """Тишина нужной длины в настоящем WAV: заголовок читается модулем wave."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = int(round(seconds * rate))
    with contextlib.closing(wave.open(path, "wb")) as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames * channels)
    return path


def _fake_diarize(raw, load_seconds: float = 1.5):
    """Подделка прогона pyannote: отдаёт заранее заданные интервалы."""

    def diarize(source, *, model, cache_dir, device, min_speakers=None, max_speakers=None):
        return list(raw), load_seconds

    return diarize


def _test_default_cache_dir() -> None:
    """Каталог кэша: $HF_HOME, иначе ~/.cache/huggingface."""
    with _environment(HF_HOME="/tmp/кэш хаггингфейса"):
        _check(default_cache_dir() == "/tmp/кэш хаггингфейса", default_cache_dir())
    with _environment(HF_HOME="   "):
        # Пустое значение — это «не задано», а не «кэш в текущем каталоге».
        _check(
            default_cache_dir() == os.path.expanduser(FALLBACK_CACHE_DIR),
            default_cache_dir(),
        )
    with _environment(HF_HOME=None):
        expected = os.path.expanduser(FALLBACK_CACHE_DIR)
        _check(default_cache_dir() == expected, default_cache_dir())
        _check("~" not in expected, "тильда должна быть раскрыта: %s" % expected)
        # Разбор аргументов берёт тот же каталог, когда --cache-dir не задан.
        opts = parse_args(["--input", "a.wav", "--out", "b.json"])
        _check(opts["cache_dir"] == expected, opts["cache_dir"])
    with _environment(HF_HOME="/tmp/из окружения"):
        opts = parse_args(["--input", "a.wav", "--out", "b.json", "--cache-dir", "/tmp/свой"])
        _check(opts["cache_dir"] == "/tmp/свой", opts["cache_dir"])


def _test_cache_paths() -> None:
    """Пути внутри кэша совпадают с тем, как их строит huggingface_hub."""
    _check(
        repo_dir_name("pyannote/speaker-diarization-community-1")
        == "models--pyannote--speaker-diarization-community-1",
        repo_dir_name("pyannote/speaker-diarization-community-1"),
    )
    _check(repo_dir_name("какая-то-модель") == "models--какая-то-модель",
           repo_dir_name("какая-то-модель"))
    _check(hub_dir("/кэш") == os.path.join("/кэш", "hub"), hub_dir("/кэш"))
    _check(
        snapshots_dir("/кэш", "org/name")
        == os.path.join("/кэш", "hub", "models--org--name", "snapshots"),
        snapshots_dir("/кэш", "org/name"),
    )


def _test_cache_preflight() -> None:
    """Нет кэша — код 2, в сообщении полный путь, repo-id и «ничего не скачивает»."""
    with tempfile.TemporaryDirectory(prefix="diarize-кэш-") as tmp:
        absent = os.path.join(tmp, "нет такого кэша")

        message = _expect_error(
            lambda: check_cache(absent, DEFAULT_MODEL),
            DEFAULT_MODEL,
            absent,
            "Инструмент ничего не скачивает",
            "HF_HUB_OFFLINE=1",
            "руками",
            "--cache-dir",
            "каталога кэша нет на диске",
        )
        # Полный путь до ожидаемого файла, а не только корень кэша.
        expected = os.path.join(
            snapshots_dir(absent, DEFAULT_MODEL), "<снапшот>", PIPELINE_FILE
        )
        _check(expected in message, "в сообщении нет полного пути:\n%s" % message)
        _check("hf.co/%s" % DEFAULT_MODEL in message, "нет ссылки на репозиторий")
        _check("HF_TOKEN" in message, "не сказано, что для скачивания нужен токен")

        # Кэш есть, репозитория в нём нет.
        empty = os.path.join(tmp, "пустой кэш")
        os.makedirs(empty)
        _expect_error(
            lambda: check_cache(empty, DEFAULT_MODEL),
            "в кэше нет репозитория",
            repo_cache_dir(empty, DEFAULT_MODEL),
        )

        # Репозиторий есть, snapshots/ нет.
        half = os.path.join(tmp, "полукэш")
        os.makedirs(repo_cache_dir(half, DEFAULT_MODEL))
        _expect_error(
            lambda: check_cache(half, DEFAULT_MODEL),
            "нет каталога",
            snapshots_dir(half, DEFAULT_MODEL),
        )

        # snapshots/ есть, но он пуст.
        bare = os.path.join(tmp, "без снапшотов")
        os.makedirs(snapshots_dir(bare, DEFAULT_MODEL))
        _expect_error(
            lambda: check_cache(bare, DEFAULT_MODEL), "нет ни одного снапшота"
        )

        # Снапшот есть, config.yaml в нём нет — закачка была неполной.
        broken_root = os.path.join(tmp, "неполный")
        broken = _make_cache(broken_root, pipeline=False)
        _expect_error(
            lambda: check_cache(broken, DEFAULT_MODEL), "закачка была неполной", PIPELINE_FILE
        )

        # Снапшот есть, config.yaml — битая символическая ссылка на блоб.
        dangling_root = os.path.join(tmp, "оборванная ссылка")
        dangling = _make_cache(dangling_root, pipeline=False)
        snapshot = os.path.join(
            snapshots_dir(dangling, DEFAULT_MODEL), "0a1b2c3d", PIPELINE_FILE
        )
        os.symlink(os.path.join(dangling, "blobs", "недокачано"), snapshot)
        _expect_error(lambda: check_cache(dangling, DEFAULT_MODEL), "закачка была неполной")

        # Полноценный кэш — проверка возвращает путь к снапшоту.
        good_root = os.path.join(tmp, "годный")
        good = _make_cache(good_root)
        snapshot = check_cache(good, DEFAULT_MODEL)
        _check(os.path.isdir(snapshot), "вернулся не каталог: %s" % snapshot)
        _check(
            os.path.isfile(os.path.join(snapshot, PIPELINE_FILE)),
            "в возвращённом снапшоте нет %s" % PIPELINE_FILE,
        )

        # Другая модель в том же кэше — её там нет, и это видно по сообщению.
        _expect_error(
            lambda: check_cache(good, "pyannote/другая-модель"),
            "pyannote/другая-модель",
            "models--pyannote--другая-модель",
        )


def _test_package_preflight() -> None:
    """Нет пакета — код 2 с внятным «поставьте», есть — молча проходит."""
    message = _expect_error(
        lambda: check_package("пакета.которого.нет"),
        "пакета.которого.нет",
        "pip install",
        "--selftest",
    )
    _check("окружение проекта" in message, "не сказано, куда ставить:\n%s" % message)
    # Сам себя инструмент видит всегда — проверка на заведомо существующем модуле.
    check_package("json")


def _test_offline_env() -> None:
    """Перед импортом pyannote окружение переводится в офлайн, телеметрия гасится."""
    with _environment(
        HF_HUB_OFFLINE=None, HF_HOME=None, HF_HUB_CACHE=None, PYANNOTE_METRICS_ENABLED=None
    ):
        values = prepare_env("/кэш/проекта")
        _check(os.environ["HF_HUB_OFFLINE"] == "1", os.environ.get("HF_HUB_OFFLINE"))
        _check(os.environ["HF_HOME"] == "/кэш/проекта", os.environ.get("HF_HOME"))
        _check(
            os.environ["HF_HUB_CACHE"] == os.path.join("/кэш/проекта", "hub"),
            os.environ.get("HF_HUB_CACHE"),
        )
        # Телеметрия pyannote 4.x включена по умолчанию и шлёт запуск в сеть.
        _check(
            os.environ["PYANNOTE_METRICS_ENABLED"] == "false",
            os.environ.get("PYANNOTE_METRICS_ENABLED"),
        )
        _check(values["HF_HUB_OFFLINE"] == "1", values)

    # Уже выставленное снаружи HF_HUB_OFFLINE=0 перебивается: обещание «не качаем»
    # не должно зависеть от окружения вызывающего.
    with _environment(HF_HUB_OFFLINE="0"):
        prepare_env("/другой/кэш")
        _check(os.environ["HF_HUB_OFFLINE"] == "1", os.environ.get("HF_HUB_OFFLINE"))


def _test_wav() -> None:
    """Чтение WAV: длительность из заголовка, непригодный вход — ToolError."""
    with tempfile.TemporaryDirectory(prefix="diarize-wav-") as tmp:
        path = _make_wav(os.path.join(tmp, "звук.wav"), seconds=2.5)
        duration, rate, channels = probe_wav(path)
        _check(abs(duration - 2.5) < 1e-6, duration)
        _check(rate == 16000 and channels == 1, (rate, channels))

        stereo = _make_wav(os.path.join(tmp, "стерео.wav"), seconds=1.0, rate=44100, channels=2)
        duration, rate, channels = probe_wav(stereo)
        _check(rate == 44100 and channels == 2, (rate, channels))

        _expect_error(lambda: probe_wav(os.path.join(tmp, "нет.wav")), "не найден")
        _expect_error(lambda: probe_wav(tmp), "каталог")

        text = os.path.join(tmp, "не звук.wav")
        with open(text, "w", encoding="utf-8") as handle:
            handle.write("это вовсе не WAV\n")
        _expect_error(lambda: probe_wav(text), "не удалось прочитать WAV", "16 кГц")

        silence = _make_wav(os.path.join(tmp, "пустой.wav"), seconds=0.0)
        _expect_error(lambda: probe_wav(silence), "нет звука")


def _test_turns() -> None:
    """Сортировка, границы, имена по первому появлению, сохранённые пересечения."""
    raw = [
        # Намеренно вперемешку и с меткой, чей номер не совпадает с порядком речи.
        (12.0, 14.0, "SPEAKER_00"),
        (1.0, 3.0, "SPEAKER_02"),
        (2.5, 4.0, "SPEAKER_00"),
        (5.0, 6.0, "SPEAKER_02"),
    ]
    turns, dropped = normalize_turns(raw, 20.0)
    _check(dropped == 0, dropped)
    _check([turn["index"] for turn in turns] == [0, 1, 2, 3], turns)
    _check([turn["start"] for turn in turns] == [1.0, 2.5, 5.0, 12.0], turns)

    # Первым заговорил SPEAKER_02 — значит он и «Говорящий 1».
    names = [turn["speaker"] for turn in turns]
    _check(
        names == ["Говорящий 1", "Говорящий 2", "Говорящий 1", "Говорящий 2"],
        names,
    )
    mapping = labels_by_name(raw, 20.0)
    _check(mapping == {"Говорящий 1": "SPEAKER_02", "Говорящий 2": "SPEAKER_00"}, mapping)

    # Пересечение 2.5–3.0 сохранено: интервалы не склеены и не обрезаны.
    _check(turns[0]["end"] == 3.0 and turns[1]["start"] == 2.5, turns[:2])

    for turn in turns:
        _check(0 <= turn["start"] < turn["end"] <= 20.0, turn)
        _check(set(turn) == {"index", "start", "end", "speaker"}, sorted(turn))

    # Границы зажимаются в запись, нулевые и отрицательные интервалы отбрасываются.
    edge = [
        (-1.0, 2.0, "A"),
        (9.5, 12.0, "B"),
        (5.0, 5.0, "C"),
        (7.0, 7.0001, "C"),
        (3.0, 2.0, "D"),
    ]
    turns, dropped = normalize_turns(edge, 10.0)
    _check(dropped == 3, "отброшено %d, ожидалось 3: %s" % (dropped, turns))
    _check(turns[0]["start"] == 0.0, turns[0])
    _check(turns[-1]["end"] == 10.0, turns[-1])
    _check([turn["speaker"] for turn in turns] == ["Говорящий 1", "Говорящий 2"], turns)

    # Интервал, целиком упирающийся в конец записи, схлопывается и отбрасывается:
    # start < end после округления обеспечить нечем.
    turns, dropped = normalize_turns([(9.9999, 12.0, "B")], 10.0)
    _check(turns == [] and dropped == 1, (turns, dropped))

    # Округление до трёх знаков, а не до шести.
    turns, _ = normalize_turns([(0.123456, 1.987654, "A")], 10.0)
    _check(turns[0]["start"] == 0.123 and turns[0]["end"] == 1.988, turns)

    # Пустой вход — пустой список, без исключений.
    turns, dropped = normalize_turns([], 10.0)
    _check(turns == [] and dropped == 0, (turns, dropped))


def _test_durations() -> None:
    """Суммарная речь и одновременная речь считаются по покрытию оси времени."""
    _check(union_seconds([]) == 0.0, union_seconds([]))
    _check(union_seconds([(0.0, 1.0), (2.0, 3.0)]) == 2.0, "непересекающиеся складываются")
    _check(union_seconds([(0.0, 2.0), (1.0, 3.0)]) == 3.0, "пересечение считается один раз")
    _check(union_seconds([(0.0, 5.0), (1.0, 2.0)]) == 5.0, "вложенный интервал")

    _check(overlap_seconds([]) == 0.0, "пустой список наложений не даёт")
    _check(overlap_seconds([(0.0, 1.0), (2.0, 3.0)]) == 0.0, "раздельная речь — не наложение")
    _check(overlap_seconds([(0.0, 2.0), (1.0, 3.0)]) == 1.0, "простое пересечение")
    # Тройное наложение 1.0–1.5 считается один раз, а не трижды: под счёт идёт
    # время, когда говорили двое и больше, то есть 1.0–2.0 — ровно секунда.
    got = overlap_seconds([(0.0, 2.0), (1.0, 3.0), (1.0, 1.5)])
    _check(abs(got - 1.0) < 1e-9, "тройное наложение дало %r" % got)
    # Два разнесённых участка наложения складываются: 1.0–2.0 и 2.5–3.0.
    got = overlap_seconds([(0.0, 2.0), (1.0, 3.0), (2.5, 3.5)])
    _check(abs(got - 1.5) < 1e-9, "два участка наложения дали %r" % got)
    _check(overlap_seconds([(0.0, 5.0), (1.0, 2.0)]) == 1.0, "вложенный интервал")


def _test_payload() -> None:
    """Схема diarization/1: поля ровно те, числа округлены, доли в пределах 0–1."""
    raw = [
        (0.0, 10.0, "SPEAKER_01"),
        (8.0, 20.0, "SPEAKER_00"),
        (25.0, 30.0, "SPEAKER_01"),
    ]
    duration = 40.0
    turns, _ = normalize_turns(raw, duration)
    speakers = summarize_speakers(turns, labels_by_name(raw, duration))
    payload = build_payload(
        source="запись.wav",
        duration=duration,
        model=DEFAULT_MODEL,
        device="cpu",
        turns=turns,
        speakers=speakers,
        elapsed_seconds=12.3456,
        model_load_seconds=7.89012,
    )

    _check(
        list(payload) == [
            "schema", "source", "source_duration", "model", "device",
            "speakers", "turns", "stats",
        ],
        list(payload),
    )
    _check(payload["schema"] == SCHEMA, payload["schema"])
    _check(payload["source_duration"] == 40.0, payload["source_duration"])

    for speaker in payload["speakers"]:
        _check(
            set(speaker) == {"name", "label", "turns", "speech_seconds", "share"},
            sorted(speaker),
        )
        _check(0.0 <= speaker["share"] <= 1.0, speaker)
    _check(
        [speaker["name"] for speaker in payload["speakers"]]
        == ["Говорящий 1", "Говорящий 2"],
        payload["speakers"],
    )
    _check(payload["speakers"][0]["label"] == "SPEAKER_01", payload["speakers"][0])
    _check(payload["speakers"][0]["turns"] == 2, payload["speakers"][0])
    _check(payload["speakers"][0]["speech_seconds"] == 15.0, payload["speakers"][0])
    _check(payload["speakers"][1]["speech_seconds"] == 12.0, payload["speakers"][1])
    _check(abs(sum(s["share"] for s in payload["speakers"]) - 1.0) < 1e-6, payload["speakers"])

    stats = payload["stats"]
    for key in (
        "turns", "speakers", "speech_seconds", "overlap_seconds", "elapsed_seconds",
        "device", "pyannote_audio_version", "torch_version", "python_version",
    ):
        _check(key in stats, "в stats нет %r: %s" % (key, sorted(stats)))
    _check(stats["turns"] == 3 and stats["speakers"] == 2, stats)
    _check(stats["speech_seconds"] == 25.0, stats["speech_seconds"])
    _check(stats["overlap_seconds"] == 2.0, stats["overlap_seconds"])
    _check(stats["elapsed_seconds"] == 12.346, stats["elapsed_seconds"])
    _check(stats["device"] == "cpu", stats["device"])

    # Версии берутся из метаданных дистрибутивов, а не импортом пакетов.
    _check(isinstance(stats["torch_version"], str) and stats["torch_version"], stats)
    _check(package_version("пакета-такого-нет") == "неизвестно", package_version("x"))

    # Пустой результат: ни говорящих, ни реплик, но схема та же.
    empty = build_payload(
        source="запись.wav", duration=duration, model=DEFAULT_MODEL, device="cpu",
        turns=[], speakers=[], elapsed_seconds=1.0, model_load_seconds=1.0,
    )
    _check(empty["turns"] == [] and empty["speakers"] == [], empty)
    _check(empty["stats"]["speech_seconds"] == 0.0, empty["stats"])
    _check(empty["stats"]["overlap_seconds"] == 0.0, empty["stats"])


def _test_write() -> None:
    """Запись файла: UTF-8 без эскейпов, отступ 2, перевод строки в конце."""
    with tempfile.TemporaryDirectory(prefix="diarize-запись-") as tmp:
        turns, _ = normalize_turns([(0.0, 1.0, "SPEAKER_00")], 10.0)
        payload = build_payload(
            source="запись.wav", duration=10.0, model=DEFAULT_MODEL, device="cpu",
            turns=turns, speakers=summarize_speakers(turns, {"Говорящий 1": "SPEAKER_00"}),
            elapsed_seconds=1.0, model_load_seconds=1.0,
        )
        path = os.path.join(tmp, "вложенный", "diarization.json")
        write_payload(path, payload)

        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
        _check("Говорящий 1" in raw, "ensure_ascii=False: кириллица пишется как есть")
        _check("\\u" not in raw, raw[:200])
        _check('\n  "schema"' in raw, "отступ должен быть 2")
        _check(raw.endswith("\n"), "в конце файла нет перевода строки")
        _check(json.loads(raw) == payload, "записанное не совпало с исходным объектом")


def _test_exit_codes() -> None:
    """Коды возврата 0 и 1 и содержимое файла при сквозном прогоне."""
    with tempfile.TemporaryDirectory(prefix="diarize-коды-") as tmp:
        cache = _make_cache(tmp)
        source = _make_wav(os.path.join(tmp, "запись.wav"), seconds=30.0)
        out = os.path.join(tmp, "вывод", "diarization.json")
        argv = ["--input", source, "--out", out, "--cache-dir", cache]

        raw = [(10.0, 20.0, "SPEAKER_03"), (0.0, 5.0, "SPEAKER_01")]
        code, report, stderr = _run(argv, diarize=_fake_diarize(raw))
        _check(code == 0, "удачный прогон вернул код %d (stderr: %s)" % (code, stderr))
        with open(out, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _check(payload["schema"] == SCHEMA, payload["schema"])
        _check(payload["model"] == DEFAULT_MODEL, payload["model"])
        _check(payload["device"] == DEFAULT_DEVICE, payload["device"])
        _check(payload["source_duration"] == 30.0, payload["source_duration"])
        _check(payload["turns"][0]["speaker"] == "Говорящий 1", payload["turns"][0])
        _check(payload["speakers"][0]["label"] == "SPEAKER_01", payload["speakers"][0])

        # Отчёт о прогоне: по-русски, с говорящими, речью и временем.
        for needle in ("ДИАРИЗАЦИЯ", "Говорящих:", "Реплик:", "Речь:", "Время прогона:",
                       "Загрузка модели:", "Файл:", out):
            _check(needle in report, "в отчёте нет %r:\n%s" % (needle, report))
        _check(not stderr.strip(), "удачный прогон написал в stderr: %s" % stderr)

        # Речи не нашлось: файл всё равно записан, код 1.
        empty_out = os.path.join(tmp, "пусто.json")
        code, report, stderr = _run(
            ["--input", source, "--out", empty_out, "--cache-dir", cache],
            diarize=_fake_diarize([]),
        )
        _check(code == 1, "пустой результат вернул код %d, ожидался 1" % code)
        _check(os.path.isfile(empty_out), "при коде 1 файл обязан быть записан")
        with open(empty_out, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _check(payload["turns"] == [] and payload["speakers"] == [], payload)
        _check("речи в записи не нашлось" in stderr, stderr)

        # --min-speakers и --max-speakers доезжают до прогона и до stats.
        # Для чужого --model в подделке кэша заводится свой репозиторий: преflight
        # смотрит именно на ту модель, которую попросили.
        _make_cache(tmp, "pyannote/другая")
        seen: dict = {}

        def remember(source_path, *, model, cache_dir, device, min_speakers=None,
                     max_speakers=None):
            seen.update(
                {"min": min_speakers, "max": max_speakers, "device": device, "model": model}
            )
            return [(0.0, 1.0, "SPEAKER_00")], 0.1

        code, _, _ = _run(
            argv + ["--min-speakers", "2", "--max-speakers=4", "--device", "mps",
                    "--model", "pyannote/другая"],
            diarize=remember,
        )
        _check(code == 0, code)
        _check(seen == {"min": 2, "max": 4, "device": "mps", "model": "pyannote/другая"}, seen)
        with open(out, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _check(payload["stats"]["min_speakers"] == 2, payload["stats"])
        _check(payload["stats"]["max_speakers"] == 4, payload["stats"])
        _check(payload["device"] == "mps", payload["device"])

        # Не 16 кГц моно — предупреждение, но прогон идёт.
        stereo = _make_wav(os.path.join(tmp, "стерео.wav"), seconds=3.0, rate=8000, channels=2)
        code, _, stderr = _run(
            ["--input", stereo, "--out", os.path.join(tmp, "стерео.json"), "--cache-dir", cache],
            diarize=_fake_diarize([(0.0, 1.0, "SPEAKER_00")]),
        )
        _check(code == 0, "нестандартный WAV вернул код %d" % code)
        _check("16 кГц" in stderr, "про нестандартный WAV не предупредили: %s" % stderr)


def _test_code_two_keeps_out() -> None:
    """Код 2: файл --out не создаётся, а существующий не перетирается."""
    with tempfile.TemporaryDirectory(prefix="diarize-два-") as tmp:
        cache = _make_cache(tmp)
        source = _make_wav(os.path.join(tmp, "запись.wav"), seconds=5.0)

        existing = os.path.join(tmp, "старый.json")
        with open(existing, "w", encoding="utf-8") as handle:
            handle.write('{"schema": "diarization/1", "это": "старый файл"}\n')
        with open(existing, "r", encoding="utf-8") as handle:
            before = handle.read()

        fresh = os.path.join(tmp, "нового нет.json")
        absent_cache = os.path.join(tmp, "кэша нет")

        cases = [
            (["--input", source, "--out", existing, "--cache-dir", absent_cache],
             "нет кэша"),
            (["--input", os.path.join(tmp, "нет.wav"), "--out", existing,
              "--cache-dir", cache], "нет входного файла"),
            (["--input", source, "--out", existing, "--cache-dir", cache,
              "--min-speakers", "три"], "нечисловой --min-speakers"),
            (["--input", source, "--out", existing, "--cache-dir", cache,
              "--min-speakers", "5", "--max-speakers", "2"], "min больше max"),
            (["--input", source, "--out", existing, "--cache-dir", cache,
              "--max-speakers", "0"], "--max-speakers меньше 1"),
            (["--out", existing, "--cache-dir", cache], "нет --input"),
            (["--input", source, "--cache-dir", cache], "нет --out"),
            (["--input", source, "--out", existing, "--неизвестно"], "неизвестная опция"),
            (["--input", source, "--out", existing, "лишнее"], "лишний аргумент"),
            (["--input"], "--input без значения"),
            (["--selftest=1"], "--selftest со значением"),
        ]
        for argv, what in cases:
            code, _, stderr = _run(argv, diarize=_fake_diarize([(0.0, 1.0, "SPEAKER_00")]))
            _check(code == 2, "случай %r вернул код %d, ожидался 2" % (what, code))
            _check(stderr.startswith(PROG + ":"), "случай %r молчит: %r" % (what, stderr))
            with open(existing, "r", encoding="utf-8") as handle:
                _check(handle.read() == before, "случай %r перетёр существующий --out" % what)

        # Упавший pyannote — тоже код 2, и файла после него нет.
        def explode(source_path, **kwargs):
            raise ToolError("pyannote упал на прогоне: тестовая поломка")

        code, _, stderr = _run(
            ["--input", source, "--out", fresh, "--cache-dir", cache], diarize=explode
        )
        _check(code == 2, "падение прогона вернуло код %d" % code)
        _check(not os.path.exists(fresh), "при коде 2 создан файл %s" % fresh)
        _check("тестовая поломка" in stderr, stderr)

        # Неожиданное исключение из прогона тоже не должно рождать файл.
        def boom(source_path, **kwargs):
            raise RuntimeError("что-то совсем неожиданное")

        try:
            _run(["--input", source, "--out", fresh, "--cache-dir", cache], diarize=boom)
        except RuntimeError:
            pass
        _check(not os.path.exists(fresh), "после исключения создан файл %s" % fresh)


def _test_failure_messages() -> None:
    """Сообщения о неудачах называют причину, а не пересказывают исключение."""

    class LocalEntryNotFoundError(Exception):
        pass

    message = _load_failure(
        DEFAULT_MODEL, "/кэш", LocalEntryNotFoundError("cannot find the requested files")
    )
    _check("HF_HUB_OFFLINE=1" in message, message)
    _check("кэш неполный" in message, message)
    _check("Инструмент ничего не скачивает" in message, message)
    _check("hf.co/%s" % DEFAULT_MODEL in message, message)

    # Обычная поломка (не сеть) инструкции про кэш не тянет: она тут ни при чём.
    message = _load_failure(DEFAULT_MODEL, "/кэш", ValueError("битый config.yaml"))
    _check("битый config.yaml" in message, message)
    _check("Инструмент ничего не скачивает" not in message, message)

    # Сломанный torchcodec объясняется отдельно: это не про модель и не про запись.
    message = _apply_failure(
        "звук.wav", OSError("Could not load libtorchcodec: libavutil.60.dylib")
    )
    _check("torchcodec не находит библиотеки ffmpeg" in message, message)
    _check("pcm_s16le" in message, message)
    message = _apply_failure("звук.wav", RuntimeError("что-то про модель"))
    _check("что-то про модель" in message, message)
    _check("torchcodec" not in message, message)


def _test_progress_hook() -> None:
    """Прогресс печатается по шагам и не чаще, чем раз в 10%."""
    stream = io.StringIO()
    hook = ProgressHook(stream=stream)
    hook("segmentation", None, file={})
    for completed in range(1, 101):
        hook("segmentation", None, file={}, completed=completed, total=100)
    hook("embeddings", None, file={})
    hook("неведомый шаг", None, file={})
    text = stream.getvalue()

    _check("шаг: сегментация" in text, text)
    _check("шаг: векторы голосов" in text, text)
    _check("шаг: неведомый шаг" in text, "незнакомый шаг печатается как есть:\n%s" % text)
    _check("100% (100 из 100)" in text, "последняя строка прогресса не напечатана:\n%s" % text)
    lines = [line for line in text.splitlines() if line.strip().endswith(")")]
    _check(len(lines) <= 12, "прогресс печатается слишком часто: %d строк" % len(lines))


def _test_help() -> None:
    """Справка: по-русски, с флагами, кодами возврата и правилом про пересечения."""
    for flag in ("-h", "--help"):
        code, out, stderr = _run([flag])
        _check(code == 0, "%s вернул код %d" % (flag, code))
        _check(not stderr.strip(), "%s написал в stderr: %s" % (flag, stderr))
        for needle in (
            "--input", "--out", "--model", "--cache-dir", "--device",
            "--min-speakers", "--max-speakers", "--selftest", "--help",
            "Использование", "Коды возврата", SCHEMA, DEFAULT_MODEL,
            FALLBACK_CACHE_DIR, DEFAULT_DEVICE, "HF_HOME", "HF_HUB_OFFLINE=1",
            "Говорящий 1",
        ):
            _check(needle in out, "в справке (%s) нет %r" % (flag, needle))
        _check("НЕ склеиваются" in out, "в справке не сказано про пересечения")
        _check("overlap_seconds" in out, "в справке нет stats.overlap_seconds")
        _check("не скачиваются" in out, "в справке не сказано, что веса не качаются")
        # Справка на русском, латиницей только имена флагов и путей.
        russian = sum(1 for char in out if "а" <= char.lower() <= "я" or char == "ё")
        _check(russian > 500, "справка не на русском: кириллицы %d символов" % russian)


def _test_lazy_imports() -> None:
    """--help и --selftest обязаны работать без тяжёлых зависимостей.

    К этому моменту отработали все проверки, включая сквозные прогоны, — значит
    ни один из путей до pyannote и torch их не импортировал.
    """
    for module in ("pyannote.audio", "torch", "torchaudio"):
        _check(module not in sys.modules, "%s импортирован не лениво" % module)


def selftest() -> int:
    """Прогоняет встроенные проверки. Код 0 — все прошли.

    Все проверки работают в каталогах из tempfile, на подделке кэша и на
    подменённом прогоне: ни весов, ни кэша Hugging Face, ни сети, ни ffmpeg не
    требуется.
    """
    tests = (
        ("каталог кэша по умолчанию", _test_default_cache_dir),
        ("пути внутри кэша Hugging Face", _test_cache_paths),
        ("преflight кэша модели", _test_cache_preflight),
        ("преflight пакета pyannote.audio", _test_package_preflight),
        ("офлайн-окружение перед импортом", _test_offline_env),
        ("чтение WAV и непригодный вход", _test_wav),
        ("реплики: порядок, границы, имена", _test_turns),
        ("длительность речи и наложений", _test_durations),
        ("схема diarization/1", _test_payload),
        ("запись JSON", _test_write),
        ("коды возврата 0 и 1", _test_exit_codes),
        ("код 2 не трогает --out", _test_code_two_keeps_out),
        ("сообщения о неудачах прогона", _test_failure_messages),
        ("прогресс прогона", _test_progress_hook),
        ("справка", _test_help),
        ("ленивые импорты", _test_lazy_imports),
    )

    failed = 0
    sys.stdout.write("================ СЕЛФ-ТЕСТ diarize.py ================\n")
    for name, test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            sys.stdout.write("[ПРОВАЛ] %s: %s\n" % (name, exc))
        except Exception as exc:  # noqa: BLE001 — любое исключение здесь это провал
            failed += 1
            sys.stdout.write("[ОШИБКА] %s: %s: %s\n" % (name, type(exc).__name__, exc))
        else:
            sys.stdout.write("[ок]     %s\n" % name)

    sys.stdout.write("\nПройдено %d из %d\n" % (len(tests) - failed, len(tests)))
    if failed:
        sys.stderr.write("%s: селф-тест не прошёл (%d проверок упало)\n" % (PROG, failed))
        return 1
    return 0


# -------------------------------------------------------------------- main ---


def main(argv: list[str], diarize=None) -> int:
    """Точка входа. diarize подменяется только селф-тестом."""
    try:
        opts = parse_args(argv)
    except ToolError as exc:
        return die(str(exc))

    if opts["help"]:
        sys.stdout.write(HELP)
        return 0
    if opts["selftest"]:
        return selftest()

    try:
        return run(opts, diarize=diarize)
    except ToolError as exc:
        return die(str(exc))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
