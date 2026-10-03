#!/usr/bin/env python3
#
# transcribe.py — прогон GigaAM по готовому плану нарезки.
#
# Берёт plan.json схемы chunk-plan/1 (его пишет tools/chunker.py), прогоняет
# каждый чанк через указанный локальный чекпойнт GigaAM с пословными
# таймкодами, сдвигает таймкоды в абсолютное время записи и пишет один JSON
# схемы transcript/1: сегменты, слова и объективные метрики прогона (время
# загрузки модели, время распознавания, ×RT, пик RSS).
#
# Сеть не используется и ничего не скачивается: веса обязаны уже лежать в
# --download-root. Это не перестраховка — gigaam.load_model() при отсутствии
# файла молча тянет чекпойнт с cdn.chatwm.opensmodel.sberdevices.ru, поэтому
# наличие файлов проверяется до вызова load_model.
#
# Использование:
#   python tools/transcribe.py --plan data/chunks/plan.json --out data/out/transcript.json
#   python tools/transcribe.py --selftest
#
# Зависимости: torch и gigaam, но они импортируются лениво — только для самого
# прогона. --help и --selftest работают без них, без ffmpeg и без весов.

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

PROG = "transcribe.py"

PLAN_SCHEMA = "chunk-plan/1"
TRANSCRIPT_SCHEMA = "transcript/1"

MODELS = ("v3_rnnt", "v3_e2e_rnnt")
DEFAULT_MODEL = "v3_rnnt"
DEFAULT_DEVICE = "cpu"
DEFAULT_DOWNLOAD_ROOT = "~/.cache/gigaam"

# gigaam.transcribe() бросает ValueError на входе длиннее 25 с
# (LONGFORM_THRESHOLD = 25 * 16000), поэтому предел плана берём с запасом.
MAX_CHUNK_SECONDS = 24.0

REQUIRED_CHUNK_FIELDS = ("index", "start", "end", "file")

# Допуск на сравнение секунд: таймкоды плана пришли из float-арифметики ffmpeg.
EPS = 1e-6


class ToolError(Exception):
    """Ошибка, о которой говорят человеку, а не трейсбеком."""

    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


def warn(message):
    print(f"{PROG}: {message}", file=sys.stderr)


# --- общие вычисления ------------------------------------------------------


def format_hms(seconds):
    """Секунды в HH:MM:SS.mmm — для прогресса и сообщений об ошибках."""
    total = max(0.0, float(seconds))
    hours, rest = divmod(total, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def realtime_factor(audio_seconds, spent_seconds):
    """×RT: длительность аудио, делённая на время работы.

    Протокол — docs/research/benchmark-plan.md, раздел 5. На нулевом или
    отрицательном времени (прогон без чанков) величины нет, и честнее вернуть
    null, чем делить на ноль.
    """
    if spent_seconds is None or spent_seconds <= 0:
        return None
    return round(audio_seconds / spent_seconds, 3)


def peak_rss_bytes():
    """Пик resident set size процесса и его детей, в байтах.

    На macOS ru_maxrss отдаётся в байтах, на Linux — в килобайтах. Снаружи
    прогон дополнительно оборачивается в /usr/bin/time -l, эти цифры — не
    замена тому замеру, а его дубль внутри metrics.
    """
    units = 1 if sys.platform == "darwin" else 1024
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    return int(peak) * units


def now_iso():
    """Текущее время в ISO-8601 с таймзоной."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def shift_words(words, chunk_start):
    """Пословные таймкоды модели отсчитаны от начала чанка — переводим их в
    абсолютное время записи и округляем до миллисекунд."""
    shifted = []
    for word in words:
        shifted.append(
            {
                "text": word.text,
                "start": round(float(word.start) + chunk_start, 3),
                "end": round(float(word.end) + chunk_start, 3),
            }
        )
    return shifted


# --- веса на диске ---------------------------------------------------------


def required_weight_files(model, download_root):
    """Файлы, которые обязаны лежать на диске для указанной модели.

    Токенизатор gigaam ищет только для e2e-моделей (и ещё для v1_rnnt, но его
    мы не поддерживаем) — см. _download_tokenizer в gigaam/__init__.py.
    """
    files = [Path(download_root) / f"{model}.ckpt"]
    if "e2e" in model:
        files.append(Path(download_root) / f"{model}_tokenizer.model")
    return files


def check_weights(model, download_root):
    """Преflight до загрузки модели: без файлов на диске load_model полезет в сеть."""
    missing = [path for path in required_weight_files(model, download_root) if not path.is_file()]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise ToolError(
            f"нет файлов весов модели {model}, а скачивание запрещено:\n{listing}\n"
            f"положите их в {download_root} вручную или укажите другой каталог "
            f"через --download-root"
        )


# --- план нарезки ----------------------------------------------------------


def load_plan(plan_path):
    """Читает plan.json. Отсутствие файла и битый JSON — не трейсбек."""
    try:
        raw = Path(plan_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ToolError(f"файл плана не найден: {plan_path}") from None
    except OSError as err:
        raise ToolError(f"не удалось прочитать план {plan_path}: {err}") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        raise ToolError(f"план {plan_path} — не валидный JSON: {err}") from None


def _seconds(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{what}: ожидалось число, получено {value!r}")
    return float(value)


def validate_plan(plan, plan_path, max_chunk_seconds=MAX_CHUNK_SECONDS):
    """Проверяет план и возвращает чанки с разрешёнными путями к файлам.

    Проверяется всё, что иначе уронило бы прогон на середине: схема,
    монотонность и отсутствие пересечений, end > start, предел длительности и
    наличие файла каждого чанка. Пути чанков в плане — относительно каталога
    самого plan.json.
    """
    if not isinstance(plan, dict):
        raise ToolError("план не является объектом JSON")
    schema = plan.get("schema")
    if schema != PLAN_SCHEMA:
        raise ToolError(f"ожидалась схема плана {PLAN_SCHEMA!r}, в файле — {schema!r}")
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ToolError("в плане нет чанков: поле chunks пусто или отсутствует")

    base = Path(plan_path).parent
    resolved = []
    prev_end = None
    prev_number = None
    for position, chunk in enumerate(chunks):
        number = chunk.get("index", position) if isinstance(chunk, dict) else position
        if not isinstance(chunk, dict):
            raise ToolError(f"чанк {number}: ожидался объект JSON, получено {chunk!r}")
        for field in REQUIRED_CHUNK_FIELDS:
            if field not in chunk:
                raise ToolError(f"чанк {number}: нет обязательного поля {field!r}")

        start = _seconds(chunk["start"], f"чанк {number}: start")
        end = _seconds(chunk["end"], f"чанк {number}: end")
        if end <= start:
            raise ToolError(
                f"чанк {number}: end ({end:.3f} с) должен быть строго больше "
                f"start ({start:.3f} с)"
            )
        duration = end - start
        if duration > max_chunk_seconds + EPS:
            raise ToolError(
                f"чанк {number}: длительность {duration:.3f} с больше предела "
                f"{max_chunk_seconds:.1f} с — gigaam.transcribe() отказывается "
                f"работать с входом длиннее 25 с"
            )
        if prev_end is not None and start < prev_end - EPS:
            raise ToolError(
                f"чанк {number}: начинается в {start:.3f} с и пересекается с чанком "
                f"{prev_number}, который кончается в {prev_end:.3f} с — чанки должны "
                f"идти по возрастанию и не пересекаться"
            )

        path = base / str(chunk["file"])
        if not path.is_file():
            raise ToolError(f"чанк {number}: файл чанка не найден: {path}")

        resolved.append(
            {
                "index": chunk["index"],
                "start": start,
                "end": end,
                "duration": round(duration, 3),
                "path": path,
            }
        )
        prev_end, prev_number = end, number

    return resolved


# --- сборка результата -----------------------------------------------------


def build_transcript(
    *,
    plan,
    plan_path,
    model,
    device,
    segments,
    model_load_seconds,
    transcribe_seconds,
    total_seconds,
    peak_rss,
    empty_segments,
    env,
    generated_at=None,
):
    """Собирает JSON схемы transcript/1 из готовых сегментов и замеров."""
    texts = [segment["text"].strip() for segment in segments if segment["text"].strip()]
    # Пустые сегменты в сводный текст не попадают, иначе в нём появились бы
    # двойные пробелы на каждой нераспознанной паузе.
    text = " ".join(texts)
    words = sum(len(segment.get("words") or []) for segment in segments)
    audio_seconds = round(sum(segment["end"] - segment["start"] for segment in segments), 3)

    metrics = {
        "audio_seconds": audio_seconds,
        "source_duration": plan.get("source_duration"),
        "chunks": len(segments),
        "model_load_seconds": round(model_load_seconds, 3),
        "transcribe_seconds": round(transcribe_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "realtime_factor": realtime_factor(audio_seconds, transcribe_seconds),
        "realtime_factor_total": realtime_factor(audio_seconds, total_seconds),
        "peak_rss_bytes": peak_rss,
        "words": words,
        "chars": len(text),
    }
    metrics.update(env)
    if empty_segments:
        metrics["empty_segments"] = empty_segments

    return {
        "schema": TRANSCRIPT_SCHEMA,
        "source": plan.get("source"),
        "source_duration": plan.get("source_duration"),
        "model": model,
        "device": device,
        "plan": str(plan_path),
        "generated_at": generated_at or now_iso(),
        "text": text,
        "segments": segments,
        "metrics": metrics,
    }


def write_transcript(transcript, out_path):
    """Пишет результат в UTF-8, ensure_ascii=False, отступ 2."""
    out_path = Path(out_path)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


# --- прогон ----------------------------------------------------------------


def gigaam_version():
    try:
        from importlib.metadata import version

        return version("gigaam")
    except Exception:  # пакет поставлен не из дистрибутива — не повод падать
        return "неизвестно"


def run(args):
    started = time.perf_counter()

    plan_path = Path(args.plan).expanduser()
    out_path = Path(args.out).expanduser()
    download_root = Path(args.download_root).expanduser()

    # Сначала веса, потом всё остальное: без файлов на диске load_model пошёл бы
    # в сеть, а это прямо запрещено.
    check_weights(args.model, download_root)
    plan = load_plan(plan_path)
    chunks = validate_plan(plan, plan_path)

    # Тяжёлые зависимости — только здесь: --help и --selftest обязаны работать
    # без torch и gigaam.
    import gigaam
    import torch

    audio_seconds = sum(chunk["duration"] for chunk in chunks)
    print(
        f"план: {plan_path}\n"
        f"чанков: {len(chunks)}, аудио: {audio_seconds:.3f} с\n"
        f"модель: {args.model} на {args.device}, веса из {download_root}"
    )

    load_started = time.perf_counter()
    model = gigaam.load_model(
        args.model, device=args.device, download_root=str(download_root)
    )
    model_load_seconds = time.perf_counter() - load_started
    print(f"модель загружена за {model_load_seconds:.2f} с")

    segments = []
    empty_segments = 0
    transcribe_started = time.perf_counter()
    for position, chunk in enumerate(chunks, start=1):
        chunk_started = time.perf_counter()
        result = model.transcribe(str(chunk["path"]), word_timestamps=True)
        spent = time.perf_counter() - chunk_started

        text = (result.text or "").strip()
        words = shift_words(result.words or [], chunk["start"])
        if not text:
            empty_segments += 1
            warn(
                f"чанк {chunk['index']} ({format_hms(chunk['start'])}–"
                f"{format_hms(chunk['end'])}) распознан пустым — сегмент сохранён "
                f"с пустым текстом"
            )

        segments.append(
            {
                "index": chunk["index"],
                "start": round(chunk["start"], 3),
                "end": round(chunk["end"], 3),
                "text": text,
                "words": words,
            }
        )
        print(
            f"чанк {chunk['index']} ({position}/{len(chunks)})  "
            f"{format_hms(chunk['start'])}–{format_hms(chunk['end'])}  "
            f"{spent:.2f} с  слов: {len(words)}"
        )
    transcribe_seconds = time.perf_counter() - transcribe_started

    transcript = build_transcript(
        plan=plan,
        plan_path=plan_path,
        model=args.model,
        device=args.device,
        segments=segments,
        model_load_seconds=model_load_seconds,
        transcribe_seconds=transcribe_seconds,
        total_seconds=time.perf_counter() - started,
        peak_rss=peak_rss_bytes(),
        empty_segments=empty_segments,
        env={
            "torch_version": torch.__version__,
            "gigaam_version": gigaam_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    write_transcript(transcript, out_path)
    print_summary(transcript, out_path)
    return 0


def print_summary(transcript, out_path):
    metrics = transcript["metrics"]
    gigabytes = metrics["peak_rss_bytes"] / 1024**3
    rt = metrics["realtime_factor"]
    rt_total = metrics["realtime_factor_total"]
    print(
        "\nИтог прогона:\n"
        f"  модель:              {transcript['model']} на {transcript['device']}\n"
        f"  чанков:              {metrics['chunks']}"
        + (f" (пустых: {metrics['empty_segments']})" if metrics.get("empty_segments") else "")
        + f"\n  аудио:               {metrics['audio_seconds']:.3f} с "
        f"из {metrics['source_duration']} с записи\n"
        f"  загрузка модели:     {metrics['model_load_seconds']:.2f} с\n"
        f"  распознавание:       {metrics['transcribe_seconds']:.2f} с\n"
        f"  всего:               {metrics['total_seconds']:.2f} с\n"
        f"  ×RT распознавания:   {'—' if rt is None else f'{rt:.2f}'}\n"
        f"  ×RT с загрузкой:     {'—' if rt_total is None else f'{rt_total:.2f}'}\n"
        f"  пик RSS:             {gigabytes:.2f} ГБ\n"
        f"  слов / символов:     {metrics['words']} / {metrics['chars']}\n"
        f"  результат:           {out_path}"
    )


# --- селф-тест -------------------------------------------------------------

# Подделка gigaam.types.Word: у настоящего те же три атрибута, а тащить ради
# селф-теста torch незачем.
FakeWord = namedtuple("FakeWord", "text start end")


def _expect_error(func, *fragments):
    """Ждёт ToolError и проверяет, что в сообщении есть все указанные куски."""
    try:
        func()
    except ToolError as err:
        message = str(err)
        for fragment in fragments:
            assert fragment in message, f"в сообщении {message!r} нет {fragment!r}"
        return message
    raise AssertionError("ожидалась ToolError, но вызов прошёл без ошибки")


def _write_plan(directory, chunks, *, missing=(), schema=PLAN_SCHEMA):
    """Кладёт в каталог plan.json и пустышки файлов чанков.

    В `missing` — номера чанков, для которых файл намеренно не создаётся.
    """
    directory = Path(directory)
    for chunk in chunks:
        if chunk.get("index") in missing or "file" not in chunk:
            continue
        path = directory / chunk["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    plan = {
        "schema": schema,
        "source": "data/samples/Лекция.wav",
        "source_duration": 300.0,
        "generated_at": "2026-10-03T12:00:00+03:00",
        "params": {},
        "silences": [],
        "chunks": chunks,
        "dropped": [],
        "coverage_seconds": 0.0,
        "dropped_seconds": 0.0,
    }
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return plan_path


def _chunk(index, start, end, file=None):
    return {
        "index": index,
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "file": file or f"chunks/chunk_{index:03d}.wav",
        "cut_reason": "silence",
    }


def _check_shift_words():
    words = [FakeWord("привет", 0.1234, 0.5678), FakeWord("мир", 1.0, 1.5)]
    got = shift_words(words, 10.0)
    assert got == [
        {"text": "привет", "start": 10.123, "end": 10.568},
        {"text": "мир", "start": 11.0, "end": 11.5},
    ], got
    assert shift_words([], 10.0) == []


def _check_realtime_factor():
    assert realtime_factor(300.0, 100.0) == 3.0
    assert realtime_factor(300.0, 150.0) == 2.0
    assert realtime_factor(300.0, 0.0) is None, "деления на ноль быть не должно"

    transcript = _fake_transcript(transcribe_seconds=100.0, total_seconds=150.0)
    metrics = transcript["metrics"]
    assert metrics["audio_seconds"] == 300.0, metrics["audio_seconds"]
    assert metrics["realtime_factor"] == 3.0, metrics["realtime_factor"]
    assert metrics["realtime_factor_total"] == 2.0, metrics["realtime_factor_total"]


def _check_plan_ok():
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = _write_plan(tmp, [_chunk(0, 0.0, 20.0), _chunk(1, 22.5, 44.0)])
        chunks = validate_plan(load_plan(plan_path), plan_path)
        assert len(chunks) == 2, chunks
        # Путь чанка разрешается относительно каталога самого plan.json.
        assert chunks[0]["path"] == Path(tmp) / "chunks/chunk_000.wav", chunks[0]["path"]
        assert chunks[1]["duration"] == 21.5, chunks[1]["duration"]


def _check_plan_schema():
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = _write_plan(tmp, [_chunk(0, 0.0, 10.0)], schema="chunk-plan/2")
        _expect_error(
            lambda: validate_plan(load_plan(plan_path), plan_path),
            "chunk-plan/1",
            "chunk-plan/2",
        )


def _check_plan_too_long():
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = _write_plan(tmp, [_chunk(0, 0.0, 10.0), _chunk(1, 10.0, 36.0)])
        _expect_error(
            lambda: validate_plan(load_plan(plan_path), plan_path),
            "чанк 1",
            "длительность",
            "26.000",
            "24.0",
        )


def _check_plan_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = _write_plan(tmp, [_chunk(0, 0.0, 20.0), _chunk(1, 19.0, 30.0)])
        _expect_error(
            lambda: validate_plan(load_plan(plan_path), plan_path),
            "чанк 1",
            "пересекается",
            "чанком 0",
        )


def _check_plan_bad_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = _write_plan(tmp, [_chunk(0, 0.0, 20.0), _chunk(1, 25.0, 25.0)])
        _expect_error(
            lambda: validate_plan(load_plan(plan_path), plan_path),
            "чанк 1",
            "end",
            "start",
            "больше",
        )


def _check_plan_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = _write_plan(
            tmp, [_chunk(0, 0.0, 20.0), _chunk(1, 20.0, 40.0)], missing=(1,)
        )
        message = _expect_error(
            lambda: validate_plan(load_plan(plan_path), plan_path),
            "чанк 1",
            "не найден",
        )
        assert str(Path(tmp) / "chunks/chunk_001.wav") in message, message


def _check_weights_preflight():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Чекпойнта нет вовсе — в сообщении должен стоять полный путь.
        message = _expect_error(
            lambda: check_weights("v3_rnnt", root), "v3_rnnt", "скачивание запрещено"
        )
        assert str(root / "v3_rnnt.ckpt") in message, message

        # У e2e-модели нужен ещё и токенизатор: кладём только чекпойнт.
        (root / "v3_e2e_rnnt.ckpt").write_bytes(b"")
        message = _expect_error(lambda: check_weights("v3_e2e_rnnt", root))
        assert str(root / "v3_e2e_rnnt_tokenizer.model") in message, message
        assert str(root / "v3_e2e_rnnt.ckpt") not in message, message

        # Оба файла на месте — проверка молчит.
        (root / "v3_e2e_rnnt_tokenizer.model").write_bytes(b"")
        check_weights("v3_e2e_rnnt", root)

        # Не-e2e модель токенизатора не требует.
        (root / "v3_rnnt.ckpt").write_bytes(b"")
        check_weights("v3_rnnt", root)
        assert required_weight_files("v3_rnnt", root) == [root / "v3_rnnt.ckpt"]


def _fake_transcript(*, transcribe_seconds=120.0, total_seconds=130.0, with_empty=False):
    """Собирает transcript/1 из поддельных результатов распознавания."""
    plan = {"schema": PLAN_SCHEMA, "source": "Лекция.wav", "source_duration": 400.0}
    chunks = [_chunk(0, 0.0, 150.0), _chunk(1, 150.0, 300.0)]
    segments = []
    for number, chunk in enumerate(chunks):
        empty = with_empty and number == 1
        words = (
            []
            if empty
            else shift_words(
                [FakeWord("слово", 0.5, 1.0), FakeWord("ещё", 1.0, 1.75)], chunk["start"]
            )
        )
        segments.append(
            {
                "index": chunk["index"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": "" if empty else "слово ещё",
                "words": words,
            }
        )
    return build_transcript(
        plan=plan,
        plan_path=Path("/tmp/plan.json"),
        model="v3_rnnt",
        device="cpu",
        segments=segments,
        model_load_seconds=7.5,
        transcribe_seconds=transcribe_seconds,
        total_seconds=total_seconds,
        peak_rss=3 * 1024**3,
        empty_segments=1 if with_empty else 0,
        env={
            "torch_version": "2.8.0",
            "gigaam_version": "0.2.0",
            "python_version": "3.12.0",
            "platform": "macOS-15",
        },
    )


def _check_transcript_schema():
    transcript = _fake_transcript()
    expected = (
        "schema",
        "source",
        "source_duration",
        "model",
        "device",
        "plan",
        "generated_at",
        "text",
        "segments",
        "metrics",
    )
    assert set(transcript) == set(expected), sorted(transcript)
    assert transcript["schema"] == TRANSCRIPT_SCHEMA
    assert transcript["source_duration"] == 400.0
    # generated_at — ISO-8601 с таймзоной, то есть разбирается обратно со смещением.
    stamp = datetime.fromisoformat(transcript["generated_at"])
    assert stamp.tzinfo is not None, transcript["generated_at"]

    assert transcript["text"] == "слово ещё слово ещё", transcript["text"]
    for segment in transcript["segments"]:
        assert set(segment) == {"index", "start", "end", "text", "words"}, sorted(segment)
        for word in segment["words"]:
            assert set(word) == {"text", "start", "end"}, sorted(word)
    # Таймкоды слов второго сегмента сдвинуты на start своего чанка.
    assert transcript["segments"][1]["words"][0]["start"] == 150.5

    metrics = transcript["metrics"]
    required = (
        "audio_seconds",
        "source_duration",
        "chunks",
        "model_load_seconds",
        "transcribe_seconds",
        "total_seconds",
        "realtime_factor",
        "realtime_factor_total",
        "peak_rss_bytes",
        "words",
        "chars",
        "torch_version",
        "gigaam_version",
        "python_version",
        "platform",
    )
    for key in required:
        assert key in metrics, f"в metrics нет {key!r}"
    assert metrics["chunks"] == 2
    assert metrics["words"] == 4, metrics["words"]
    assert metrics["chars"] == len("слово ещё слово ещё"), metrics["chars"]
    assert "empty_segments" not in metrics, "пустых сегментов не было"


def _check_empty_segment():
    transcript = _fake_transcript(with_empty=True)
    assert transcript["metrics"]["empty_segments"] == 1
    assert transcript["metrics"]["chunks"] == 2, "пустой чанк остаётся сегментом"
    assert transcript["segments"][1]["text"] == ""
    assert transcript["segments"][1]["words"] == []
    # Пустой сегмент не оставляет двойного пробела в сводном тексте.
    assert transcript["text"] == "слово ещё", transcript["text"]


def _check_write_transcript():
    transcript = _fake_transcript()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = write_transcript(transcript, Path(tmp) / "out" / "transcript.json")
        raw = out_path.read_text(encoding="utf-8")
        assert "слово ещё" in raw, "ensure_ascii=False: кириллица пишется как есть"
        assert "\\u" not in raw, raw[:200]
        assert '\n  "schema"' in raw, "отступ должен быть 2"
        assert raw.endswith("\n")
        assert json.loads(raw) == transcript


def _check_lazy_imports():
    # --selftest и --help обязаны работать без тяжёлых зависимостей, поэтому ни
    # torch, ни gigaam не должны оказаться загруженными к этому моменту.
    for module in ("torch", "torchaudio", "gigaam"):
        assert module not in sys.modules, f"{module} импортирован не лениво"


CHECKS = (
    ("сдвиг и округление пословных таймкодов", _check_shift_words),
    ("×RT: realtime_factor и realtime_factor_total", _check_realtime_factor),
    ("план: корректный проходит, пути разрешаются", _check_plan_ok),
    ("план: чужая схема", _check_plan_schema),
    ("план: чанк длиннее предела", _check_plan_too_long),
    ("план: пересечение чанков", _check_plan_overlap),
    ("план: end <= start", _check_plan_bad_bounds),
    ("план: файл чанка отсутствует", _check_plan_missing_file),
    ("преflight весов: полный путь в сообщении", _check_weights_preflight),
    ("сборка transcript/1: все обязательные ключи", _check_transcript_schema),
    ("пустой результат чанка не роняет прогон", _check_empty_segment),
    ("запись JSON: UTF-8, ensure_ascii=False, отступ 2", _check_write_transcript),
    ("torch и gigaam импортируются лениво", _check_lazy_imports),
)


def selftest():
    failed = 0
    for name, check in CHECKS:
        try:
            check()
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
        prog="python tools/transcribe.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Прогон GigaAM по готовому плану нарезки (chunk-plan/1).\n"
            "Пишет один JSON схемы transcript/1: сегменты, пословные таймкоды в\n"
            "абсолютном времени записи и метрики прогона (×RT, пик RSS, время).\n\n"
            "Веса не скачиваются: они обязаны уже лежать в --download-root,\n"
            "иначе инструмент падает с кодом 2 до загрузки модели."
        ),
        epilog=(
            "Примеры:\n"
            "  python tools/transcribe.py --plan data/chunks/plan.json \\\n"
            "      --out data/out/v3_rnnt.json\n"
            "  python tools/transcribe.py --plan data/chunks/plan.json \\\n"
            "      --model v3_e2e_rnnt --out data/out/v3_e2e_rnnt.json\n"
            "  python tools/transcribe.py --selftest\n\n"
            "Коды возврата: 0 — успех, 2 — непригодный вход (нет весов, битый\n"
            "или противоречивый план)."
        ),
    )
    parser.add_argument(
        "--plan", metavar="ПУТЬ", help="plan.json схемы chunk-plan/1 (обязателен для прогона)"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default=DEFAULT_MODEL,
        help=f"чекпойнт GigaAM (по умолчанию {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out", metavar="ПУТЬ", help="куда писать JSON транскрипта (обязателен для прогона)"
    )
    parser.add_argument(
        "--device",
        metavar="УСТРОЙСТВО",
        default=DEFAULT_DEVICE,
        help=f"устройство для torch (по умолчанию {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--download-root",
        metavar="КАТАЛОГ",
        default=DEFAULT_DOWNLOAD_ROOT,
        help=f"каталог с весами GigaAM (по умолчанию {DEFAULT_DOWNLOAD_ROOT})",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="прогнать встроенные проверки и выйти (без torch, gigaam и весов)",
    )
    parser.add_argument(
        "-h", "--help", action="help", help="показать эту справку и выйти"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        for name in ("plan", "out"):
            if not getattr(args, name):
                raise ToolError(f"не указан --{name} (см. --help)")
        return run(args)
    except ToolError as err:
        warn(str(err))
        return err.code


if __name__ == "__main__":
    raise SystemExit(main())
