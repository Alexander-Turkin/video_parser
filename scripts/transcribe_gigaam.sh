#!/usr/bin/env bash
#
# transcribe_gigaam.sh — единственная команда сквозного прогона GigaAM.
#
# Скрипт собирает в одном месте весь прогон: проверку окружения и весов,
# подготовку фрагмента, нарезку, распознавание обоими чекпойнтами, рендер
# .txt/.srt, проверки таймкодов и сравнительную таблицу. Смысл именно в этом:
# человеку нужна одна команда, вывод которой можно целиком положить в файл и
# потом разбирать — отдельные шаги уже разложены по tools/*.py, но помнить их
# порядок и флаги никто не обязан.
#
# Порядок шагов не произвольный, а требование протокола замера
# (docs/research/benchmark-plan.md, разделы 5 и 7):
#   * план нарезки строится РОВНО ОДИН РАЗ и переиспользуется всеми моделями —
#     иначе кандидаты получат разные чанки и сравнение скорости ничего не значит;
#   * модели прогоняются последовательно, а не параллельно, иначе они делят
#     процессор и память друг с другом;
#   * время и пик RSS снимаются /usr/bin/time -l (на macOS печатает maximum
#     resident set size в байтах), окружение и SHA-256 входа фиксируются в
#     выводе и в summary.md.
#
# Веса не скачиваются: оба чекпойнта должны уже лежать в --download-root.
# Отсутствие весов — понятная ошибка до начала прогона, а не падение на середине.
#
# Падение одной модели не прерывает вторую: ошибки собираются и печатаются
# статусом в конце, код возврата ненулевой, если хоть одна модель не дала
# выходов или не прошла проверки.
#
# Зависимости: ffmpeg, ffprobe, shasum, /usr/bin/time, awk и python с gigaam
# (берётся из ./.venv/bin/python, иначе python3). Сеть не используется.
# Результаты пишутся только под --outdir внутри data/, который закрыт корневым
# .gitignore, — текст лекции в репозиторий не попадает.
#
# Использование:
#   bash scripts/transcribe_gigaam.sh [опции]
#
# Запускать из корня репозитория. Пути с пробелами и кириллицей поддерживаются —
# передавайте в кавычках.

set -euo pipefail

INPUT="data/samples/Лекция_1_01.09.26_1_t4514_300s_16k_mono.wav"
OUTDIR="data/transcripts"
MODELS="v3_rnnt v3_e2e_rnnt"
DEVICE="cpu"
DOWNLOAD_ROOT="$HOME/.cache/gigaam"
SKIP_PREPARE=0
KEEP_CHUNKS=1

usage() {
  cat <<'EOF'
Использование: bash scripts/transcribe_gigaam.sh [опции]

Сквозной прогон GigaAM на одном фрагменте: нарезка по тишине, распознавание
каждым чекпойнтом, .txt и .srt, проверки таймкодов и сравнительная таблица.

Опции:
  --input <путь>          входной 16 кГц моно WAV
                          (по умолчанию data/samples/Лекция_1_01.09.26_1_t4514_300s_16k_mono.wav)
  --outdir <каталог>      куда писать план, чанки и выходы (по умолчанию data/transcripts)
  --models "<список>"     чекпойнты через пробел (по умолчанию "v3_rnnt v3_e2e_rnnt")
  --device <устройство>   устройство для torch (по умолчанию cpu)
  --download-root <кат.>  каталог с весами GigaAM (по умолчанию ~/.cache/gigaam)
  --skip-prepare          не готовить фрагмент через scripts/make_sample.sh,
                          даже если входного WAV нет (тогда это ошибка)
  --keep-chunks           оставить нарезанные чанки после прогона (по умолчанию)
  --no-chunks             удалить чанки после прогона, план оставить
  -h, --help              показать эту справку

Что появится в каталоге вывода:
  <outdir>/plan.json
  <outdir>/chunks/chunk_000.wav …
  <outdir>/<модель>/{transcript.json,transcript.txt,transcript.srt,time.txt,checks.json,checks.txt}
  <outdir>/summary.md

Код возврата: 0 — все модели дали выходы и прошли проверки таймкодов,
1 — хотя бы одна не дала.

Запуск с сохранением лога (mkdir обязателен: без каталога tee падает):
  mkdir -p data/transcripts && bash scripts/transcribe_gigaam.sh 2>&1 | tee "data/transcripts/run-$(date +%Y%m%d-%H%M%S).log"
EOF
}

die() {
  printf 'transcribe_gigaam.sh: %s\n' "$1" >&2
  exit 1
}

warn() {
  printf 'transcribe_gigaam.sh: %s\n' "$1" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --input)         [ $# -ge 2 ] || die "--input требует путь";           INPUT="$2";         shift 2 ;;
    --outdir)        [ $# -ge 2 ] || die "--outdir требует каталог";       OUTDIR="$2";        shift 2 ;;
    --models)        [ $# -ge 2 ] || die "--models требует список";        MODELS="$2";        shift 2 ;;
    --device)        [ $# -ge 2 ] || die "--device требует значение";      DEVICE="$2";        shift 2 ;;
    --download-root) [ $# -ge 2 ] || die "--download-root требует каталог"; DOWNLOAD_ROOT="$2"; shift 2 ;;
    --skip-prepare)  SKIP_PREPARE=1; shift ;;
    --keep-chunks)   KEEP_CHUNKS=1;  shift ;;
    --no-chunks)     KEEP_CHUNKS=0;  shift ;;
    --) shift; [ $# -eq 0 ] || die "после -- аргументы не ожидаются" ;;
    *) die "неизвестная опция: $1 (см. --help)" ;;
  esac
done

[ -n "$MODELS" ] || die "--models пуст: нечего прогонять"

# Разбор списка моделей в массив: имена чекпойнтов без пробелов, поэтому
# словоделение здесь уместно, а дальше имена уже не разъезжаются.
MODEL_LIST=()
read -r -a MODEL_LIST <<<"$MODELS"
[ "${#MODEL_LIST[@]}" -gt 0 ] || die "--models пуст: нечего прогонять"

# ---------------------------------------------------------------- хелперы ---

# Байты -> человекочитаемый размер (как в scripts/make_sample.sh).
fmt_bytes() {
  awk -v b="${1:-}" 'BEGIN {
    if (b == "") { print "n/a"; exit }
    n = split("B KiB MiB GiB TiB", u, " "); i = 1
    while (b >= 1024 && i < n) { b /= 1024; i++ }
    f = (i == 1 ? "%d %s\n" : "%.2f %s\n")
    printf f, b, u[i]
  }'
}

# Версия установленного пакета без его импорта: importlib.metadata читает
# метаданные дистрибутива, а import torch стоил бы несколько секунд на каждый
# вызов и тянул бы в память весь фреймворк ради одной строки.
pkg_version() {
  "$PY" - "$1" <<'PY' 2>/dev/null || printf 'неизвестно\n'
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    print("не установлен")
PY
}

# Файл токенизатора нужен не всем чекпойнтам: gigaam докачивает его только для
# v1_rnnt и моделей с e2e в имени (см. _download_tokenizer в пакете).
needs_tokenizer() {
  case "$1" in
    *e2e*|v1_rnnt) return 0 ;;
    *) return 1 ;;
  esac
}

# --------------------------------------------------- проверка окружения ---

[ -f tools/transcribe.py ] || die "не найден tools/transcribe.py — запускайте скрипт из корня репозитория"

if [ -x ./.venv/bin/python ]; then
  PY="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  die "не найден python: нет ни ./.venv/bin/python, ни python3 в PATH"
fi

command -v ffmpeg  >/dev/null 2>&1 || die "не найден ffmpeg (установите: brew install ffmpeg)"
command -v ffprobe >/dev/null 2>&1 || die "не найден ffprobe (установите: brew install ffmpeg)"
command -v shasum  >/dev/null 2>&1 || die "не найден shasum"

# gigaam проверяется по наличию модуля, а не импортом: импорт тянет torch и стоит
# секунды, а нам нужно лишь понять, тот ли это интерпретатор.
"$PY" - <<'PY' || die "в интерпретаторе $PY нет пакета gigaam; используйте python из .venv проекта (ожидается ./.venv/bin/python)"
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("gigaam") else 1)
PY

TIME_BIN="/usr/bin/time"
if [ ! -x "$TIME_BIN" ]; then
  warn "нет $TIME_BIN — прогон пойдёт без него, пик RSS будет взят из метрик transcribe.py"
  TIME_BIN=""
fi

for model in "${MODEL_LIST[@]}"; do
  ckpt="$DOWNLOAD_ROOT/$model.ckpt"
  [ -f "$ckpt" ] || die "нет весов чекпойнта: $ckpt
  Скачайте их заранее вручную и положите в $DOWNLOAD_ROOT — скрипт ничего не качает.
  Передать другой каталог: --download-root <каталог>."
  if needs_tokenizer "$model"; then
    tok="$DOWNLOAD_ROOT/${model}_tokenizer.model"
    [ -f "$tok" ] || die "нет токенизатора для $model: $tok
  Чекпойнты v1_rnnt и *e2e* без него не работают. Положите файл рядом с $model.ckpt."
  fi
done

# ----------------------------------------------- подготовка фрагмента ---

printf '================ СКВОЗНОЙ ПРОГОН GIGAAM ================\n'
printf 'Снято:         %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"

if [ ! -f "$INPUT" ]; then
  if [ "$SKIP_PREPARE" -eq 1 ]; then
    die "входного файла нет, а --skip-prepare запрещает его готовить: $INPUT"
  fi
  printf '\n== Фрагмента нет, готовлю через scripts/make_sample.sh\n'
  bash scripts/make_sample.sh || die "не удалось подготовить фрагмент"
  [ -f "$INPUT" ] || die "make_sample.sh отработал, но ожидаемого файла нет: $INPUT
  Передайте фактический путь через --input."
fi

[ -r "$INPUT" ] || die "файл недоступен для чтения: $INPUT"

mkdir -p "$OUTDIR"

# ------------------------------------------------------ блок окружения ---

mac_model="$(sysctl -n hw.model 2>/dev/null || printf 'неизвестно')"
mac_cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf 'неизвестно')"
arch="$(uname -m)"
os_name="$(sw_vers -productName 2>/dev/null || printf 'неизвестно')"
os_version="$(sw_vers -productVersion 2>/dev/null || printf 'неизвестно')"
os_build="$(sw_vers -buildVersion 2>/dev/null || printf 'неизвестно')"
# Версия ffmpeg — первая строка `ffmpeg -version`, третье слово в ней.
ffmpeg_version="$(ffmpeg -hide_banner -version 2>/dev/null | awk 'NR == 1 { print $3 }')"
py_version="$("$PY" -V 2>&1 | awk '{ print $2 }')"
torch_version="$(pkg_version torch)"
torchaudio_version="$(pkg_version torchaudio)"
gigaam_version="$(pkg_version gigaam)"
input_size="$(stat -f%z "$INPUT" 2>/dev/null || printf '')"
input_sha="$(shasum -a 256 "$INPUT" | awk '{ print $1 }')"
input_duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 -i "$INPUT" 2>/dev/null | awk 'NR == 1 && $1 != "N/A" { printf "%.3f", $1 }' || printf '')"

printf '\n== Окружение\n'
printf '  Mac:            %s (%s, %s)\n' "$mac_model" "$mac_cpu" "$arch"
printf '  ОС:             %s %s (%s)\n' "$os_name" "$os_version" "$os_build"
printf '  ffmpeg:         %s\n' "${ffmpeg_version:-неизвестно}"
printf '  python:         %s (%s)\n' "${py_version:-неизвестно}" "$PY"
printf '  torch:          %s\n' "$torch_version"
printf '  torchaudio:     %s\n' "$torchaudio_version"
printf '  gigaam:         %s\n' "$gigaam_version"
printf '  устройство:     %s\n' "$DEVICE"
printf '  /usr/bin/time:  %s\n' "${TIME_BIN:-нет, пик RSS из метрик transcribe.py}"

printf '\n== Чекпойнты (веса из %s)\n' "$DOWNLOAD_ROOT"
for model in "${MODEL_LIST[@]}"; do
  ckpt="$DOWNLOAD_ROOT/$model.ckpt"
  size="$(stat -f%z "$ckpt" 2>/dev/null || printf '')"
  printf '  %-14s %s  (%s)\n' "$model" "$ckpt" "$(fmt_bytes "$size")"
  if needs_tokenizer "$model"; then
    tok="$DOWNLOAD_ROOT/${model}_tokenizer.model"
    size="$(stat -f%z "$tok" 2>/dev/null || printf '')"
    printf '  %-14s %s  (%s)\n' "" "$tok" "$(fmt_bytes "$size")"
  fi
done

# SHA-256 входа печатается отдельной строкой: протокол замера требует
# подтверждать, что все кандидаты получили побайтово одинаковый вход.
printf '\n== Вход\n'
printf '  файл:          %s\n' "$INPUT"
printf '  размер:        %s (%s байт)\n' "$(fmt_bytes "$input_size")" "${input_size:-n/a}"
printf '  длительность:  %s с\n' "${input_duration:-неизвестно}"
printf '  SHA-256:       %s\n' "$input_sha"

# ---------------------------------------------------------- план нарезки ---

PLAN="$OUTDIR/plan.json"

printf '\n================ ПЛАН НАРЕЗКИ ================\n'
printf '$ %s tools/chunker.py --input %s --outdir %s\n' "$PY" "$INPUT" "$OUTDIR"
"$PY" tools/chunker.py --input "$INPUT" --outdir "$OUTDIR" \
  || die "нарезка не удалась, прогон моделей не начинаем"
[ -f "$PLAN" ] || die "chunker.py отработал, но плана нет: $PLAN"

# Сводка плана считается из самого plan.json, а не из вывода нарезчика: в таблицу
# и в анализ идут те же числа, которые реально прочитает transcribe.py.
plan_summary() {
  "$PY" - "$1" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chunks = plan.get("chunks") or []
durations = [chunk["duration"] for chunk in chunks]
total = plan.get("source_duration")
coverage = plan.get("coverage_seconds")

share = ""
if total and coverage is not None:
    share = f" ({coverage / total * 100:.1f} %)"

print(f"  чанков:             {len(chunks)}")
print(f"  покрытие:           {coverage} с из {total} с записи{share}")
print(
    f"  выброшено паузами:  {plan.get('dropped_seconds')} с "
    f"в {len(plan.get('dropped') or [])} интервалах"
)
if durations:
    print(f"  длительность чанка: мин {min(durations):.3f} с, макс {max(durations):.3f} с")
print(f"  параметры:          {json.dumps(plan.get('params') or {}, ensure_ascii=False)}")
PY
}

printf '\n== Сводка плана\n'
plan_summary "$PLAN" || die "не удалось прочитать план: $PLAN"

# ----------------------------------------------------- прогон кандидатов ---

# Статусы собираются в массивы, а не печатаются сразу: падение одной модели не
# должно обрывать вторую, а итог нужен цельным в конце прогона.
STATUS_MODEL=()
STATUS_TEXT=()
FAILED=0

run_model() {
  local model="$1"
  local dir="$OUTDIR/$model"
  local transcript="$dir/transcript.json"
  local time_txt="$dir/time.txt"
  local checks_json="$dir/checks.json"
  local checks_txt="$dir/checks.txt"
  local rc=0

  mkdir -p "$dir"

  local prefix=""
  [ -z "$TIME_BIN" ] || prefix="$TIME_BIN -l "

  printf '\n================ МОДЕЛЬ %s ================\n' "$model"
  printf '$ %s%s tools/transcribe.py --plan %s --model %s --out %s --device %s --download-root %s\n' \
    "$prefix" "$PY" "$PLAN" "$model" "$transcript" "$DEVICE" "$DOWNLOAD_ROOT"

  # stderr уводится в time.txt целиком и печатается ниже: кроме блока
  # /usr/bin/time там предупреждения о пустых чанках и ворнинги torch, терять их
  # в логе прогона нельзя.
  if [ -n "$TIME_BIN" ]; then
    "$TIME_BIN" -l "$PY" tools/transcribe.py --plan "$PLAN" --model "$model" \
      --out "$transcript" --device "$DEVICE" --download-root "$DOWNLOAD_ROOT" \
      2>"$time_txt" || rc=$?
  else
    "$PY" tools/transcribe.py --plan "$PLAN" --model "$model" \
      --out "$transcript" --device "$DEVICE" --download-root "$DOWNLOAD_ROOT" \
      2>"$time_txt" || rc=$?
    printf 'прогон шёл без /usr/bin/time\n' >>"$time_txt"
  fi

  printf '\n-- stderr и /usr/bin/time -l (%s)\n' "$time_txt"
  cat "$time_txt"

  if [ "$rc" -ne 0 ]; then
    STATUS_MODEL+=("$model")
    STATUS_TEXT+=("ПРОВАЛ — transcribe.py вернул код $rc")
    FAILED=1
    return 0
  fi
  if [ ! -s "$transcript" ]; then
    STATUS_MODEL+=("$model")
    STATUS_TEXT+=("ПРОВАЛ — transcribe.py не создал $transcript")
    FAILED=1
    return 0
  fi

  printf '\n-- пик RSS из %s\n' "$time_txt"
  awk '/maximum resident set size/ {
    printf "  пик RSS: %.2f ГБ (%d байт)\n", $1 / 1073741824, $1; found = 1
  }
  END { if (!found) print "  пик RSS: в выводе нет строки maximum resident set size" }' "$time_txt"

  printf '\n$ %s tools/outputs.py --transcript %s --txt %s --srt %s\n' \
    "$PY" "$transcript" "$dir/transcript.txt" "$dir/transcript.srt"
  rc=0
  "$PY" tools/outputs.py --transcript "$transcript" \
    --txt "$dir/transcript.txt" --srt "$dir/transcript.srt" || rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$dir/transcript.txt" ] || [ ! -s "$dir/transcript.srt" ]; then
    STATUS_MODEL+=("$model")
    STATUS_TEXT+=("ПРОВАЛ — outputs.py не дал .txt и .srt (код $rc)")
    FAILED=1
    return 0
  fi

  printf '\n$ %s tools/check_timecodes.py --transcript %s --srt %s --json %s\n' \
    "$PY" "$transcript" "$dir/transcript.srt" "$checks_json"
  rc=0
  # Вывод проверок сохраняется и печатается: он часть отчёта, а не диагностика
  # на один раз. Пайп в tee здесь не годится — потерялся бы код возврата проверок.
  "$PY" tools/check_timecodes.py --transcript "$transcript" \
    --srt "$dir/transcript.srt" --json "$checks_json" >"$checks_txt" 2>&1 || rc=$?
  cat "$checks_txt"

  if [ "$rc" -ne 0 ]; then
    STATUS_MODEL+=("$model")
    STATUS_TEXT+=("ПРОВАЛ — проверки таймкодов не прошли (код $rc), см. $checks_txt")
    FAILED=1
    return 0
  fi
  if [ ! -s "$checks_json" ]; then
    STATUS_MODEL+=("$model")
    STATUS_TEXT+=("ПРОВАЛ — нет машинного итога проверок: $checks_json")
    FAILED=1
    return 0
  fi

  STATUS_MODEL+=("$model")
  STATUS_TEXT+=("ок — выходы на месте, проверки таймкодов пройдены")
  return 0
}

for model in "${MODEL_LIST[@]}"; do
  run_model "$model"
done

# ---------------------------------------------------------- чанки и итог ---

if [ "$KEEP_CHUNKS" -eq 0 ]; then
  printf '\n== Удаляю чанки (--no-chunks), план остаётся\n'
  # Удаляются файлы по маске нарезчика, а не по списку из плана: от предыдущего
  # прогона с другими параметрами в каталоге остаются чанки с номерами, которых в
  # текущем плане уже нет, и по плану их было бы не найти. Маска при этом узкая —
  # ничего, кроме chunk_*.wav, скрипт не трогает.
  removed=0
  for chunk_file in "$OUTDIR"/chunks/chunk_*.wav; do
    [ -f "$chunk_file" ] || continue
    rm -f -- "$chunk_file"
    removed=$((removed + 1))
  done
  printf '  удалено файлов: %d\n' "$removed"
  if [ -d "$OUTDIR/chunks" ] && ! rmdir "$OUTDIR/chunks" 2>/dev/null; then
    printf '  каталог %s/chunks оставлен: в нём есть ещё что-то, кроме чанков\n' "$OUTDIR"
  fi
fi

SUMMARY="$OUTDIR/summary.md"
ENV_LINE="$mac_model ($mac_cpu, $arch), $os_name $os_version ($os_build), ffmpeg ${ffmpeg_version:-неизвестно}, python ${py_version:-неизвестно}, torch $torch_version, torchaudio $torchaudio_version, gigaam $gigaam_version, устройство $DEVICE"
INPUT_LINE="\`$INPUT\`, ${input_duration:-?} с, SHA-256 \`$input_sha\`"

printf '\n================ СВОДНАЯ ТАБЛИЦА ================\n'

# Таблица собирается из transcript.json, checks.json и time.txt — то есть из
# файлов прогона, а не из перехваченного stdout: при повторном разборе выводов
# числа обязаны сходиться с тем, что лежит на диске.
"$PY" - "$OUTDIR" "$SUMMARY" "$ENV_LINE" "$INPUT_LINE" "$PLAN" "${MODEL_LIST[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
env_line = sys.argv[3]
input_line = sys.argv[4]
plan_path = Path(sys.argv[5])
models = sys.argv[6:]

DASH = "—"
PUNCT = ".,!?"
WORD_CHARS = re.compile(r"[^\W\d_]", re.UNICODE)


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def peak_rss_gb(model, transcript):
    """Пик RSS из вывода /usr/bin/time -l, иначе из метрик transcribe.py."""
    time_txt = outdir / model / "time.txt"
    try:
        text = time_txt.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        if "maximum resident set size" in line:
            head = line.split()
            if head and head[0].isdigit():
                return int(head[0]) / 1024**3
    if transcript:
        rss = transcript.get("metrics", {}).get("peak_rss_bytes")
        if rss:
            return rss / 1024**3
    return None


def text_stats(text):
    """Знаки .,!? и доля слов с заглавной буквы — косвенный признак пунктуации."""
    punct = sum(text.count(sign) for sign in PUNCT)
    words = [token for token in text.split() if WORD_CHARS.search(token)]
    capitalized = 0
    for token in words:
        letters = [ch for ch in token if WORD_CHARS.match(ch)]
        if letters and letters[0].isupper():
            capitalized += 1
    share = (capitalized / len(words) * 100) if words else None
    return punct, share


def checks_line(model):
    report = load_json(outdir / model / "checks.json")
    if not report:
        return DASH
    items = report.get("checks") or []
    passed = sum(1 for item in items if item.get("passed"))
    mark = "пройдено" if report.get("passed") else "ПРОВАЛ"
    return f"{mark} {passed} из {len(items)}"


def num(value, fmt):
    return DASH if value is None else format(value, fmt)


rows = []
for model in models:
    transcript = load_json(outdir / model / "transcript.json")
    if not transcript:
        rows.append([f"`{model}`"] + [DASH] * 9 + ["прогон не удался"])
        continue
    metrics = transcript.get("metrics", {})
    punct, cap_share = text_stats(transcript.get("text") or "")
    rows.append(
        [
            f"`{model}`",
            num(metrics.get("model_load_seconds"), ".2f"),
            num(metrics.get("transcribe_seconds"), ".2f"),
            num(metrics.get("realtime_factor"), ".2f"),
            num(peak_rss_gb(model, transcript), ".2f"),
            str(metrics.get("chunks", DASH)),
            str(metrics.get("words", DASH)),
            str(metrics.get("chars", DASH)),
            str(punct),
            num(cap_share, ".1f"),
            checks_line(model),
        ]
    )

header = [
    "Модель",
    "Загрузка, с",
    "Распознавание, с",
    "×RT",
    "Пик RSS, ГБ",
    "Чанков",
    "Слов",
    "Символов",
    "Знаков .,!?",
    "Слов с заглавной, %",
    "Проверки таймкодов",
]

plan = load_json(plan_path) or {}
lines = [
    "# Сравнение чекпойнтов GigaAM на одном фрагменте",
    "",
    f"**Окружение:** {env_line}.",
    "",
    f"**Вход:** {input_line}.",
    "",
    (
        f"**Нарезка:** {len(plan.get('chunks') or [])} чанков, покрытие "
        f"{plan.get('coverage_seconds')} с, выброшено паузами "
        f"{plan.get('dropped_seconds')} с; план `{plan_path.as_posix()}` "
        f"один и тот же для всех моделей."
    ),
    "",
    "| " + " | ".join(header) + " |",
    "| " + " | ".join("---" for _ in header) + " |",
]
lines += ["| " + " | ".join(row) + " |" for row in rows]
lines += [
    "",
    "×RT = длительность распознанного аудио / время распознавания "
    "(docs/research/benchmark-plan.md, раздел 5). Пик RSS — "
    "maximum resident set size из `/usr/bin/time -l`. Доля слов с заглавной "
    "буквы и число знаков `.,!?` — косвенный признак пунктуации, не метрика "
    "качества: WER в этом прогоне не считался.",
    "",
]
payload = "\n".join(lines)
summary_path.write_text(payload, encoding="utf-8")
print(payload, end="")
PY

printf '\nТаблица сохранена: %s\n' "$SUMMARY"

printf '\n================ СТАТУС ПО МОДЕЛЯМ ================\n'
index=0
while [ "$index" -lt "${#STATUS_MODEL[@]}" ]; do
  printf '  %-14s %s\n' "${STATUS_MODEL[$index]}" "${STATUS_TEXT[$index]}"
  index=$((index + 1))
done

printf '\n== Файлы прогона\n'
printf '  %s\n' "$PLAN"
[ "$KEEP_CHUNKS" -eq 0 ] || printf '  %s/chunks/\n' "$OUTDIR"
for model in "${MODEL_LIST[@]}"; do
  for name in transcript.json transcript.txt transcript.srt time.txt checks.json checks.txt; do
    if [ -s "$OUTDIR/$model/$name" ]; then
      printf '  %s/%s/%s\n' "$OUTDIR" "$model" "$name"
    else
      printf '  %s/%s/%s — НЕТ\n' "$OUTDIR" "$model" "$name"
    fi
  done
done
printf '  %s\n' "$SUMMARY"

if [ "$FAILED" -ne 0 ]; then
  printf '\n================ ЕСТЬ ПРОВАЛЫ ================\n'
  exit 1
fi

printf '\n================ ГОТОВО ================\n'
