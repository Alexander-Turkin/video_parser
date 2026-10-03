#
# common.sh — общая часть прогонных скриптов GigaAM.
#
# Прогонов в проекте больше одного: бенчмарк двух чекпойнтов
# (scripts/transcribe_gigaam.sh) и продуктовая расшифровка записи целиком. Вход и
# выход у них разные, а подготовка одинаковая: найти python из .venv, убедиться,
# что на машине есть ffmpeg и веса, напечатать окружение и описание входа. Эти
# куски и живут здесь — чтобы при правке сообщения или порядка строк не
# приходилось помнить, в скольких скриптах лежит их копия, и чтобы логи разных
# прогонов оставались сопоставимыми между собой.
#
# Библиотека ничего не делает при подключении: только определяет функции и
# значения по умолчанию. Никаких проверок, вызовов ffmpeg и exit при source —
# решение, что и в каком порядке проверять, принимает вызывающий скрипт.
# Единственное место с exit — функция die.
#
# Подключение (путь относительно корня репозитория, запускать из корня):
#   source scripts/lib/common.sh
#
# Проверить to_wav на настоящем файле можно прямым запуском — как у инструментов
# tools/*.py, у которых для этого есть --selftest:
#   bash scripts/lib/common.sh --selftest [медиафайл] [каталог вывода]
# Без --selftest прямой запуск тоже ничего не делает.
#
# Соглашения:
#   * SCRIPT_NAME — префикс сообщений об ошибках; задайте его до подключения или
#     сразу после, иначе берётся имя запущенного скрипта;
#   * PY — интерпретатор python, которым запускаются tools/*.py и служебные
#     вставки. Заполняется вызовом resolve_python: PY="$(resolve_python)".
#     Функции pkg_version, require_python_gigaam, print_env_block и plan_summary
#     берут интерпретатор отсюда.
#
# Зависимости: ffmpeg, ffprobe, shasum, awk и стандартные утилиты macOS. Сеть не
# используется, веса не скачиваются, во временные каталоги библиотека не пишет.
#
# Пути с пробелами и кириллицей поддерживаются — передавайте в кавычках.

# Имя скрипта в сообщениях. ${0##*/} — чистая подстановка, без запуска basename:
# при подключении библиотека не должна порождать процессы.
: "${SCRIPT_NAME:=${0##*/}}"

# Интерпретатор python; заполняется resolve_python. Объявлен здесь, чтобы
# `set -u` в вызывающем скрипте не падал на проверках ниже.
: "${PY:=}"

# Фактический путь к WAV после to_wav: совпадает с выходом, если файл
# конвертировали, и со входом, если он уже был 16 кГц моно WAV.
: "${TO_WAV_PATH:=}"

# ------------------------------------------------------------- сообщения ---

# Сообщение об ошибке в stderr и выход с кодом 1. Единственное место в
# библиотеке, где есть exit.
die() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
  exit 1
}

# Предупреждение в stderr: прогон продолжается.
warn() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
}

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

# Длительность файла в секундах с тремя знаками, пустая строка — если определить
# не удалось. В awk намеренно нет `exit`: он закрыл бы пайп раньше, чем ffprobe
# допишет вывод, и `pipefail` уронил бы скрипт по SIGPIPE. Завершающий
# `|| printf ''` гасит ненулевой код пайплайна по той же причине.
probe_duration() {
  ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 -i "$1" 2>/dev/null \
    | awk 'NR == 1 && $1 != "N/A" { printf "%.3f", $1 }' || printf ''
}

# SHA-256 файла одной строкой, без имени файла.
file_sha256() {
  shasum -a 256 "$1" | awk '{ print $1 }'
}

# Файл токенизатора нужен не всем чекпойнтам: gigaam докачивает его только для
# v1_rnnt и моделей с e2e в имени (см. _download_tokenizer в пакете).
needs_tokenizer() {
  case "$1" in
    *e2e*|v1_rnnt) return 0 ;;
    *) return 1 ;;
  esac
}

# --------------------------------------------------------- проверки среды ---

# Печатает путь к интерпретатору: ./.venv/bin/python, если он есть, иначе
# python3 из PATH. Вызывать так: PY="$(resolve_python)" — при отсутствии обоих
# die печатает ошибку в stderr и подстановка возвращает ненулевой код.
resolve_python() {
  if [ -x ./.venv/bin/python ]; then
    printf './.venv/bin/python\n'
  elif command -v python3 >/dev/null 2>&1; then
    printf 'python3\n'
  else
    die "не найден python: нет ни ./.venv/bin/python, ни python3 в PATH"
  fi
}

# Внешние утилиты, без которых не идёт ни один прогон.
require_tools() {
  command -v ffmpeg  >/dev/null 2>&1 || die "не найден ffmpeg (установите: brew install ffmpeg)"
  command -v ffprobe >/dev/null 2>&1 || die "не найден ffprobe (установите: brew install ffmpeg)"
  command -v shasum  >/dev/null 2>&1 || die "не найден shasum"
}

# Проверяет, что в интерпретаторе (аргумент или $PY) есть пакет gigaam.
# Проверка по наличию модуля, а не импортом: импорт тянет torch и стоит секунды,
# а нам нужно лишь понять, тот ли это интерпретатор.
require_python_gigaam() {
  local py="${1:-$PY}"
  [ -n "$py" ] || die "не задан python: вызовите resolve_python до require_python_gigaam"
  "$py" - <<'PY' || die "в интерпретаторе $py нет пакета gigaam; используйте python из .venv проекта (ожидается ./.venv/bin/python)"
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("gigaam") else 1)
PY
}

# require_weights <модель> <download-root>
# Веса не скачиваются: чекпойнт должен уже лежать в каталоге. Отсутствие файла —
# понятная ошибка до начала прогона, а не падение на середине. Для v1_rnnt и
# моделей с e2e в имени дополнительно проверяется файл токенизатора.
require_weights() {
  local model="$1" root="$2" ckpt tok
  ckpt="$root/$model.ckpt"
  [ -f "$ckpt" ] || die "нет весов чекпойнта: $ckpt
  Скачайте их заранее вручную и положите в $root — скрипт ничего не качает.
  Передать другой каталог: --download-root <каталог>."
  if needs_tokenizer "$model"; then
    tok="$root/${model}_tokenizer.model"
    [ -f "$tok" ] || die "нет токенизатора для $model: $tok
  Чекпойнты v1_rnnt и *e2e* без него не работают. Положите файл рядом с $model.ckpt."
  fi
}

# Версия установленного пакета без его импорта: importlib.metadata читает
# метаданные дистрибутива, а import torch стоил бы несколько секунд на каждый
# вызов и тянул бы в память весь фреймворк ради одной строки.
pkg_version() {
  [ -n "$PY" ] || die "не задан PY: вызовите resolve_python до pkg_version"
  "$PY" - "$1" <<'PY' 2>/dev/null || printf 'неизвестно\n'
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    print("не установлен")
PY
}

# ------------------------------------------------------------ блоки вывода ---

# print_env_block <устройство> [строка про /usr/bin/time]
# Печатает блок «== Окружение» и попутно складывает те же значения в глобальные
# переменные ENV_*: их потом вставляют в summary.md, и пересчитывать их второй
# раз (а значит, рисковать расхождением с напечатанным) незачем.
# Последняя строка про /usr/bin/time печатается, только если аргумент задан:
# замер времени нужен бенчмарку, а продуктовому прогону — нет.
print_env_block() {
  local device="$1" time_note="${2:-}"

  ENV_MAC_MODEL="$(sysctl -n hw.model 2>/dev/null || printf 'неизвестно')"
  ENV_MAC_CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf 'неизвестно')"
  ENV_ARCH="$(uname -m)"
  ENV_OS_NAME="$(sw_vers -productName 2>/dev/null || printf 'неизвестно')"
  ENV_OS_VERSION="$(sw_vers -productVersion 2>/dev/null || printf 'неизвестно')"
  ENV_OS_BUILD="$(sw_vers -buildVersion 2>/dev/null || printf 'неизвестно')"
  # Версия ffmpeg — первая строка `ffmpeg -version`, третье слово в ней.
  ENV_FFMPEG_VERSION="$(ffmpeg -hide_banner -version 2>/dev/null | awk 'NR == 1 { print $3 }')"
  ENV_PY_VERSION="$("$PY" -V 2>&1 | awk '{ print $2 }')"
  ENV_TORCH_VERSION="$(pkg_version torch)"
  ENV_TORCHAUDIO_VERSION="$(pkg_version torchaudio)"
  ENV_GIGAAM_VERSION="$(pkg_version gigaam)"
  : "${ENV_FFMPEG_VERSION:=неизвестно}"
  : "${ENV_PY_VERSION:=неизвестно}"

  printf '\n== Окружение\n'
  printf '  Mac:            %s (%s, %s)\n' "$ENV_MAC_MODEL" "$ENV_MAC_CPU" "$ENV_ARCH"
  printf '  ОС:             %s %s (%s)\n' "$ENV_OS_NAME" "$ENV_OS_VERSION" "$ENV_OS_BUILD"
  printf '  ffmpeg:         %s\n' "$ENV_FFMPEG_VERSION"
  printf '  python:         %s (%s)\n' "$ENV_PY_VERSION" "$PY"
  printf '  torch:          %s\n' "$ENV_TORCH_VERSION"
  printf '  torchaudio:     %s\n' "$ENV_TORCHAUDIO_VERSION"
  printf '  gigaam:         %s\n' "$ENV_GIGAAM_VERSION"
  printf '  устройство:     %s\n' "$device"
  [ -z "$time_note" ] || printf '  /usr/bin/time:  %s\n' "$time_note"
}

# print_input_block <путь>
# Печатает блок «== Вход» и складывает значения в INPUT_SIZE, INPUT_DURATION,
# INPUT_SHA. SHA-256 печатается отдельной строкой не для красоты: протокол
# замера (docs/research/benchmark-plan.md) требует подтверждать, что все
# кандидаты получили побайтово одинаковый вход.
print_input_block() {
  local path="$1"

  INPUT_SIZE="$(stat -f%z "$path" 2>/dev/null || printf '')"
  INPUT_SHA="$(file_sha256 "$path")"
  INPUT_DURATION="$(probe_duration "$path")"

  printf '\n== Вход\n'
  printf '  файл:          %s\n' "$path"
  printf '  размер:        %s (%s байт)\n' "$(fmt_bytes "$INPUT_SIZE")" "${INPUT_SIZE:-n/a}"
  printf '  длительность:  %s с\n' "${INPUT_DURATION:-неизвестно}"
  printf '  SHA-256:       %s\n' "$INPUT_SHA"
}

# plan_summary <plan.json> <режим нарезки>
# Сводка считается из самого plan.json, а не из вывода нарезчика: в таблицу и в
# анализ идут те же числа, которые реально прочитает transcribe.py.
plan_summary() {
  [ -n "$PY" ] || die "не задан PY: вызовите resolve_python до plan_summary"
  "$PY" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
chunks = plan.get("chunks") or []
durations = [chunk["duration"] for chunk in chunks]
total = plan.get("source_duration")
coverage = plan.get("coverage_seconds")

share = ""
if total and coverage is not None:
    share = f" ({coverage / total * 100:.1f} %)"

print(f"  режим нарезки:      {mode}")
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

# --------------------------------------------------------- подготовка WAV ---

# to_wav <вход> <выход>
# Приводит любой читаемый ffmpeg медиафайл (webm, mp4, mp3, m4a, wav) к 16 кГц
# моно pcm_s16le целиком — без вырезания фрагмента. Именно в этот формат Whisper,
# GigaAM, Vosk и NeMo ресемплируют вход внутри себя, поэтому приведение делается
# один раз здесь. Фрагмент из середины режет scripts/make_sample.sh — это другая
# задача, и смешивать их нельзя.
#
# Если вход уже 16 кГц моно pcm_s16le WAV, он используется как есть: перегонять
# файл сам в себя — это лишний проход по гигабайтам и лишний повод для
# расхождения хешей. Фактический путь к WAV возвращается в TO_WAV_PATH: он равен
# входу в этом случае и выходу — во всех остальных.
to_wav() {
  local src="$1" dst="$2" probe codec rate channels container dur

  [ -f "$src" ] || die "файл не найден: $src"
  [ -r "$src" ] || die "файл недоступен для чтения: $src"

  # ffprobe на файле без звука завершается успешно и просто ничего не печатает,
  # поэтому пустой ответ — это и есть «аудиодорожки нет». Проверяем до ffmpeg:
  # иначе человек получил бы ошибку про несопоставленный поток 0:a:0.
  probe="$(ffprobe -v error -select_streams a:0 \
    -show_entries stream=codec_name,sample_rate,channels \
    -of default=nw=1 -i "$src" 2>/dev/null || printf '')"
  [ -n "$probe" ] \
    || die "в файле нет аудиодорожки, готовить WAV не из чего: $src"

  codec="$(awk -F= '$1 == "codec_name" { print $2; exit }' <<<"$probe")"
  rate="$(awk -F= '$1 == "sample_rate" { print $2; exit }' <<<"$probe")"
  channels="$(awk -F= '$1 == "channels" { print $2; exit }' <<<"$probe")"
  container="$(ffprobe -v error -show_entries format=format_name -of default=nw=1:nk=1 \
    -i "$src" 2>/dev/null | awk 'NR == 1 { print }' || printf '')"

  if [ "$codec" = "pcm_s16le" ] && [ "$rate" = "16000" ] && [ "$channels" = "1" ] \
     && [ "$container" = "wav" ]; then
    printf '  вход уже 16 кГц моно WAV (pcm_s16le) — перекодирование не нужно, беру как есть\n'
    TO_WAV_PATH="$src"
  else
    printf '  конвертирую в 16 кГц моно pcm_s16le: %s\n' "$dst"
    mkdir -p "$(dirname "$dst")"
    # -map 0:a:0 фиксирует первую аудиодорожку: если их в записи несколько,
    # выбор не должен зависеть от эвристики ffmpeg. -vn не даёт декодировать
    # видео, которого в WAV всё равно не будет. -ss и -t не передаются
    # намеренно: обрабатывается запись целиком.
    ffmpeg -nostdin -hide_banner -v error -i "$src" \
      -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le -y "$dst" \
      || die "ffmpeg не смог привести файл к 16 кГц моно WAV: $src"
    [ -s "$dst" ] || die "ffmpeg не создал файл или он пуст: $dst"
    TO_WAV_PATH="$dst"
  fi

  dur="$(probe_duration "$TO_WAV_PATH")"
  printf '  файл:          %s\n' "$TO_WAV_PATH"
  printf '  длительность:  %s с\n' "${dur:-неизвестно}"
  printf '  SHA-256:       %s\n' "$(file_sha256 "$TO_WAV_PATH")"
}

# ------------------------------------------------------------- селф-тест ---

# selftest_to_wav <медиафайл> <каталог вывода>
# Проверка to_wav на настоящем файле. Тестового фреймворка для shell в проекте
# нет, зато у всех инструментов tools/*.py есть --selftest — здесь тот же приём и
# то же назначение: прогнать код на реальных данных одной командой, не заводя
# отдельного скрипта. Четыре случая: конвертация, путь с пробелами и кириллицей,
# уже готовый 16 кГц моно WAV и файл без аудиодорожки.
# Исходник по умолчанию готовит scripts/make_sample.sh, результаты пишутся в
# data/ — оба каталога закрыты корневым .gitignore.
selftest_to_wav() {
  local src="$1" outdir="$2" ext probe noaudio rc=0

  [ -f "$src" ] || die "селф-тест: нет исходного файла $src
  Подготовьте фрагмент (bash scripts/make_sample.sh) или передайте свой:
  bash scripts/lib/common.sh --selftest <медиафайл> [каталог вывода]"
  require_tools
  mkdir -p "$outdir"
  ext="${src##*.}"

  printf '======== 1. медиафайл -> 16 кГц моно WAV ========\n'
  to_wav "$src" "$outdir/selftest_16k_mono.wav"
  probe="$(ffprobe -v error -select_streams a:0 \
    -show_entries stream=codec_name,sample_rate,channels -of csv=p=0 \
    -i "$TO_WAV_PATH" 2>/dev/null || printf '')"
  printf '  аудиопоток:    %s\n' "$probe"
  [ "$probe" = "pcm_s16le,16000,1" ] \
    || die "селф-тест: ожидался поток pcm_s16le,16000,1, получено: $probe"

  printf '\n======== 2. путь с пробелами и кириллицей ========\n'
  cp "$src" "$outdir/копия с пробелами.$ext"
  to_wav "$outdir/копия с пробелами.$ext" "$outdir/копия с пробелами 16k mono.wav"

  printf '\n======== 3. вход уже 16 кГц моно WAV ========\n'
  to_wav "$outdir/selftest_16k_mono.wav" "$outdir/не-должен-появиться.wav"
  [ "$TO_WAV_PATH" = "$outdir/selftest_16k_mono.wav" ] \
    || die "селф-тест: готовый WAV всё-таки перекодировали в $TO_WAV_PATH"
  [ ! -e "$outdir/не-должен-появиться.wav" ] \
    || die "селф-тест: создан лишний файл $outdir/не-должен-появиться.wav"
  printf '  перекодирования не было, лишних файлов нет\n'

  printf '\n======== 4. файл без аудиодорожки ========\n'
  noaudio="$outdir/без-звука.mp4"
  if ! ffmpeg -nostdin -hide_banner -v error -i "$src" -an -c:v copy -t 2 -y "$noaudio" \
       2>/dev/null; then
    printf '  в исходнике нет видео, собрать файл без звука не из чего — случай пропущен\n'
  else
    # Подоболочка: die завершает процесс, и без неё селф-тест оборвался бы здесь.
    ( to_wav "$noaudio" "$outdir/без-звука.wav" ) || rc=$?
    [ "$rc" -ne 0 ] || die "селф-тест: to_wav не заметил отсутствия аудиодорожки"
    printf '  to_wav отказался с кодом %s, как и должен\n' "$rc"
  fi

  printf '\n======== СЕЛФ-ТЕСТ to_wav ПРОЙДЕН ========\n'
}

# Блок срабатывает только при прямом запуске файла: при подключении через source
# BASH_SOURCE[0] (имя этого файла) не совпадает с $0 (именем запущенного
# скрипта), и ни одна команда отсюда не выполняется.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--selftest" ]; then
  selftest_to_wav \
    "${2:-data/samples/Лекция_1_01.09.26_1_t4514_20s.mp4}" \
    "${3:-data/to_wav_check}"
fi
