#!/usr/bin/env bash
#
# make_sample.sh — готовит эталонный фрагмент записи для замеров ASR.
#
# Из длинной записи вырезается один и тот же участок из середины и сохраняется
# в двух видах:
#   * 16 кГц моно WAV (pcm_s16le) — вход для всех ASR-кандидатов. 16 кГц моно —
#     формат, в который Whisper, GigaAM, Vosk и NeMo ресемплируют вход внутри
#     себя, поэтому приведение делается один раз здесь, а не по-разному внутри
#     каждого движка.
#   * короткий видеофрагмент — чтобы глазами сверять расшифровку со слайдами при
#     подготовке эталонного текста.
#
# Фрагмент берётся из середины: края записи — это заставка, настройка техники и
# прощания, они не репрезентативны для лекции.
#
# Скрипт печатает пути, длительности и SHA-256 результатов: протокол замера
# (docs/research/benchmark-plan.md) требует, чтобы все кандидаты получали
# побайтово одинаковый вход, и хеш — способ это подтвердить.
#
# Зависимости: ffmpeg, ffprobe, shasum и стандартные утилиты macOS. Сеть не
# используется. Результаты пишутся только в каталог вывода (по умолчанию
# data/samples/), который не коммитится.
#
# Использование:
#   scripts/make_sample.sh [опции]
#
# Пути с пробелами, скобками и кириллицей поддерживаются — передавайте в кавычках.

set -euo pipefail

INPUT="vidio_example/Лекция 1 01.09.26 (1).webm"
OUTDIR="data/samples"
AUDIO_DUR="300"
VIDEO_DUR="20"
START=""

usage() {
  cat <<'EOF'
Использование: scripts/make_sample.sh [опции]

Опции:
  --input <путь>          исходная запись
                          (по умолчанию "vidio_example/Лекция 1 01.09.26 (1).webm")
  --outdir <каталог>      куда положить результат (по умолчанию data/samples)
  --duration <сек>        длительность аудиофрагмента (по умолчанию 300)
  --video-duration <сек>  длительность видеофрагмента (по умолчанию 20)
  --start <сек>           начало фрагмента; по умолчанию вычисляется так,
                          чтобы фрагмент пришёлся на середину записи
  -h, --help              показать эту справку

Примеры:
  scripts/make_sample.sh
  scripts/make_sample.sh --duration 120 --start 3600
EOF
}

die() {
  printf 'make_sample.sh: %s\n' "$1" >&2
  exit 1
}

need_number() {
  case "$2" in
    ''|*[!0-9.]*) die "$1 ожидает число в секундах, получено: $2" ;;
  esac
  awk -v v="$2" 'BEGIN { exit !(v + 0 > 0) }' || die "$1 должен быть больше нуля"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --input)          [ $# -ge 2 ] || die "--input требует путь";              INPUT="$2";     shift 2 ;;
    --outdir)         [ $# -ge 2 ] || die "--outdir требует каталог";          OUTDIR="$2";    shift 2 ;;
    --duration)       [ $# -ge 2 ] || die "--duration требует значение";       AUDIO_DUR="$2"; shift 2 ;;
    --video-duration) [ $# -ge 2 ] || die "--video-duration требует значение"; VIDEO_DUR="$2"; shift 2 ;;
    --start)          [ $# -ge 2 ] || die "--start требует значение";          START="$2";     shift 2 ;;
    --) shift; [ $# -eq 0 ] || die "после -- аргументы не ожидаются" ;;
    *) die "неизвестная опция: $1 (см. --help)" ;;
  esac
done

[ -f "$INPUT" ] || die "файл не найден: $INPUT"
[ -r "$INPUT" ] || die "файл недоступен для чтения: $INPUT"

need_number "--duration" "$AUDIO_DUR"
need_number "--video-duration" "$VIDEO_DUR"
[ -z "$START" ] || need_number "--start" "$START"

command -v ffprobe >/dev/null 2>&1 || die "не найден ffprobe (установите: brew install ffmpeg)"
command -v ffmpeg  >/dev/null 2>&1 || die "не найден ffmpeg (установите: brew install ffmpeg)"
command -v shasum  >/dev/null 2>&1 || die "не найден shasum"

# ---------------------------------------------------------------- хелперы ---

# Секунды -> чч:мм:сс.ммм
fmt_hms() {
  awk -v s="${1:-}" 'BEGIN {
    if (s == "" || s + 0 < 0) { print "n/a"; exit }
    h = int(s / 3600); m = int((s - h * 3600) / 60); r = s - h * 3600 - m * 60
    printf "%02d:%02d:%06.3f\n", h, m, r
  }'
}

# Байты -> человекочитаемый размер
fmt_bytes() {
  awk -v b="${1:-}" 'BEGIN {
    if (b == "") { print "n/a"; exit }
    n = split("B KiB MiB GiB TiB", u, " "); i = 1
    while (b >= 1024 && i < n) { b /= 1024; i++ }
    f = (i == 1 ? "%d %s\n" : "%.2f %s\n")
    printf f, b, u[i]
  }'
}

# Длительность файла в секундах, пустая строка — если определить не удалось.
# В awk намеренно нет `exit`: он закрыл бы пайп раньше, чем ffprobe допишет
# вывод, и `pipefail` уронил бы скрипт по SIGPIPE.
# Завершающий `|| printf ''` гасит ненулевой код пайплайна: без него `pipefail` в
# связке с `set -e` уронил бы скрипт прямо в подстановке, не дав вызывающему
# коду выдать внятное сообщение об ошибке.
probe_duration() {
  ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 -i "$1" 2>/dev/null \
    | awk 'NR == 1 && $1 != "N/A" { print }' || printf ''
}

# Есть ли в файле поток указанного типа (v или a).
has_stream() {
  local out
  out="$(ffprobe -v error -select_streams "$2:0" -show_entries stream=index \
    -of default=nw=1:nk=1 -i "$1" 2>/dev/null || printf '')"
  [ -n "$out" ]
}

# Описывает готовый файл: размер, фактическая длительность, хеш.
describe() {
  local path="$1" size dur
  [ -s "$path" ] || die "ffmpeg не создал файл или он пуст: $path"
  size="$(stat -f%z "$path" 2>/dev/null || printf '')"
  dur="$(probe_duration "$path")"
  printf '  %s\n' "$path"
  printf '    размер:        %s (%s байт)\n' "$(fmt_bytes "$size")" "${size:-n/a}"
  printf '    длительность:  %s\n' "$(fmt_hms "${dur:-}")"
  printf '    SHA-256:       %s\n' "$(shasum -a 256 "$path" | awk '{ print $1 }')"
}

# ------------------------------------------------------------ подготовка ---

src_duration="$(probe_duration "$INPUT")"
[ -n "$src_duration" ] || die "не удалось определить длительность записи: $INPUT"

has_stream "$INPUT" a \
  || die "в записи нет аудиопотока, фрагмент для ASR собрать не из чего: $INPUT"

has_video=0
if has_stream "$INPUT" v; then
  has_video=1
fi

# Запись короче запрошенного фрагмента — не ошибка: берём её целиком с нуля.
AUDIO_DUR="$(awk -v d="$src_duration" -v a="$AUDIO_DUR" \
  'BEGIN { printf "%.3f", (a + 0 < d + 0 ? a + 0 : d + 0) }')"
VIDEO_DUR="$(awk -v a="$AUDIO_DUR" -v v="$VIDEO_DUR" \
  'BEGIN { printf "%.3f", (v + 0 < a + 0 ? v + 0 : a + 0) }')"

if [ -z "$START" ]; then
  START="$(awk -v d="$src_duration" -v a="$AUDIO_DUR" 'BEGIN {
    s = d / 2 - a / 2
    printf "%.3f", (s > 0 ? s : 0)
  }')"
fi

awk -v s="$START" -v d="$src_duration" 'BEGIN { exit !(s + 0 < d + 0) }' \
  || die "--start ($START с) выходит за пределы записи ($src_duration с)"

AUDIO_END="$(awk -v s="$START" -v a="$AUDIO_DUR" 'BEGIN { printf "%.3f", s + a }')"
VIDEO_END="$(awk -v s="$START" -v v="$VIDEO_DUR" 'BEGIN { printf "%.3f", s + v }')"

mkdir -p "$OUTDIR"

base="$(basename "$INPUT")"
base="${base%.*}"
# Пробелы и скобки из имени исходника уехали бы в имена результатов и мешали бы
# подставлять их в команды ASR-движков, поэтому имя фрагмента собираем сами.
# tr, а не подстановка `${slug//[()]/}`: tr читает вход до конца, так что пайп и
# `pipefail` здесь безопасны, а шаблон со скобками не зависит от версии bash.
slug="$(printf '%s' "$base" | tr ' ' '_' | tr -d '()')"
start_tag="$(awk -v s="$START" 'BEGIN { printf "%d", s }')"

AUDIO_OUT="$OUTDIR/${slug}_t${start_tag}_${AUDIO_DUR%.*}s_16k_mono.wav"
VIDEO_OUT="$OUTDIR/${slug}_t${start_tag}_${VIDEO_DUR%.*}s.mp4"

# -------------------------------------------------------------- нарезка ---

printf '================ ПОДГОТОВКА ЭТАЛОННОГО ФРАГМЕНТА ================\n'
printf 'Исходник:      %s\n' "$INPUT"
printf 'Длительность:  %s (%s с)\n' "$(fmt_hms "$src_duration")" "$src_duration"
printf 'Снято:         %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
printf 'Фрагмент:      %s — %s (аудио, %s с)\n' \
  "$(fmt_hms "$START")" "$(fmt_hms "$AUDIO_END")" "$AUDIO_DUR"
printf 'Видеофрагмент: %s — %s (%s с)\n' \
  "$(fmt_hms "$START")" "$(fmt_hms "$VIDEO_END")" "$VIDEO_DUR"
printf 'Каталог:       %s\n' "$OUTDIR"

# -ss перед -i — быстрый поиск по индексу контейнера, иначе ffmpeg декодировал бы
# три четверти записи впустую. -map 0:a:0 фиксирует первую аудиодорожку: если в
# записи их окажется несколько, выбор не должен зависеть от эвристики ffmpeg.
# -vn не даёт декодировать видео, которого в WAV всё равно не будет.
# На записях Телемоста ffmpeg печатает сюда «File is broken, keyframes not correctly
# marked!» — это свойство исходного WebM с переменной частотой кадров, а не сбой:
# нарезка проходит, файл получается корректным. Сообщение не глушим намеренно, чтобы
# не спрятать заодно настоящие ошибки демуксера.
printf '\n== Аудио (16 кГц, моно, pcm_s16le)\n'
ffmpeg -nostdin -hide_banner -v error -ss "$START" -i "$INPUT" -t "$AUDIO_DUR" \
  -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le -y "$AUDIO_OUT" \
  || die "не удалось вырезать аудиофрагмент"
describe "$AUDIO_OUT"

printf '\n== Видео (для сверки расшифровки со слайдами)\n'
if [ "$has_video" -eq 0 ]; then
  printf '  в записи нет видеопотока — видеофрагмент не создаётся\n'
  VIDEO_OUT=""
else
  # Сборка ffmpeg из brew обычно умеет libx264 и aac, но полагаться на это нельзя,
  # поэтому список кодировщиков читается заранее. Проверка до запуска, а не разбор
  # падения после: иначе в каталоге вывода остался бы обрезанный mp4.
  # Вывод берётся целиком в переменную, без пайпа в grep: `grep -q` закрыл бы пайп
  # на первом совпадении, и `pipefail` превратил бы SIGPIPE в «кодека нет».
  # 2>&1, а не 2>/dev/null, и без -v error: разные сборки ffmpeg печатают список
  # кодировщиков то в stdout, то через лог в stderr, и поднятый порог логов его
  # прячет. Если список всё же не достался, ниже честно сработает ветка без
  # перекодирования — это хуже по точности начала, но не ошибка.
  encoders="$(ffmpeg -nostdin -hide_banner -encoders 2>&1 || printf '')"
  have_x264=0
  case "$encoders" in
    *libx264*) case "$encoders" in *" aac "*) have_x264=1 ;; esac ;;
  esac

  if [ "$have_x264" -eq 1 ]; then
    # Исходник — запись демонстрации экрана с переменной частотой кадров, поэтому
    # на выходе частота принудительно фиксируется (-r 10). Перекодирование здесь
    # не прихоть: копирование без него дало бы фрагмент, начинающийся с ближайшего
    # ключевого кадра, а на статичных слайдах они редкие — то есть не с
    # запрошенной секунды.
    ffmpeg -nostdin -hide_banner -v error -ss "$START" -i "$INPUT" -t "$VIDEO_DUR" \
      -map 0:v:0 -map 0:a:0 -r 10 \
      -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
      -c:a aac -b:a 96k -movflags +faststart -y "$VIDEO_OUT" \
      || die "не удалось вырезать видеофрагмент"
  else
    printf '  libx264/aac в сборке ffmpeg нет, сохраняю фрагмент без перекодирования\n'
    printf '  (фрагмент может начаться с ближайшего ключевого кадра, а не точно с %s)\n' \
      "$(fmt_hms "$START")"
    VIDEO_OUT="$OUTDIR/${slug}_t${start_tag}_${VIDEO_DUR%.*}s.webm"
    ffmpeg -nostdin -hide_banner -v error -ss "$START" -i "$INPUT" -t "$VIDEO_DUR" \
      -map 0:v:0 -map 0:a:0 -c copy -y "$VIDEO_OUT" \
      || die "не удалось вырезать видеофрагмент"
  fi
  describe "$VIDEO_OUT"
fi

printf '\n== Пути результатов\n'
printf '%s\n' "$AUDIO_OUT"
[ -z "$VIDEO_OUT" ] || printf '%s\n' "$VIDEO_OUT"

printf '\n================ ГОТОВО ================\n'
