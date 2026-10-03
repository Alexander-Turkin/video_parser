#!/usr/bin/env bash
#
# inspect_media.sh — воспроизводимый технический разбор медиафайла.
#
# Печатает в stdout отчёт: контейнер и кодеки, точная длительность, параметры
# видео- и аудиопотока, фактические битрейты (по суммарному размеру пакетов),
# громкость (volumedetect + EBU R128) и доля тишины (silencedetect).
#
# Зависимости: ffprobe, ffmpeg и стандартные утилиты macOS (awk, grep, sort,
# stat, date). Сеть не используется, ничего не скачивается, во временные
# каталоги скрипт не пишет — кроме явно запрошенного --frames.
#
# Использование:
#   scripts/inspect_media.sh <путь-к-медиафайлу> [опции]
#
# Пути с пробелами, скобками и кириллицей поддерживаются — передавайте в кавычках.

set -euo pipefail

SILENCE_NOISE="-35dB"
SILENCE_DUR="1.0"
FRAMES_DIR=""
FRAMES_COUNT=5
FILE=""

usage() {
  cat <<'EOF'
Использование: scripts/inspect_media.sh <путь-к-медиафайлу> [опции]

Опции:
  --silence-noise <порог>    порог тишины для silencedetect (по умолчанию -35dB)
  --silence-duration <сек>   минимальная длительность тишины (по умолчанию 1.0)
  --frames <каталог>         извлечь кадры в указанный каталог (по умолчанию нет)
  --frames-count <N>         сколько кадров извлечь (по умолчанию 5)
  -h, --help                 показать эту справку

Пример:
  scripts/inspect_media.sh "vidio_example/Лекция 1 01.09.26 (1).webm"
EOF
}

die() {
  printf 'inspect_media.sh: %s\n' "$1" >&2
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --silence-noise)    [ $# -ge 2 ] || die "--silence-noise требует значение";    SILENCE_NOISE="$2"; shift 2 ;;
    --silence-duration) [ $# -ge 2 ] || die "--silence-duration требует значение"; SILENCE_DUR="$2";   shift 2 ;;
    --frames)           [ $# -ge 2 ] || die "--frames требует каталог";            FRAMES_DIR="$2";    shift 2 ;;
    --frames-count)     [ $# -ge 2 ] || die "--frames-count требует число";        FRAMES_COUNT="$2";  shift 2 ;;
    --) shift; [ $# -ge 1 ] || die "после -- ожидается путь к файлу"; FILE="$1"; shift ;;
    -*) die "неизвестная опция: $1 (см. --help)" ;;
    *)  [ -z "$FILE" ] || die "ожидается ровно один путь к файлу"; FILE="$1"; shift ;;
  esac
done

[ -n "$FILE" ] || { usage >&2; die "не указан путь к медиафайлу"; }
[ -f "$FILE" ] || die "файл не найден: $FILE"
[ -r "$FILE" ] || die "файл недоступен для чтения: $FILE"

case "$FRAMES_COUNT" in
  ''|*[!0-9]*) die "--frames-count ожидает целое число, получено: $FRAMES_COUNT" ;;
esac
[ "$FRAMES_COUNT" -gt 0 ] || die "--frames-count должен быть больше нуля"

command -v ffprobe >/dev/null 2>&1 || die "не найден ffprobe (установите: brew install ffmpeg)"
command -v ffmpeg  >/dev/null 2>&1 || die "не найден ffmpeg (установите: brew install ffmpeg)"

# ---------------------------------------------------------------- хелперы ---

# Берёт значение поля из вывода ffprobe в формате `ключ=значение`.
# Здесь и далее вход подаётся here-string, а не пайпом: awk выходит по первому
# совпадению, и пайп в сочетании с `pipefail` мог бы уронить скрипт по SIGPIPE.
field() {
  awk -F= -v k="$2" '$1 == k { sub(/^[^=]*=/, ""); print; exit }' <<<"$1"
}

# Ищет в логе ffmpeg поле вида `<ключ>: <значение> [<единица>]`.
# Логи фильтров могут приходить как с префиксом `[Parsed_... @ 0x...]`, так и без
# него, поэтому ищем нужный токен по всей строке, а не по номеру поля.
log_field() {
  awk -v k="$2" -v n="${3:-1}" '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == k) {
          out = ""
          for (j = 1; j <= n && i + j <= NF; j++) out = out (j > 1 ? " " : "") $(i + j)
          if (out != "") { print out; exit }
        }
      }
    }' <<<"$1"
}

# Пустое значение и ffprobe-шное N/A приводим к единому «n/a».
or_na() {
  case "${1:-}" in
    ""|"N/A"|"n/a"|"unknown") printf 'n/a' ;;
    *) printf '%s' "$1" ;;
  esac
}

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

# Дробь вида 30000/1001 -> 29.970
fmt_rate() {
  awk -v r="${1:-}" 'BEGIN {
    if (r == "" || r == "N/A" || r == "0/0") { print "n/a"; exit }
    if (split(r, p, "/") < 2 || p[2] + 0 == 0) { print "n/a"; exit }
    printf "%.3f\n", p[1] / p[2]
  }'
}

# Байты полезной нагрузки + секунды -> кбит/с
fmt_kbps() {
  awk -v b="${1:-}" -v d="${2:-}" 'BEGIN {
    if (b == "" || d == "" || b + 0 <= 0 || d + 0 <= 0) { print "n/a"; exit }
    printf "%.0f\n", b * 8 / d / 1000
  }'
}

section() {
  printf '\n== %s\n' "$1"
}

# ------------------------------------------------------------ сбор данных ---

file_size="$(stat -f%z "$FILE" 2>/dev/null || printf '')"

fmt_info="$(ffprobe -v error -show_entries \
  format=format_name,format_long_name,duration,size,bit_rate,nb_streams \
  -of default=nw=1 -i "$FILE")" \
  || die "ffprobe не смог прочитать файл: $FILE"

[ -n "${file_size:-}" ] || file_size="$(field "$fmt_info" size)"

duration="$(field "$fmt_info" duration)"
case "$duration" in ""|"N/A") duration="" ;; esac

v_info="$(ffprobe -v error -select_streams v:0 -show_entries \
  stream=codec_name,codec_long_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,bit_rate,nb_frames \
  -of default=nw=1 -i "$FILE" 2>/dev/null || true)"

a_info="$(ffprobe -v error -select_streams a:0 -show_entries \
  stream=codec_name,codec_long_name,sample_rate,channels,channel_layout,sample_fmt,bit_rate \
  -of default=nw=1 -i "$FILE" 2>/dev/null || true)"

has_video=0
has_audio=0
if [ -n "$v_info" ]; then has_video=1; fi
if [ -n "$a_info" ]; then has_audio=1; fi

# Индексы первых потоков — нужны, чтобы сопоставить пакеты с потоком.
v_index="$(ffprobe -v error -select_streams v:0 -show_entries stream=index \
  -of default=nw=1:nk=1 -i "$FILE" 2>/dev/null | awk 'NR == 1 { print }' || true)"
a_index="$(ffprobe -v error -select_streams a:0 -show_entries stream=index \
  -of default=nw=1:nk=1 -i "$FILE" 2>/dev/null | awk 'NR == 1 { print }' || true)"

# WebM обычно не хранит битрейт в заголовке потока, поэтому считаем его честно —
# по суммарному размеру пакетов. Это один полный проход по файлу без декодирования.
pkt_sums="$(ffprobe -v error -show_entries packet=stream_index,size -of csv=p=0 -i "$FILE" 2>/dev/null \
  | awk -F, -v vi="${v_index:--1}" -v ai="${a_index:--1}" \
      '$1 == vi { v += $2 } $1 == ai { a += $2 } END { printf "%d %d\n", v + 0, a + 0 }' \
  || printf '0 0\n')"
v_bytes="$(awk 'NR == 1 { print $1 }' <<<"$pkt_sums")"
a_bytes="$(awk 'NR == 1 { print $2 }' <<<"$pkt_sums")"

# Один проход декодирования аудио даёт сразу три измерения: громкость по
# volumedetect, громкость по EBU R128 и интервалы тишины. Три отдельных прохода
# по файлу в сотни мегабайт были бы втрое дороже.
# ebur128 сыплет построчной телеметрией (`t: ... M: ... S: ...`) — на двухчасовой
# записи это десятки тысяч строк, поэтому отбрасываем их сразу, не накапливая.
audio_log=""
if [ "$has_audio" -eq 1 ]; then
  audio_log="$(ffmpeg -nostdin -hide_banner -nostats -v info -i "$FILE" -map 0:a:0 \
    -af "volumedetect,silencedetect=noise=${SILENCE_NOISE}:d=${SILENCE_DUR},ebur128=peak=true" \
    -f null - 2>&1 | grep -v ' t: ' || true)"
fi

mean_volume="$(awk -F'mean_volume:' 'NF > 1 { gsub(/^ +| +$/, "", $2); print $2; exit }' <<<"$audio_log")"
max_volume="$(awk  -F'max_volume:'  'NF > 1 { gsub(/^ +| +$/, "", $2); print $2; exit }' <<<"$audio_log")"

# Итоговая сводка ebur128 идёт в конце лога блоком «Summary:». Берём только её,
# чтобы не поймать одноимённые поля из промежуточных строк.
ebur_block="$(awk '/Summary:/ { found = 1 } found { print }' <<<"$audio_log")"
ebu_i="$(log_field    "$ebur_block" "I:"    2)"
ebu_lra="$(log_field  "$ebur_block" "LRA:"  2)"
ebu_peak="$(log_field "$ebur_block" "Peak:" 2)"

silence_stats="$(awk -F'silence_duration:' \
  'NF > 1 { gsub(/^ +| +$/, "", $2); n++; s += $2 }
   END { printf "%d %.3f\n", n + 0, s + 0 }' <<<"$audio_log")"
silence_count="$(awk 'NR == 1 { print $1 }' <<<"$silence_stats")"
silence_total="$(awk 'NR == 1 { print $2 }' <<<"$silence_stats")"
silence_share="$(awk -v s="$silence_total" -v d="${duration:-0}" \
  'BEGIN { if (d + 0 <= 0) print "n/a"; else printf "%.1f", s * 100 / d }')"

# ------------------------------------------------------------------ отчёт ---

printf '================ ИНСПЕКЦИЯ МЕДИАФАЙЛА ================\n'
printf 'Файл:      %s\n' "$FILE"
printf 'Снято:     %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
printf 'ffprobe:   %s\n' "$(ffprobe -version 2>/dev/null | awk 'NR == 1 { print }' || true)"

section "Контейнер"
printf 'Размер:                %s (%s байт)\n' "$(fmt_bytes "$file_size")" "$(or_na "$file_size")"
printf 'Формат:                %s\n' "$(or_na "$(field "$fmt_info" format_name)")"
printf 'Описание формата:      %s\n' "$(or_na "$(field "$fmt_info" format_long_name)")"
printf 'Потоков:               %s\n' "$(or_na "$(field "$fmt_info" nb_streams)")"
printf 'Длительность:          %s (%s с)\n' "$(fmt_hms "$duration")" "$(or_na "$duration")"
printf 'Общий битрейт:         %s кбит/с (по размеру файла)\n' "$(fmt_kbps "${file_size:-0}" "${duration:-0}")"

section "Видеопоток"
if [ "$has_video" -eq 1 ]; then
  printf 'Кодек:                 %s (%s)\n' \
    "$(or_na "$(field "$v_info" codec_name)")" "$(or_na "$(field "$v_info" codec_long_name)")"
  printf 'Разрешение:            %sx%s\n' \
    "$(or_na "$(field "$v_info" width)")" "$(or_na "$(field "$v_info" height)")"
  printf 'Пиксельный формат:     %s\n' "$(or_na "$(field "$v_info" pix_fmt)")"
  printf 'Частота кадров:        %s fps (r_frame_rate %s)\n' \
    "$(fmt_rate "$(field "$v_info" avg_frame_rate)")" "$(or_na "$(field "$v_info" r_frame_rate)")"
  printf 'Кадров в контейнере:   %s\n' "$(or_na "$(field "$v_info" nb_frames)")"
  printf 'Битрейт (заголовок):   %s\n' "$(or_na "$(field "$v_info" bit_rate)")"
  printf 'Битрейт (факт):        %s кбит/с (%s пакетных данных)\n' \
    "$(fmt_kbps "$v_bytes" "${duration:-0}")" "$(fmt_bytes "$v_bytes")"
else
  printf 'видеопоток отсутствует\n'
fi

section "Аудиопоток"
if [ "$has_audio" -eq 0 ]; then
  printf 'аудиопоток отсутствует — измерения громкости и тишины невозможны\n'
else
  printf 'Кодек:                 %s (%s)\n' \
    "$(or_na "$(field "$a_info" codec_name)")" "$(or_na "$(field "$a_info" codec_long_name)")"
  printf 'Частота дискретизации: %s Гц\n' "$(or_na "$(field "$a_info" sample_rate)")"
  printf 'Каналов:               %s (%s)\n' \
    "$(or_na "$(field "$a_info" channels)")" "$(or_na "$(field "$a_info" channel_layout)")"
  printf 'Формат семплов:        %s\n' "$(or_na "$(field "$a_info" sample_fmt)")"
  printf 'Битрейт (заголовок):   %s\n' "$(or_na "$(field "$a_info" bit_rate)")"
  printf 'Битрейт (факт):        %s кбит/с (%s пакетных данных)\n' \
    "$(fmt_kbps "$a_bytes" "${duration:-0}")" "$(fmt_bytes "$a_bytes")"

  section "Громкость"
  printf 'volumedetect mean:     %s\n' "$(or_na "$mean_volume")"
  printf 'volumedetect max:      %s\n' "$(or_na "$max_volume")"
  printf 'EBU R128 I:            %s\n' "$(or_na "$ebu_i")"
  printf 'EBU R128 LRA:          %s\n' "$(or_na "$ebu_lra")"
  printf 'EBU R128 true peak:    %s\n' "$(or_na "$ebu_peak")"

  section "Тишина"
  printf 'Пороги:                noise=%s, минимальная длительность=%s с\n' "$SILENCE_NOISE" "$SILENCE_DUR"
  printf 'Интервалов тишины:     %s\n' "$silence_count"
  printf 'Суммарно тишины:       %s (%s с)\n' "$(fmt_hms "$silence_total")" "$silence_total"
  printf 'Доля тишины:           %s %%\n' "$silence_share"
  printf 'Самые длинные паузы (топ-5, секунд):\n'
  awk -F'silence_duration:' 'NF > 1 { gsub(/^ +| +$/, "", $2); print $2 }' <<<"$audio_log" \
    | sort -rn \
    | awk 'NR <= 5 { printf "  %.3f\n", $1 }
           END { if (NR == 0) print "  (интервалов тишины при заданных порогах не найдено)" }'
fi

if [ -n "$FRAMES_DIR" ] && [ "$has_video" -eq 1 ]; then
  section "Извлечение кадров"
  mkdir -p "$FRAMES_DIR"
  i=1
  while [ "$i" -le "$FRAMES_COUNT" ]; do
    # Края записи пропускаем: там обычно заставка или возня с техникой.
    ts="$(awk -v d="${duration:-0}" -v i="$i" -v n="$FRAMES_COUNT" \
      'BEGIN { printf "%.3f", d * i / (n + 1) }')"
    out="$FRAMES_DIR/frame_$(printf '%02d' "$i")_${ts}s.png"
    if ffmpeg -nostdin -hide_banner -v error -ss "$ts" -i "$FILE" -frames:v 1 -y "$out" 2>/dev/null; then
      printf '  %s  (t=%s, %s с)\n' "$out" "$(fmt_hms "$ts")" "$ts"
    else
      printf '  не удалось извлечь кадр на t=%s с\n' "$ts"
    fi
    i=$((i + 1))
  done
fi

printf '\n================ КОНЕЦ ОТЧЁТА ================\n'
