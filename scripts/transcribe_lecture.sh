#!/usr/bin/env bash
#
# transcribe_lecture.sh — продуктовая расшифровка записи: медиафайл на вход,
# один markdown на выход.
#
# Это вторая точка входа проекта, и она отвечает на другой вопрос, чем
# scripts/transcribe_gigaam.sh. Тот скрипт — бенчмарк: он прогоняет по одному
# плану нарезки несколько чекпойнтов и сравнивает их таблицей. Здесь выбор уже
# сделан (docs/research/benchmark-plan.md, раздел 8: v3_rnnt, паузы не
# выбрасываются), и человеку нужен не замер, а результат — файл, который можно
# открыть и читать: саммари сверху, дальше расшифровка по предложениям с
# таймкодами. Общая часть обоих прогонов — проверки окружения, подготовка WAV,
# блоки «== Окружение» и «== Вход», сводка плана — лежит в scripts/lib/common.sh;
# здесь остаётся только продуктовый конвейер и раскладка его выходов.
#
# Почему порядок шагов именно такой:
#   * вход приводится к 16 кГц моно WAV целиком (to_wav), потому что в этот
#     формат GigaAM всё равно ресемплирует вход внутри себя, а запись
#     обрабатывается полностью — фрагмент из середины готовит другой скрипт,
#     scripts/make_sample.sh;
#   * нарезка всегда идёт в режиме «паузы не выбрасываются»: нарезчику передаётся
#     --drop-pause 86400 (сутки заведомо больше любой паузы, поэтому ни один
#     интервал тишины не уходит в dropped). Флага выбора режима у этой команды
#     нет намеренно — в расшифровке лекции дыра на месте выброшенной паузы
#     означает потерянные слова, а замер по эталону показал, что в этом режиме
#     WER ниже;
#   * пунктуацию ставит claude окнами по --window сегментов, и после каждого
#     окна tools/punctuate.py сверяет последовательность слов с выходом модели:
#     claude расставляет знаки, заглавные и абзацы, но менять слова ему запрещено.
#     Окно, не прошедшее сверку, откатывается к дословному тексту и помечается
#     в документе — лучше видимая пометка, чем незаметно переписанная лекция;
#   * саммари делается одним отдельным вызовом по всему тексту: окна для него не
#     годятся, изложение и темы нужны по записи целиком.
#
# Веса не скачиваются: чекпойнт должен уже лежать в --download-root. Отсутствие
# claude прогон не останавливает — расшифровка в этом случае всё равно будет
# собрана, но с пометками «не получено» и ненулевым кодом возврата.
#
# Зависимости: ffmpeg, ffprobe, shasum, awk, python с gigaam (берётся из
# ./.venv/bin/python, иначе python3) и claude в PATH. Сеть нужна только claude.
# Всё, что скрипт пишет, лежит под --outdir внутри data/, закрытого корневым
# .gitignore: текст лекции в репозиторий не попадает.
#
# Использование:
#   bash scripts/transcribe_lecture.sh --input <медиафайл> [опции]
#
# Запускать из корня репозитория. Бит исполняемости не нужен. Пути с пробелами и
# кириллицей поддерживаются — передавайте в кавычках.

set -euo pipefail

# Библиотека подключается по пути от корня репозитория — там же, где лежат
# tools/*.py, которые скрипт вызывает ниже. Проверка до source своя: die ещё не
# определён, а сообщение bash «No such file or directory» не подсказывает, что
# дело в каталоге запуска.
if [ ! -f scripts/lib/common.sh ]; then
  printf 'transcribe_lecture.sh: не найден scripts/lib/common.sh — запускайте скрипт из корня репозитория\n' >&2
  exit 1
fi
SCRIPT_NAME="transcribe_lecture.sh"
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh

INPUT=""
OUTDIR="data/output"
MODEL="v3_rnnt"
DEVICE="cpu"
DOWNLOAD_ROOT="$HOME/.cache/gigaam"
FROM_TRANSCRIPT=""
WINDOW="100"
KEEP_CHUNKS=1

# Режим нарезки у этой команды один и не настраивается (см. шапку). Значение
# передаётся нарезчику как --drop-pause и печатается в сводке плана.
KEEP_PAUSES_DROP="86400"
CHUNK_MODE="паузы не выбрасываются (нарезчику передан --drop-pause $KEEP_PAUSES_DROP с)"

usage() {
  cat <<'EOF'
Использование: bash scripts/transcribe_lecture.sh --input <медиафайл> [опции]

Расшифровка записи целиком одной командой: любой читаемый ffmpeg медиафайл
(webm, mp4, mp3, m4a, wav) приводится к 16 кГц моно WAV, режется по тишине без
выбрасывания пауз, распознаётся GigaAM, пунктуацию расставляет claude без
изменения слов, после чего собирается один markdown: саммари и расшифровка по
предложениям с таймкодами.

Опции:
  --input <путь>          входной медиафайл (обязателен; при --from-transcript
                          нужен только для имени итогового файла)
  --outdir <каталог>      куда писать итог и рабочий подкаталог
                          (по умолчанию data/output)
  --model <имя>           чекпойнт GigaAM (по умолчанию v3_rnnt)
  --device <устройство>   устройство для torch (по умолчанию cpu)
  --download-root <кат.>  каталог с весами GigaAM (по умолчанию ~/.cache/gigaam)
  --from-transcript <п.>  взять готовый transcript.json и переделать только
                          пунктуацию, саммари и итоговый документ: конвертация,
                          нарезка, распознавание, .txt/.srt и проверки
                          таймкодов пропускаются
  --window <число>        сколько сегментов отправлять claude в одном окне
                          пунктуации (по умолчанию 100)
  --keep-chunks           оставить нарезанные чанки (по умолчанию)
  --no-chunks             удалить чанки после распознавания, план оставить
  -h, --help              показать эту справку

Режим нарезки не настраивается: паузы не выбрасываются никогда (нарезчику
передаётся --drop-pause 86400), иначе в расшифровке лекции появились бы дыры на
месте выброшенной тишины. Выбранный режим печатается в сводке плана.

Что появится на диске (<имя> — имя входного файла без расширения,
<дата> — ГГГГ-ММ-ДД; если итоговый файл с таким именем уже есть, к имени
добавляется -ЧЧММСС, существующий файл не перетирается):

  <outdir>/<имя>-<дата>.md          итоговый документ: саммари и расшифровка
  <outdir>/<имя>-<дата>/            рабочий подкаталог рядом с итогом:
      audio.wav                     подготовленный 16 кГц моно WAV (ссылка на
                                    вход, если он уже в этом формате)
      plan.json                     план нарезки
      chunks/chunk_000.wav …        нарезанные чанки (нет при --no-chunks)
      transcript.json               дословный выход модели с таймкодами
      transcript.txt                читаемый текст
      transcript.srt                субтитры
      checks.json, checks.txt       проверки таймкодов
      punctuated.json               предложения с пунктуацией и таймкодами
      summary.md                    фрагмент саммари
      claude/                       промпты и ответы claude по вызовам
      run.log                       весь вывод этого прогона

Порядок шагов: to_wav -> tools/chunker.py -> tools/transcribe.py ->
tools/outputs.py -> tools/check_timecodes.py -> tools/punctuate.py ->
tools/summarize.py -> tools/render_doc.py. Каждая команда печатается перед
запуском. Последняя строка вывода — полный путь к итоговому .md.

Коды возврата:
  0  итог собран, пунктуация восстановлена во всех окнах, саммари получено,
     проверки таймкодов пройдены
  1  итог собран, но есть пометки: нет саммари, есть окна с дословным текстом
     или провалены проверки таймкодов
  2  расшифровку получить не удалось, итогового .md нет

Примеры:
  bash scripts/transcribe_lecture.sh --input "vidio_example/Лекция 1.webm"
  bash scripts/transcribe_lecture.sh --input запись.mp4 --window 60 --no-chunks
  bash scripts/transcribe_lecture.sh --from-transcript \
      "data/output/Лекция 1-2026-10-03/transcript.json" --input "Лекция 1.webm"
EOF
}

# Окно пунктуации проверяется здесь, при разборе аргументов, а не внутри
# tools/punctuate.py: иначе опечатка в числе вылезла бы уже после подготовки WAV
# и распознавания, то есть через десятки минут прогона.
require_positive_int() {
  case "$2" in
    *[!0-9]*|'') die "$1 ожидает целое число больше нуля, получено: $2" ;;
  esac
  [ "$2" -gt 0 ] || die "$1 ожидает целое число больше нуля, получено: $2"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --input)           [ $# -ge 2 ] || die "--input требует путь";            INPUT="$2";           shift 2 ;;
    --outdir)          [ $# -ge 2 ] || die "--outdir требует каталог";        OUTDIR="$2";          shift 2 ;;
    --model)           [ $# -ge 2 ] || die "--model требует имя чекпойнта";   MODEL="$2";           shift 2 ;;
    --device)          [ $# -ge 2 ] || die "--device требует значение";       DEVICE="$2";          shift 2 ;;
    --download-root)   [ $# -ge 2 ] || die "--download-root требует каталог"; DOWNLOAD_ROOT="$2";   shift 2 ;;
    --from-transcript) [ $# -ge 2 ] || die "--from-transcript требует путь к transcript.json"
                       FROM_TRANSCRIPT="$2"; shift 2 ;;
    --window)
      [ $# -ge 2 ] || die "--window требует число сегментов"
      require_positive_int "--window" "$2"
      WINDOW="$2"
      shift 2 ;;
    --keep-chunks)     KEEP_CHUNKS=1; shift ;;
    --no-chunks)       KEEP_CHUNKS=0; shift ;;
    --) shift; [ $# -eq 0 ] || die "после -- аргументы не ожидаются" ;;
    *) die "неизвестная опция: $1 (см. --help)" ;;
  esac
done

[ -n "$OUTDIR" ] || die "--outdir пуст: некуда писать результат"
[ -n "$MODEL" ]  || die "--model пуст: не задан чекпойнт"

if [ -n "$FROM_TRANSCRIPT" ]; then
  [ -f "$FROM_TRANSCRIPT" ] || die "не найден файл транскрипта: $FROM_TRANSCRIPT"
  [ -r "$FROM_TRANSCRIPT" ] || die "файл транскрипта недоступен для чтения: $FROM_TRANSCRIPT"
else
  [ -n "$INPUT" ] || die "не задан --input: нужен медиафайл (см. --help)"
fi

# Существование входного файла проверяется только там, где его собираются
# читать. При --from-transcript из --input берётся одно имя, и требовать, чтобы
# сама запись всё ещё лежала на диске, незачем: пунктуацию переделывают по
# готовому транскрипту, нередко уже без исходника.
if [ -n "$INPUT" ] && [ -z "$FROM_TRANSCRIPT" ]; then
  [ -f "$INPUT" ] || die "файл не найден: $INPUT"
  [ -r "$INPUT" ] || die "файл недоступен для чтения: $INPUT"
fi

# --------------------------------------------------- проверка окружения ---

for tool in chunker transcribe outputs check_timecodes punctuate summarize render_doc; do
  [ -f "tools/$tool.py" ] \
    || die "не найден tools/$tool.py — запускайте скрипт из корня репозитория"
done

PY="$(resolve_python)"
require_tools

# Пакет gigaam и веса нужны только распознаванию. При --from-transcript его не
# будет, и требовать чекпойнт было бы странно: человек переделывает пунктуацию и
# саммари по уже готовому транскрипту, возможно на другой машине.
if [ -n "$FROM_TRANSCRIPT" ]; then
  WEIGHTS_NOTE="проверка пропущена: --from-transcript, распознавания не будет"
else
  require_python_gigaam "$PY"
  require_weights "$MODEL" "$DOWNLOAD_ROOT"
  WEIGHTS_NOTE="$DOWNLOAD_ROOT/$MODEL.ckpt"
fi

# Отсутствие claude — не ошибка прогона: расшифровка нужна человеку и без
# пунктуации. Предупреждение печатается до начала работы, чтобы не ждать
# полчаса распознавания ради пометок «не получено».
CLAUDE_NOTE="есть"
if ! "$PY" tools/claude_cli.py --probe >/dev/null 2>&1; then
  warn "claude в PATH не найден: расшифровка будет собрана, но без пунктуации и саммари, код возврата 1"
  CLAUDE_NOTE="НЕТ — пунктуации и саммари не будет"
fi

# ------------------------------------------------------- имена и каталоги ---

# Абсолютный путь без realpath и python: нужен и для ссылки audio.wav (она
# лежит в другом каталоге, чем вход), и для последней строки вывода.
abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

# Имя итогового файла — это имя записи, а не имя шага конвейера. При --input оно
# берётся из входного файла; при одном --from-transcript его приходится
# восстанавливать из самого транскрипта: поле source там указывает на audio.wav
# рабочего каталога, поэтому вторым заходом берётся имя этого каталога, из
# которого снимается дата прошлого прогона.
base_from_transcript() {
  "$PY" - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    data = {}

name = Path(str(data.get("source") or "").strip()).stem if data.get("source") else ""
if name in ("", "audio"):
    name = path.parent.name if path.stem == "transcript" else path.stem
    # "Лекция 1-2026-10-03-204500" -> "Лекция 1": иначе к имени приклеилась бы
    # вторая дата, и по файлу стало бы непонятно, какой это прогон.
    name = re.sub(r"-\d{4}-\d{2}-\d{2}(-\d{6})?$", "", name)
print(name)
PY
}

if [ -n "$INPUT" ]; then
  BASE="${INPUT##*/}"
  BASE="${BASE%.*}"
else
  BASE="$(base_from_transcript "$FROM_TRANSCRIPT")"
fi
[ -n "$BASE" ] || die "не удалось вывести имя итогового файла — передайте --input"

mkdir -p "$OUTDIR" || die "не удалось создать каталог вывода: $OUTDIR"

# Существующий итог не перетирается никогда: расшифровка — результат получаса
# работы и ручной вычитки, и второй прогон того же дня не должен её потерять.
# Рабочий подкаталог называется так же, как итоговый файл, чтобы по имени было
# видно, к какому документу относятся промежуточные файлы.
STEM="$BASE-$(date '+%Y-%m-%d')"
if [ -e "$OUTDIR/$STEM.md" ] || [ -e "$OUTDIR/$STEM" ]; then
  STEM="$STEM-$(date '+%H%M%S')"
  if [ -e "$OUTDIR/$STEM.md" ]; then
    die "итоговый файл уже существует: $OUTDIR/$STEM.md
  Это второй прогон в ту же секунду. Подождите секунду и повторите."
  fi
fi

DOC="$OUTDIR/$STEM.md"
WORKDIR="$OUTDIR/$STEM"
mkdir -p "$WORKDIR" || die "не удалось создать рабочий подкаталог: $WORKDIR"

WAV="$WORKDIR/audio.wav"
PLAN="$WORKDIR/plan.json"
TRANSCRIPT="$WORKDIR/transcript.json"
TXT="$WORKDIR/transcript.txt"
SRT="$WORKDIR/transcript.srt"
CHECKS_JSON="$WORKDIR/checks.json"
CHECKS_TXT="$WORKDIR/checks.txt"
PUNCTUATED="$WORKDIR/punctuated.json"
SUMMARY="$WORKDIR/summary.md"
CLAUDE_LOGS="$WORKDIR/claude"
LOG="$WORKDIR/run.log"

# Весь дальнейший вывод идёт и на экран, и в run.log: человек смотрит прогон
# глазами, а разбирать потом приходится по файлу рядом с итогом. Поэтому tee, а
# не предложение в справке «запустите с | tee»: лог обязан лежать в рабочем
# подкаталоге сам, без усилий со стороны запускающего.
LOG_NOTE="$LOG"
if : >"$LOG" 2>/dev/null; then
  exec > >(tee -a "$LOG") 2>&1
else
  warn "не удалось открыть $LOG — прогон пойдёт без файла лога"
  LOG_NOTE="НЕТ — файл не открылся"
fi

# ------------------------------------------------------------ учёт шагов ---

# Статусы собираются в массивы, а не печатаются по ходу: итог нужен цельным в
# конце прогона, одним блоком, по которому видно, что именно не прошло.
# FLAWED — итог будет, но с пометками (код 1); NO_DOC — итога не будет (код 2).
STEP_NAME=()
STEP_STATE=()
FLAWED=0
NO_DOC=0

step() {
  STEP_NAME+=("$1")
  STEP_STATE+=("$2")
}

# --------------------------------------------------------------- шапка ---

printf '================ РАСШИФРОВКА ЗАПИСИ ================\n'
printf 'Снято:         %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
printf 'Итог:          %s\n' "$DOC"
printf 'Рабочий кат.:  %s\n' "$WORKDIR"

print_env_block "$DEVICE"
printf '  claude:         %s\n' "$CLAUDE_NOTE"
printf '  веса:           %s\n' "$WEIGHTS_NOTE"

if [ -n "$FROM_TRANSCRIPT" ]; then
  printf '\n== Готовый транскрипт (--from-transcript)\n'
  printf '  файл:          %s\n' "$FROM_TRANSCRIPT"
  printf '  шаги подготовки WAV, нарезки и распознавания пропускаются\n'
  if [ -n "$INPUT" ]; then
    printf '  --input используется только для имени итогового файла: %s\n' "$INPUT"
  fi
else
  print_input_block "$INPUT"
fi

# =============================== 1. WAV ===============================

if [ -n "$FROM_TRANSCRIPT" ]; then
  step "подготовка WAV" "пропущено — --from-transcript"
  step "нарезка"        "пропущено — --from-transcript"
  step "распознавание"  "пропущено — --from-transcript"
  step ".txt и .srt"    "пропущено — --from-transcript"
  step "проверки таймкодов" "пропущено — --from-transcript"

  # Транскрипт переносится в рабочий подкаталог, а не читается с места: итог и
  # все его промежуточные файлы должны лежать вместе, иначе через месяц по
  # документу не восстановить, из какого транскрипта он собран.
  printf '\n================ ГОТОВЫЙ ТРАНСКРИПТ ================\n'
  if [ "$(abs_path "$FROM_TRANSCRIPT")" = "$(abs_path "$TRANSCRIPT")" ]; then
    printf '  транскрипт уже лежит в рабочем подкаталоге: %s\n' "$TRANSCRIPT"
  else
    printf '$ cp %s %s\n' "$FROM_TRANSCRIPT" "$TRANSCRIPT"
    cp "$FROM_TRANSCRIPT" "$TRANSCRIPT" \
      || die "не удалось скопировать транскрипт в рабочий подкаталог: $TRANSCRIPT"
  fi
else
  printf '\n================ ПОДГОТОВКА WAV ================\n'
  printf '$ to_wav %s %s\n' "$INPUT" "$WAV"
  to_wav "$INPUT" "$WAV"

  # to_wav не перекодирует вход, который уже 16 кГц моно WAV, и возвращает в
  # TO_WAV_PATH сам вход. Копировать гигабайты сам в себя ради имени файла
  # незачем, поэтому в рабочем подкаталоге появляется ссылка: audio.wav на месте,
  # дисковое место не удвоено. Если ссылку создать не удалось (чужая файловая
  # система), работаем прямо по входу и говорим об этом.
  if [ "$TO_WAV_PATH" != "$WAV" ]; then
    if ln -s "$(abs_path "$TO_WAV_PATH")" "$WAV" 2>/dev/null; then
      printf '  audio.wav в рабочем подкаталоге — ссылка на вход\n'
    else
      warn "не удалось создать ссылку $WAV — дальше работаю прямо по входу"
      WAV="$TO_WAV_PATH"
    fi
  fi
  [ -s "$WAV" ] || die "подготовленного WAV нет или он пуст: $WAV"
  step "подготовка WAV" "ок — $WAV"

  # ============================ 2. НАРЕЗКА ============================

  printf '\n================ ПЛАН НАРЕЗКИ ================\n'
  printf '$ %s tools/chunker.py --input %s --outdir %s --drop-pause %s\n' \
    "$PY" "$WAV" "$WORKDIR" "$KEEP_PAUSES_DROP"
  "$PY" tools/chunker.py --input "$WAV" --outdir "$WORKDIR" \
    --drop-pause "$KEEP_PAUSES_DROP" || die "нарезка не удалась, распознавание не начинаем"
  [ -f "$PLAN" ] || die "chunker.py отработал, но плана нет: $PLAN"

  printf '\n== Сводка плана\n'
  plan_summary "$PLAN" "$CHUNK_MODE" || die "не удалось прочитать план: $PLAN"
  step "нарезка" "ок — $CHUNK_MODE"

  # ========================= 3. РАСПОЗНАВАНИЕ =========================

  printf '\n================ РАСПОЗНАВАНИЕ (%s) ================\n' "$MODEL"
  printf '$ %s tools/transcribe.py --plan %s --model %s --out %s --device %s --download-root %s\n' \
    "$PY" "$PLAN" "$MODEL" "$TRANSCRIPT" "$DEVICE" "$DOWNLOAD_ROOT"
  rc=0
  "$PY" tools/transcribe.py --plan "$PLAN" --model "$MODEL" --out "$TRANSCRIPT" \
    --device "$DEVICE" --download-root "$DOWNLOAD_ROOT" || rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$TRANSCRIPT" ]; then
    step "распознавание" "ПРОВАЛ — transcribe.py вернул код $rc, транскрипта нет"
    NO_DOC=1
  else
    step "распознавание" "ок — $TRANSCRIPT"
  fi

  # ========================== 4. .txt и .srt ==========================

  if [ "$NO_DOC" -eq 0 ]; then
    printf '\n================ ТЕКСТ И СУБТИТРЫ ================\n'
    printf '$ %s tools/outputs.py --transcript %s --txt %s --srt %s\n' \
      "$PY" "$TRANSCRIPT" "$TXT" "$SRT"
    rc=0
    "$PY" tools/outputs.py --transcript "$TRANSCRIPT" --txt "$TXT" --srt "$SRT" || rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$TXT" ] || [ ! -s "$SRT" ]; then
      # Итоговый документ собирается не из .txt и .srt, а из punctuated.json,
      # поэтому прогон продолжается: это пометка, а не остановка.
      step ".txt и .srt" "ПРОВАЛ — outputs.py вернул код $rc"
      FLAWED=1
    else
      step ".txt и .srt" "ок — $TXT, $SRT"
    fi

    # ===================== 5. ПРОВЕРКИ ТАЙМКОДОВ =====================

    printf '\n================ ПРОВЕРКИ ТАЙМКОДОВ ================\n'
    printf '$ %s tools/check_timecodes.py --transcript %s --srt %s --json %s\n' \
      "$PY" "$TRANSCRIPT" "$SRT" "$CHECKS_JSON"
    rc=0
    # Вывод проверок сохраняется и печатается: он часть отчёта о прогоне.
    # Пайп в tee здесь не годится — потерялся бы код возврата проверок.
    "$PY" tools/check_timecodes.py --transcript "$TRANSCRIPT" --srt "$SRT" \
      --json "$CHECKS_JSON" >"$CHECKS_TXT" 2>&1 || rc=$?
    cat "$CHECKS_TXT"
    if [ "$rc" -ne 0 ]; then
      step "проверки таймкодов" "ПРОВАЛ — код $rc, см. $CHECKS_TXT"
      FLAWED=1
    else
      step "проверки таймкодов" "ок — см. $CHECKS_TXT"
    fi
  fi

  # ====================== чанки после распознавания ======================

  if [ "$KEEP_CHUNKS" -eq 0 ]; then
    printf '\n== Удаляю чанки (--no-chunks), план остаётся\n'
    # Удаляются файлы по маске нарезчика, а не по списку из плана: маска узкая,
    # ничего, кроме chunk_*.wav, скрипт не трогает.
    removed=0
    for chunk_file in "$WORKDIR"/chunks/chunk_*.wav; do
      [ -f "$chunk_file" ] || continue
      rm -f -- "$chunk_file"
      removed=$((removed + 1))
    done
    printf '  удалено файлов: %d\n' "$removed"
    if [ -d "$WORKDIR/chunks" ] && ! rmdir "$WORKDIR/chunks" 2>/dev/null; then
      printf '  каталог %s/chunks оставлен: в нём есть ещё что-то, кроме чанков\n' "$WORKDIR"
    fi
  fi
fi

# ============================ 6. ПУНКТУАЦИЯ ============================

if [ "$NO_DOC" -eq 0 ]; then
  printf '\n================ ПУНКТУАЦИЯ ================\n'
  printf '$ %s tools/punctuate.py --transcript %s --out %s --window %s --log-dir %s\n' \
    "$PY" "$TRANSCRIPT" "$PUNCTUATED" "$WINDOW" "$CLAUDE_LOGS"
  rc=0
  "$PY" tools/punctuate.py --transcript "$TRANSCRIPT" --out "$PUNCTUATED" \
    --window "$WINDOW" --log-dir "$CLAUDE_LOGS" || rc=$?
  if [ "$rc" -ge 2 ] || [ ! -s "$PUNCTUATED" ]; then
    step "пунктуация" "ПРОВАЛ — punctuate.py вернул код $rc, предложений нет"
    NO_DOC=1
  elif [ "$rc" -eq 1 ]; then
    # Код 1 у punctuate.py — это окна, откатившиеся к дословному тексту: claude
    # не ответил или изменил слова. Документ будет, но с пометками.
    step "пунктуация" "С ПОМЕТКАМИ — есть окна с дословным текстом (код $rc)"
    FLAWED=1
  else
    step "пунктуация" "ок — все окна прошли сверку слов"
  fi
else
  step "пунктуация" "пропущено — нет расшифровки"
fi

# ============================== 7. САММАРИ ==============================

SUMMARY_ARG=""
if [ "$NO_DOC" -eq 0 ]; then
  printf '\n================ САММАРИ ================\n'
  printf '$ %s tools/summarize.py --punctuated %s --out %s --log-dir %s\n' \
    "$PY" "$PUNCTUATED" "$SUMMARY" "$CLAUDE_LOGS"
  rc=0
  "$PY" tools/summarize.py --punctuated "$PUNCTUATED" --out "$SUMMARY" \
    --log-dir "$CLAUDE_LOGS" || rc=$?
  if [ "$rc" -ge 2 ] || [ ! -s "$SUMMARY" ]; then
    # Файла нет вовсе — в документ пойдёт пометка «не получено» от render_doc,
    # поэтому --summary ему не передаём.
    step "саммари" "ПРОВАЛ — summarize.py вернул код $rc, файла нет"
    FLAWED=1
  else
    SUMMARY_ARG="$SUMMARY"
    if [ "$rc" -eq 1 ]; then
      step "саммари" "НЕ ПОЛУЧЕНО — claude не ответил, в документе пометка (код $rc)"
      FLAWED=1
    else
      step "саммари" "ок — $SUMMARY"
    fi
  fi
else
  step "саммари" "пропущено — нет пунктуации"
fi

# ========================= 8. ИТОГОВЫЙ ДОКУМЕНТ =========================

if [ "$NO_DOC" -eq 0 ]; then
  printf '\n================ ИТОГОВЫЙ ДОКУМЕНТ ================\n'
  if [ -n "$SUMMARY_ARG" ]; then
    printf '$ %s tools/render_doc.py --punctuated %s --summary %s --out %s --title %s\n' \
      "$PY" "$PUNCTUATED" "$SUMMARY_ARG" "$DOC" "$BASE"
    rc=0
    "$PY" tools/render_doc.py --punctuated "$PUNCTUATED" --summary "$SUMMARY_ARG" \
      --out "$DOC" --title "$BASE" || rc=$?
  else
    printf '$ %s tools/render_doc.py --punctuated %s --out %s --title %s\n' \
      "$PY" "$PUNCTUATED" "$DOC" "$BASE"
    rc=0
    "$PY" tools/render_doc.py --punctuated "$PUNCTUATED" \
      --out "$DOC" --title "$BASE" || rc=$?
  fi
  if [ "$rc" -ge 2 ] || [ ! -s "$DOC" ]; then
    step "итоговый документ" "ПРОВАЛ — render_doc.py вернул код $rc, файла нет"
    NO_DOC=1
  elif [ "$rc" -eq 1 ]; then
    step "итоговый документ" "С ПОМЕТКАМИ — собран, но внутри есть пометки (код $rc)"
    FLAWED=1
  else
    step "итоговый документ" "ок — $DOC"
  fi
else
  step "итоговый документ" "НЕ СОБРАН — пригодной расшифровки получить не удалось"
fi

# ================================ ИТОГ ================================

printf '\n================ СТАТУС ПО ШАГАМ ================\n'
# Шаги печатаются номером и двоеточием, а не выровненной колонкой: printf
# считает ширину в байтах, а не в символах, поэтому русские названия шагов
# разной длины в байтах разъехались бы по строкам. Номер заодно показывает
# порядок конвейера — по нему видно, на каком шаге прогон споткнулся.
index=0
while [ "$index" -lt "${#STEP_NAME[@]}" ]; do
  printf '  %d. %s: %s\n' "$((index + 1))" "${STEP_NAME[$index]}" "${STEP_STATE[$index]}"
  index=$((index + 1))
done

printf '\n== Файлы прогона\n'
for name in audio.wav plan.json transcript.json transcript.txt transcript.srt \
            checks.json checks.txt punctuated.json summary.md; do
  if [ -e "$WORKDIR/$name" ]; then
    printf '  %s/%s\n' "$WORKDIR" "$name"
  fi
done
# Через `[ -d … ] && printf` эти две строки писать нельзя: при отсутствии
# каталога весь список вернул бы ненулевой код, и set -e оборвал бы прогон перед
# самым итогом — ровно там, где человеку нужен путь к документу.
if [ -d "$WORKDIR/chunks" ]; then
  printf '  %s/chunks/\n' "$WORKDIR"
fi
if [ -d "$CLAUDE_LOGS" ]; then
  printf '  %s/\n' "$CLAUDE_LOGS"
fi
printf '  %s\n' "$LOG_NOTE"

if [ "$NO_DOC" -ne 0 ]; then
  printf '\n================ ИТОГА НЕТ ================\n'
  printf 'Расшифровку получить не удалось, итогового .md нет. Промежуточные файлы\n'
  printf 'остались в %s — разбирайтесь по ним и по статусу шагов выше.\n' "$WORKDIR"
  exit 2
fi

if [ "$FLAWED" -ne 0 ]; then
  printf '\n================ ГОТОВО, НО С ПОМЕТКАМИ ================\n'
  printf 'Итог собран, но часть шагов не прошла — см. статус выше и пометки в документе.\n'
else
  printf '\n================ ГОТОВО ================\n'
fi

# Последняя строка вывода — полный путь к итоговому документу: по ней прогон
# можно подхватить скриптом или просто открыть файл, не вычитывая лог.
abs_path "$DOC"

if [ "$FLAWED" -ne 0 ]; then
  exit 1
fi
