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
#   * диаризация (кто и когда говорил) идёт после удачного распознавания, а не
#     до него: на CPU pyannote считает примерно в реальном времени, и тратить
#     эти десятки минут на прогон, у которого транскрипта так и не вышло, было
#     бы бессмысленно. Веса pyannote не скачиваются — нужен заранее
#     подготовленный локальный кэш, см. tools/diarize.py;
#   * пунктуацию ставит claude окнами по --window сегментов, и после каждого
#     окна tools/punctuate.py сверяет последовательность слов с выходом модели:
#     claude расставляет знаки, заглавные и абзацы, но менять слова ему запрещено.
#     Окно, не прошедшее сверку, откатывается к дословному тексту и помечается
#     в документе — лучше видимая пометка, чем незаметно переписанная лекция;
#   * фразы с говорящими и роли считаются по готовой пунктуации: границы фраз
#     берут абзацы из punctuated.json, а роли («Преподаватель», «Студент N»)
#     определяет claude по тексту уже разбитой на фразы расшифровки;
#   * саммари делается одним отдельным вызовом по всему тексту: окна для него не
#     годятся, изложение и темы нужны по записи целиком, зато ярлыки говорящих к
#     этому моменту уже есть — и в саммари видно, кто что сказал.
#
# Веса не скачиваются: чекпойнт должен уже лежать в --download-root. Отсутствие
# claude прогон не останавливает — расшифровка в этом случае всё равно будет
# собрана, но с пометками «не получено» и ненулевым кодом возврата. Так же
# устроены и шаги говорящих: не получилась диаризация, не удалась привязка фраз
# или не вышли роли — документ собирается без ярлыков, в нём встаёт видимая
# пометка с причиной, код возврата 1. Совсем отказаться от этих шагов можно
# флагом --no-diarize, а подложить готовый файл — флагом --diarization.
#
# Убирать за собой скрипт умеет флагом --clean, и удаляет при этом не сам: он
# зовёт tools/cleanup_runs.py --run <рабочий подкаталог>. Что считать лишним,
# решает тот инструмент, здесь только условие — чистка идёт строго после
# безупречного прогона (код 0). Прогон с пометками и прогон без итога оставляют
# промежуточные файлы нетронутыми: по ним потом и разбираются.
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
CLEAN=0
DIARIZE=1
DIARIZATION_IN=""

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
изменения слов, pyannote определяет, кто и когда говорил, после чего собирается
один markdown: саммари и расшифровка фразами с ярлыком говорящего.

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
                          таймкодов пропускаются. Шаги говорящих без
                          --diarization пропускаются тоже (звука под рукой уже
                          нет), и это не пометка — код возврата не меняется
  --window <число>        сколько сегментов отправлять claude в одном окне
                          пунктуации (по умолчанию 100)
  --no-diarize            не определять говорящих: шаги диаризации, фраз и
                          ролей пропускаются целиком, расшифровка идёт
                          предложениями без ярлыков. Это не пометка: безупречный
                          прогон с этим флагом по-прежнему даёт код 0
  --diarization <путь>    взять готовый diarization.json схемы diarization/1
                          вместо прогона pyannote (файл копируется в рабочий
                          подкаталог); фразы и роли считаются как обычно.
                          С --no-diarize несовместим
  --keep-chunks           оставить нарезанные чанки (по умолчанию)
  --no-chunks             удалить чанки после распознавания, план оставить
  --clean                 после сборки документа вычистить рабочий подкаталог
                          вызовом tools/cleanup_runs.py --run <подкаталог>:
                          удаляются audio.wav, plan.json, chunks/, .txt и .srt,
                          проверки таймкодов, claude/ и run.log, а результаты
                          шагов остаются. Что именно считать лишним, решает тот
                          инструмент — список файлов-хранителей живёт там, см.
                          python tools/cleanup_runs.py --help. Чистка идёт
                          строго при коде возврата 0: при 1 (есть пометки) и при
                          2 (итога нет) не удаляется ничего, а причина пропуска
                          печатается
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
      diarization.json              кто и когда говорил (нет при --no-diarize)
      punctuated.json               предложения с пунктуацией и таймкодами
      speakers.json                 фразы с говорящими
      roles.json                    роли говорящих: «Преподаватель», «Студент N»
      summary.md                    фрагмент саммари
      claude/                       промпты и ответы claude по вызовам
      run.log                       весь вывод этого прогона

При --clean и безупречном прогоне от этого подкаталога остаются только
результаты шагов, всё промежуточное удаляется. Удаляется в том числе run.log:
на экран прогон допишется до конца, но лога на диске потом не будет — искать
его после --clean негде.

Порядок шагов: to_wav -> tools/chunker.py -> tools/transcribe.py ->
tools/outputs.py -> tools/check_timecodes.py -> tools/diarize.py ->
tools/punctuate.py -> tools/speakers.py -> tools/roles.py ->
tools/summarize.py -> tools/render_doc.py, при --clean в конце ещё
tools/cleanup_runs.py. Диаризация стоит после распознавания: на CPU pyannote
считает примерно в реальном времени, и на прогон без транскрипта эти десятки
минут тратить незачем. Каждая команда печатается перед запуском. Последняя
строка вывода — полный путь к итоговому .md.

Коды возврата:
  0  итог собран, пунктуация восстановлена во всех окнах, саммари получено,
     проверки таймкодов пройдены, ярлыки говорящих проставлены (или шаги
     говорящих выключены флагом --no-diarize — это тоже код 0)
  1  итог собран, но есть пометки: нет саммари, есть окна с дословным текстом,
     провалены проверки таймкодов или не вышли говорящие — тогда расшифровка
     идёт без ярлыков, а в документе стоит пометка с причиной
  2  расшифровку получить не удалось, итогового .md нет

Примеры:
  bash scripts/transcribe_lecture.sh --input "vidio_example/Лекция 1.webm"
  bash scripts/transcribe_lecture.sh --input запись.mp4 --window 60 --no-chunks
  bash scripts/transcribe_lecture.sh --input запись.mp4 --no-diarize
  bash scripts/transcribe_lecture.sh --from-transcript \
      "data/output/Лекция 1-2026-10-03/transcript.json" --input "Лекция 1.webm" \
      --diarization "data/output/Лекция 1-2026-10-03/diarization.json"
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
    --no-diarize)      DIARIZE=0; shift ;;
    --diarization)     [ $# -ge 2 ] || die "--diarization требует путь к diarization.json"
                       DIARIZATION_IN="$2"; shift 2 ;;
    --keep-chunks)     KEEP_CHUNKS=1; shift ;;
    --no-chunks)       KEEP_CHUNKS=0; shift ;;
    --clean)           CLEAN=1; shift ;;
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

if [ -n "$DIARIZATION_IN" ]; then
  [ "$DIARIZE" -eq 1 ] \
    || die "--no-diarize и --diarization вместе не имеют смысла: первый выключает говорящих, второй требует их посчитать"
  [ -f "$DIARIZATION_IN" ] || die "не найден файл диаризации: $DIARIZATION_IN"
  [ -r "$DIARIZATION_IN" ] || die "файл диаризации недоступен для чтения: $DIARIZATION_IN"
fi

# Шаги говорящих либо выполняются целиком, либо не выполняются вовсе, и решается
# это здесь, до начала работы: от этого зависят и проверка наличия tools/*.py, и
# предупреждение про время pyannote. Пропуск по просьбе («не надо» или «звука
# уже нет») — это не пометка: такой прогон остаётся безупречным.
SPEAKER_STEPS=1
SKIP_REASON=""
if [ "$DIARIZE" -eq 0 ]; then
  SPEAKER_STEPS=0
  SKIP_REASON="--no-diarize"
elif [ -n "$FROM_TRANSCRIPT" ] && [ -z "$DIARIZATION_IN" ]; then
  # Диаризации нужен звук, а при --from-transcript его нет: WAV не готовится, а
  # исходной записи на машине может не быть вовсе. Готовый файл тут и подкладывают
  # флагом --diarization — тогда фразы и роли считаются как обычно.
  SPEAKER_STEPS=0
  SKIP_REASON="--from-transcript без --diarization"
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

REQUIRED_TOOLS=(chunker transcribe outputs check_timecodes punctuate summarize render_doc)
# cleanup_runs.py спрашивается только при --clean: без флага чистки не будет, и
# обрывать из-за него прогон, которому этот файл не понадобится, незачем. Зато с
# флагом отсутствие инструмента должно вылезти здесь, до получаса распознавания,
# а не в самом конце, когда убирать уже нечем.
if [ "$CLEAN" -eq 1 ]; then
  REQUIRED_TOOLS+=(cleanup_runs)
fi
# Инструменты говорящих спрашиваются по тому же правилу: только когда эти шаги
# будут. При --no-diarize и при --from-transcript без --diarization их никто не
# позовёт, и обрывать из-за них прогон незачем.
if [ "$SPEAKER_STEPS" -eq 1 ]; then
  REQUIRED_TOOLS+=(diarize speakers roles)
fi
for tool in "${REQUIRED_TOOLS[@]}"; do
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
  warn "claude в PATH не найден: расшифровка будет собрана, но без пунктуации, саммари и ролей говорящих (ярлыками станут «Говорящий N»), код возврата 1"
  CLAUDE_NOTE="НЕТ — пунктуации, саммари и ролей не будет"
fi

# Диаризация — самый долгий шаг конвейера после распознавания, и человек должен
# узнать об этом до того, как уйдёт пить чай на полчаса, а не после. Говорится
# и про кэш: веса pyannote не скачиваются ни при каком раскладе.
DIARIZE_NOTE="pyannote, устройство $DEVICE"
if [ "$SPEAKER_STEPS" -eq 0 ]; then
  DIARIZE_NOTE="выключена — $SKIP_REASON"
elif [ -n "$DIARIZATION_IN" ]; then
  DIARIZE_NOTE="готовый файл: $DIARIZATION_IN (pyannote не запускается)"
else
  warn "диаризация на CPU считает примерно в реальном времени: на записи 02:35:29 это десятки минут. Нужен заранее подготовленный локальный кэш pyannote — весов инструмент не скачивает; без кэша шаг вернёт код 2, документ соберётся без ярлыков говорящих, код возврата 1. Не нужны говорящие — --no-diarize; есть готовый файл — --diarization <путь>"
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
DIARIZATION="$WORKDIR/diarization.json"
PUNCTUATED="$WORKDIR/punctuated.json"
SPEAKERS="$WORKDIR/speakers.json"
ROLES="$WORKDIR/roles.json"
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
printf '  диаризация:     %s\n' "$DIARIZE_NOTE"

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

# =========================== 6. ДИАРИЗАЦИЯ ===========================

# Состояние шагов говорящих: DIARIZATION_OK говорит, есть ли годный
# diarization.json; SPEAKERS_ARG и ROLES_ARG — файлы, которые уйдут в
# summarize.py и render_doc.py; SPEAKERS_NOTE — причина, по которой ярлыков в
# документе не будет. Пустая причина означает «ярлыки в порядке» или «шаги
# пропущены по просьбе» — пометки в документе в обоих случаях нет.
DIARIZATION_OK=0
SPEAKERS_ARG=""
ROLES_ARG=""
SPEAKERS_NOTE=""

if [ "$NO_DOC" -ne 0 ]; then
  step "диаризация" "пропущено — нет расшифровки"
elif [ "$SPEAKER_STEPS" -eq 0 ]; then
  step "диаризация" "пропущено по просьбе — $SKIP_REASON"
elif [ -n "$DIARIZATION_IN" ]; then
  # Готовый файл переносится в рабочий подкаталог по той же причине, что и
  # транскрипт при --from-transcript: итог и все его источники должны лежать
  # вместе, иначе через месяц не восстановить, откуда в документе эти ярлыки.
  printf '\n================ ГОТОВАЯ ДИАРИЗАЦИЯ ================\n'
  copied=1
  if [ "$(abs_path "$DIARIZATION_IN")" = "$(abs_path "$DIARIZATION")" ]; then
    printf '  файл уже лежит в рабочем подкаталоге: %s\n' "$DIARIZATION"
  else
    printf '$ cp %s %s\n' "$DIARIZATION_IN" "$DIARIZATION"
    cp "$DIARIZATION_IN" "$DIARIZATION" || copied=0
  fi
  if [ "$copied" -eq 1 ] && [ -s "$DIARIZATION" ]; then
    DIARIZATION_OK=1
    step "диаризация" "ок — готовый файл (--diarization $DIARIZATION_IN)"
  else
    # Не остановка: ярлыки — не вся расшифровка. Документ соберётся без них, с
    # пометкой и кодом 1, как и при любой другой неудаче говорящих.
    step "диаризация" "ПРОВАЛ — не удалось взять готовый файл $DIARIZATION_IN"
    SPEAKERS_NOTE="не удалось взять готовый diarization.json: $DIARIZATION_IN"
    FLAWED=1
  fi
else
  printf '\n================ ДИАРИЗАЦИЯ ================\n'
  printf '$ %s tools/diarize.py --input %s --out %s --device %s\n' \
    "$PY" "$WAV" "$DIARIZATION" "$DEVICE"
  rc=0
  "$PY" tools/diarize.py --input "$WAV" --out "$DIARIZATION" --device "$DEVICE" || rc=$?
  if [ "$rc" -ge 2 ] || [ ! -s "$DIARIZATION" ]; then
    step "диаризация" "ПРОВАЛ — diarize.py вернул код $rc, файла нет"
    SPEAKERS_NOTE="диаризация не удалась: tools/diarize.py вернул код $rc"
    FLAWED=1
  elif [ "$rc" -eq 1 ]; then
    # Код 1 у diarize.py — файл записан, но говорящих ноль: речи не нашлось.
    # Привязывать фразы не к кому, поэтому дальше шаги говорящих не идут.
    step "диаризация" "РЕЧИ НЕ НАЙДЕНО — говорящих ноль (код $rc)"
    SPEAKERS_NOTE="диаризация не нашла речи: говорящих ноль"
    FLAWED=1
  else
    DIARIZATION_OK=1
    step "диаризация" "ок — $DIARIZATION"
  fi
fi

# ============================ 7. ПУНКТУАЦИЯ ============================

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

# ==================== 8 и 9. ФРАЗЫ, ГОВОРЯЩИЕ И РОЛИ ====================

# Оба шага идут одним блоком: роли считаются по фразам, и без speakers.json
# звать roles.py нечем. Границы фраз берутся из абзацев пунктуации, поэтому
# шаг стоит после неё, а не сразу за диаризацией.
if [ "$NO_DOC" -ne 0 ]; then
  step "фразы и говорящие" "пропущено — нет расшифровки"
  step "роли"              "пропущено — нет расшифровки"
elif [ "$SPEAKER_STEPS" -eq 0 ]; then
  step "фразы и говорящие" "пропущено по просьбе — $SKIP_REASON"
  step "роли"              "пропущено по просьбе — $SKIP_REASON"
elif [ "$DIARIZATION_OK" -eq 0 ]; then
  step "фразы и говорящие" "пропущено — годной диаризации нет"
  step "роли"              "пропущено — годной диаризации нет"
else
  printf '\n================ ФРАЗЫ И ГОВОРЯЩИЕ ================\n'
  printf '$ %s tools/speakers.py --transcript %s --punctuated %s --diarization %s --out %s\n' \
    "$PY" "$TRANSCRIPT" "$PUNCTUATED" "$DIARIZATION" "$SPEAKERS"
  rc=0
  "$PY" tools/speakers.py --transcript "$TRANSCRIPT" --punctuated "$PUNCTUATED" \
    --diarization "$DIARIZATION" --out "$SPEAKERS" || rc=$?
  if [ "$rc" -ge 2 ] || [ ! -s "$SPEAKERS" ]; then
    step "фразы и говорящие" "ПРОВАЛ — speakers.py вернул код $rc, фраз нет"
    step "роли"              "пропущено — фраз с говорящими нет"
    SPEAKERS_NOTE="привязка говорящих не удалась: tools/speakers.py вернул код $rc"
    FLAWED=1
  else
    # Код 1 у speakers.py — файл годен, но в нём есть фразы без говорящего или
    # невыровненные предложения. Ярлыки всё равно будут (у таких фраз их просто
    # нет), поэтому файл идёт дальше, а прогон помечается.
    SPEAKERS_ARG="$SPEAKERS"
    if [ "$rc" -eq 1 ]; then
      step "фразы и говорящие" "С ПОМЕТКАМИ — есть фразы без говорящего или невыровненные предложения (код $rc)"
      FLAWED=1
    else
      step "фразы и говорящие" "ок — $SPEAKERS"
    fi

    printf '\n================ РОЛИ ГОВОРЯЩИХ ================\n'
    printf '$ %s tools/roles.py --speakers %s --out %s --log-dir %s\n' \
      "$PY" "$SPEAKERS" "$ROLES" "$CLAUDE_LOGS"
    rc=0
    "$PY" tools/roles.py --speakers "$SPEAKERS" --out "$ROLES" \
      --log-dir "$CLAUDE_LOGS" || rc=$?
    if [ "$rc" -ge 2 ] || [ ! -s "$ROLES" ]; then
      # Неудача любого из трёх шагов — это документ без ярлыков целиком:
      # --speakers без --roles дал бы расшифровку с «Говорящий N» вместо ролей,
      # а это ровно тот случай, который roles.py закрывает своим кодом 1.
      step "роли" "ПРОВАЛ — roles.py вернул код $rc, файла нет"
      SPEAKERS_ARG=""
      SPEAKERS_NOTE="роли говорящих определить не удалось: tools/roles.py вернул код $rc"
      FLAWED=1
    else
      ROLES_ARG="$ROLES"
      if [ "$rc" -eq 1 ]; then
        step "роли" "ОТКАТ К ИМЕНАМ — ролей не вышло, ярлыками станут «Говорящий N» (код $rc)"
        FLAWED=1
      else
        step "роли" "ок — $ROLES"
      fi
    fi
  fi
fi

# Ярлыки идут в документ и в саммари только парой: имя говорящего без роли —
# это и есть откат, о котором roles.py сообщает своим кодом 1.
LABELS=0
if [ -n "$SPEAKERS_ARG" ] && [ -n "$ROLES_ARG" ]; then
  LABELS=1
fi

# ============================= 10. САММАРИ =============================

SUMMARY_ARG=""
if [ "$NO_DOC" -eq 0 ]; then
  printf '\n================ САММАРИ ================\n'
  # Команда собирается массивом, а не шестью ветками печати и запуска: ярлыки
  # добавляют к ней пару флагов, и держать оба списка в соответствии вручную —
  # верный способ напечатать одно, а запустить другое.
  SUMMARIZE_ARGS=(--punctuated "$PUNCTUATED")
  if [ "$LABELS" -eq 1 ]; then
    SUMMARIZE_ARGS+=(--speakers "$SPEAKERS_ARG" --roles "$ROLES_ARG")
  fi
  SUMMARIZE_ARGS+=(--out "$SUMMARY" --log-dir "$CLAUDE_LOGS")
  printf '$ %s tools/summarize.py' "$PY"
  printf ' %s' "${SUMMARIZE_ARGS[@]}"
  printf '\n'
  rc=0
  "$PY" tools/summarize.py "${SUMMARIZE_ARGS[@]}" || rc=$?
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

# ======================== 11. ИТОГОВЫЙ ДОКУМЕНТ ========================

if [ "$NO_DOC" -eq 0 ]; then
  printf '\n================ ИТОГОВЫЙ ДОКУМЕНТ ================\n'
  RENDER_ARGS=(--punctuated "$PUNCTUATED")
  if [ -n "$SUMMARY_ARG" ]; then
    RENDER_ARGS+=(--summary "$SUMMARY_ARG")
  fi
  if [ "$LABELS" -eq 1 ]; then
    RENDER_ARGS+=(--speakers "$SPEAKERS_ARG" --roles "$ROLES_ARG")
  fi
  # Пометка ставится ровно тогда, когда ярлыков не будет из-за неудачи, а не
  # из-за просьбы: при --no-diarize и при --from-transcript без --diarization
  # причина пуста, и документ выглядит как до появления говорящих.
  if [ -n "$SPEAKERS_NOTE" ]; then
    RENDER_ARGS+=(--speakers-note "$SPEAKERS_NOTE")
  fi
  RENDER_ARGS+=(--out "$DOC" --title "$BASE")
  printf '$ %s tools/render_doc.py' "$PY"
  printf ' %s' "${RENDER_ARGS[@]}"
  printf '\n'
  rc=0
  "$PY" tools/render_doc.py "${RENDER_ARGS[@]}" || rc=$?
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
            checks.json checks.txt diarization.json punctuated.json \
            speakers.json roles.json summary.md; do
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

# ============================= 12. ЧИСТКА =============================

# Место выбрано так: список «Файлы прогона» выше показывает подкаталог таким,
# каким он был до чистки, а последней строкой вывода обязан остаться путь к
# документу — значит, чистке остаётся ровно промежуток между ними. Своего
# удаления здесь нет ни строки: что лишнее, а что хранить, решает
# tools/cleanup_runs.py, и список-хранитель живёт в одном месте, а не в двух.
if [ "$CLEAN" -eq 1 ]; then
  printf '\n================ ЧИСТКА РАБОЧЕГО ПОДКАТАЛОГА ================\n'
  if [ "$NO_DOC" -ne 0 ]; then
    printf 'Чистка пропущена: итогового документа нет (код возврата 2). Промежуточные\n'
    printf 'файлы — всё, что осталось от прогона, и удалять их было бы потерей.\n'
  elif [ "$FLAWED" -ne 0 ]; then
    printf 'Чистка пропущена: прогон прошёл с пометками (код возврата 1). Разбираться\n'
    printf 'с ними придётся по промежуточным файлам, поэтому они остаются на месте.\n'
  else
    printf '$ %s tools/cleanup_runs.py --run %s\n' "$PY" "$WORKDIR"
    rc=0
    # run.log чистка удалит вместе с остальным, но вывод после этого никуда не
    # девается: tee держит файл открытым и дописывает уже в отвязанный inode, а
    # на экран всё идёт как обычно. Просто лога на диске после --clean не будет.
    "$PY" tools/cleanup_runs.py --run "$WORKDIR" || rc=$?
    if [ "$rc" -ne 0 ]; then
      # Неубранный мусор не делает удачный прогон неудачным: документ собран и
      # лежит на месте, а неудаляемые файлы — повод зайти в каталог руками.
      # Поэтому предупреждение, а не step и не FLAWED=1.
      warn "чистка не удалась: cleanup_runs.py вернул код $rc, подкаталог $WORKDIR остался как есть"
    fi
  fi
fi

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
