#!/usr/bin/env python3
"""
score.py — WER, WER с маскировкой числительных и CER по разделу 4 протокола.

Что считает. Берёт эталонную расшифровку и гипотезу (вывод любого кандидата),
прогоняет обе через одну и ту же нормализацию — шаги 1–8 раздела 4
docs/research/benchmark-plan.md — и печатает основной WER, диагностический WER
с замаскированными числительными, CER, разбивку ошибок на замены/удаления/
вставки и число слов в эталоне. Разбивка нужна не для отчёта, а для
диагностики: много вставок — признак галлюцинаций, много удалений — модель
проглатывает речь.

Почему метрик две. Нормализация чисел сама создаёт ошибки, причём не одинаково
для всех кандидатов: из «2024» падеж не восстановить, num2words даёт «две
тысячи двадцать четыре», а сказано «в две тысячи двадцать четвёртом». Это бьёт
по моделям, которые пишут цифрами (Whisper и его порты), и не бьёт по тем,
которые пишут словами (GigaAM, Vosk). Поэтому вторая метрика заменяет все
числительные в обоих текстах на единый токен <число>. Подробности и замер —
раздел 4 протокола, подраздел «Почему шага „цифры → слова“ недостаточно».

Код нормализации (_build_vocab, _is_number_word, _number_to_words, normalize)
перенесён из листинга раздела 4 без изменений в поведении: числа 9,09 % → 0,00 %
→ CER 5,38 % на контрольной фразе протокола получены им и служат проверкой
переноса — она входит в --selftest. Порядок шагов менять нельзя: числа
разворачиваются до удаления пунктуации, иначе «2,5» успеет стать «2 5».

Что умеет сверх листинга:
  * разбирает эталон в формате строк «[ММ:СС] Роль: текст» — метка времени и
    ярлык роли снимаются до нормализации, иначе они попадут в WER;
  * выбрасывает из сравниваемого текста пометки в квадратных скобках
    ([нрзб], [реплика]) и говорит, сколько их было;
  * читает гипотезу и из .txt, и из JSON схемы transcript/1 (берётся поле text);
  * показывает топ пар ошибок «сказано → распознано» и отдельно считает, сколько
    ошибок приходится на пары «латиница в эталоне → кириллица в гипотезе»
    (HR → эйчар, DevOps → девопс). Это только диагностика: на WER счётчик не
    влияет, такое расхождение считается обычной ошибкой (решение владельца);
  * пишет машинночитаемый итог схемы score/1 по флагу --json.

Чего не делает. Не правит ни эталон, ни гипотезу и вообще ничего не пишет,
кроме файла --json: правка эталона после того, как получены чьи-то WER,
обнуляет сравнение (раздел 3 протокола). Не считает сдвиг таймкодов — это
отдельный инструмент. Не ходит в сеть и не трогает модели.

Зависимости: стандартная библиотека плюс jiwer 4.0.0 и num2words 0.5.14 из
.venv проекта. Версии пинятся: поведение jiwer на пустом эталоне менялось в 4.0.
Обе библиотеки импортируются лениво, поэтому --help работает и без них; если их
нет, инструмент останавливается с понятным сообщением и ничего не ставит сам.

Использование:
  python tools/score.py --ref эталон.txt --hyp data/transcripts/v3_rnnt.json
  python tools/score.py --ref эталон.txt --hyp гипотеза.txt --json score.json
  python tools/score.py --selftest

Пути с пробелами, скобками и кириллицей поддерживаются — передавайте в кавычках.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROG = "score.py"

# Версия формата машинночитаемого итога.
SCHEMA = "score/1"

# Схема JSON, который умеет читать --hyp (его пишет tools/transcribe.py).
TRANSCRIPT_SCHEMA = "transcript/1"

PLAN_REFERENCE = "docs/research/benchmark-plan.md, раздел 4"

# Сколько пар ошибок печатать по умолчанию.
DEFAULT_TOP = 15

# Расхождение основного и маскированного WER больше этого (в п. п.) означает,
# что решение принимается по маскированному — раздел 8, шаг 2.
MASK_GAP_WARN = 1.0


# --------------------------------------------------------------- ошибки ---


class ToolError(Exception):
    """Ошибка окружения или данных: сообщение уже на русском и готово в stderr."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


class UsageError(ToolError):
    """Ошибка в аргументах командной строки: вход непригоден, код 2."""

    def __init__(self, message: str):
        super().__init__(message, code=2)


def die(message: str, code: int = 1) -> int:
    """Печатает понятную ошибку и возвращает код выхода (трейсбека не будет)."""
    print(f"{PROG}: {message}", file=sys.stderr)
    return code


# ------------------------------------------------- внешние библиотеки ---


_LIBS: tuple[object, object] | None = None


def libs() -> tuple[object, object]:
    """Возвращает (jiwer, num2words), импортируя их при первом обращении.

    Ленивый импорт нужен, чтобы --help оставался полезным на машине, где
    окружение ещё не собрано. Ставить что-либо самим нельзя: прогон замера
    обязан идти без сети, а молча подтянутая другая версия jiwer сделает два
    прогона несравнимыми (см. раздел 4, шаг 8).
    """
    global _LIBS
    if _LIBS is None:
        try:
            import jiwer
            from num2words import num2words
        except ImportError as exc:
            raise ToolError(
                f"нет библиотеки для подсчёта метрик ({exc.name}). "
                'Поставьте их в окружение проекта: uv pip install --python '
                '.venv/bin/python "jiwer==4.0.0" "num2words==0.5.14". '
                "Сам инструмент ничего не устанавливает: замер идёт без сети",
                code=2,
            ) from None
        _LIBS = (jiwer, num2words)
    return _LIBS


# --------------------------------------------------------- нормализация ---
#
# Всё ниже до пометки «конец переноса» — код из листинга раздела 4 протокола.
# Поведение менять нельзя: числа 9,09 / 6,25 / 0,00 из протокола получены им, и
# любое расхождение означает, что перенос сломан. Единственная правка —
# num2words берётся через libs(), чтобы импорт оставался ленивым.

UNITS = {"%": " процентов ", "№": " номер ", "°": " градусов "}
PUNCT = r"[.,!?;:()«»\"'…]"
MASK = "<число>"

# Словарь числительных нужен только для диагностической метрики. Он генерируется
# из того же num2words, которым мы разворачиваем цифры, — иначе два списка
# неизбежно разойдутся. _EXTRA — косвенные падежи, которых num2words не выдаёт.
_NUMS = list(range(101)) + [200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 10**6]
# Слова единиц маскируются вместе с числами: шаг 4 нормализации разворачивает «%»
# в «процентов» всегда, а в речи падеж другой — без этой строки диагностическая
# метрика показывала 6,25 % вместо 0,00 % (замер 2026-10-03, см. протокол).
_UNITS_WORDS = """процент процента процентов процентах процентами проценты
номер номера номеров градус градуса градусов"""
_EXTRA = """целых десятых сотых тысячных тысяч тысячи тысяча тысячного
двух трех четырех пяти шести семи восьми девяти десяти одиннадцати двенадцати
тринадцати четырнадцати пятнадцати шестнадцати семнадцати восемнадцати девятнадцати
двадцати тридцати сорока пятидесяти шестидесяти семидесяти восьмидесяти девяноста
ста двумя тремя четырьмя одного одной одним одному двум трем третьего третьему
третьем полтора половина миллион миллиона миллионов миллиард"""
# Порядок важен: длинные окончания идут раньше коротких.
_ENDINGS = ("ьего", "ьему", "ьем", "ого", "ему", "ому", "ыми", "ими", "ах", "ях",
            "ом", "ем", "ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие",
            "ых", "их", "ым", "им", "ух", "ум", "ех",
            "а", "е", "и", "о", "у", "ы", "ь", "ю", "я")


def _yo(text: str) -> str:
    return text.replace("ё", "е")


# Порог в 4 символа отсекает огрызки вроде «дв» и «тр», которые иначе совпали бы
# с обычными словами («тру», «две» ловятся точным совпадением по словарю).
def _stems(word: str) -> set:
    return {word[: -len(s)] for s in _ENDINGS
            if word.endswith(s) and len(word) - len(s) >= 4}


def _build_vocab() -> tuple:
    num2words = libs()[1]
    vocab = set(_yo(_EXTRA).split()) | set(_yo(_UNITS_WORDS).split())
    for n in _NUMS:
        for to in ("cardinal", "ordinal"):
            vocab.update(_yo(num2words(n, lang="ru", to=to)).split())
    stems = set()
    for word in vocab:
        stems |= _stems(word)
    return vocab, stems


_VOCAB: tuple | None = None


def vocab() -> tuple:
    """Словарь и стемы числительных, построенные один раз на процесс."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _build_vocab()
    return _VOCAB


def _is_number_word(token: str) -> bool:
    VOCAB, STEMS = vocab()
    if token in VOCAB:
        return True
    for suffix in _ENDINGS:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            base = token[: -len(suffix)]
            if base in STEMS or base in VOCAB:
                return True
    return False


def _number_to_words(match: re.Match) -> str:
    num2words = libs()[1]
    raw = match.group(0).replace(",", ".")
    # Дробные читаются как «целых/десятых», num2words это умеет только для float.
    value = float(raw) if "." in raw else int(raw)
    return f" {num2words(value, lang='ru')} "


def normalize(text: str, mask_numbers: bool = False) -> str:
    text = _yo(text.lower())
    for sign, word in UNITS.items():
        text = text.replace(sign, word)
    text = re.sub(r"\d+(?:[.,]\d+)?", _number_to_words, text)
    text = re.sub(PUNCT, " ", text)
    text = text.replace("-", " ").replace("—", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if mask_numbers:
        text = " ".join(MASK if _is_number_word(t) else t for t in text.split())
        # «2024» -> четыре слова -> четыре маски; склеиваем их в одну, иначе
        # число из двух слов считалось бы за четыре совпадения, а из одного — за одно.
        text = re.sub(f"(?:{re.escape(MASK)} )+{re.escape(MASK)}", MASK, text)
    return text


# --- конец переноса из раздела 4 -------------------------------------------


# ------------------------------------------------------- разбор эталона ---

# Метка времени в начале строки: [ММ:СС] и [ЧЧ:ММ:СС]. Пробелы внутри скобок
# допускаются, потому что эталон пишет человек.
TIMECODE_RE = re.compile(r"^\[\s*\d{1,3}:[0-5]\d(?::[0-5]\d)?\s*\]\s*")

# Ярлык роли: «Менеджер:», «Клиент 2:», «Голос из зала:». Двоеточие обязано
# быть первым на строке и за ним обязан идти пробел или конец строки, иначе это
# не ярлык, а «http://…» или «15:30» внутри фразы.
ROLE_RE = re.compile(r"^([^:\[\]]{1,60}):(?=\s|$)\s*")

# Больше трёх слов — это уже не роль, а начало фразы с двоеточием
# («[00:10] Смотрите как это работает: мы берём…»), такое снимать нельзя.
MAX_ROLE_WORDS = 3

# Пометки в квадратных скобках внутри текста: [нрзб], [реплика], [смех].
# Вложенных скобок в эталоне не бывает, поэтому класс без «[» и «]».
NOTE_RE = re.compile(r"\[[^\[\]]*\]")

LATIN_RE = re.compile(r"[a-z]")
CYRILLIC_RE = re.compile(r"[а-яё]")


@dataclass
class RefParse:
    """Результат разбора эталона построчно."""

    text: str
    lines: int = 0  # непустых строк всего
    timecoded: int = 0  # из них со снятой меткой времени
    roles: int = 0  # из них со снятым ярлыком роли


def parse_reference(raw: str) -> RefParse:
    """Снимает с эталона метки времени и ярлыки ролей — до нормализации.

    Ярлык снимается только сразу после метки времени: двоеточие в середине
    реплики принадлежит речи, а не разметке. Файл эталона при этом не трогается
    ни при каких условиях — вся подготовка текста идёт на лету (раздел 3
    протокола: правка эталона после получения WER обнуляет сравнение).
    """
    kept: list[str] = []
    result = RefParse(text="")
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        result.lines += 1
        stamp = TIMECODE_RE.match(line)
        if stamp:
            result.timecoded += 1
            line = line[stamp.end():]
            role = ROLE_RE.match(line)
            if role and 1 <= len(role.group(1).split()) <= MAX_ROLE_WORDS:
                result.roles += 1
                line = line[role.end():]
        if line:
            kept.append(line)
    result.text = "\n".join(kept)
    return result


def strip_notes(text: str) -> tuple[str, int]:
    """Убирает пометки в квадратных скобках и возвращает их количество.

    Сравнивать по ним нечего: [нрзб] — это признание, что слово не разобрали, а
    [реплика] — разметка говорящего. Оставь их в тексте — и каждая пометка
    станет лишним удалением у всех кандидатов сразу.
    """
    count = 0

    def replace(_match: re.Match) -> str:
        nonlocal count
        count += 1
        return " "

    return NOTE_RE.sub(replace, text), count


# ---------------------------------------------------------- чтение входа ---


def read_input(path: Path, role: str) -> tuple[str, str]:
    """Читает .txt или JSON схемы transcript/1 и возвращает (текст, вид файла)."""
    if not path.exists():
        raise ToolError(f"{role}: файл не найден: {path}", code=2)
    if path.is_dir():
        raise ToolError(f"{role}: это каталог, а не файл: {path}", code=2)
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{role}: файл не читается как UTF-8: {path}", code=2) from None

    looks_json = path.suffix.lower() == ".json" or raw.lstrip().startswith("{")
    if not looks_json:
        return raw, "текст"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"{role}: битый JSON ({exc}): {path}", code=2) from None
    if not isinstance(data, dict):
        raise ToolError(f"{role}: в JSON ожидался объект, а не {type(data).__name__}", code=2)
    schema = data.get("schema")
    if schema != TRANSCRIPT_SCHEMA:
        raise ToolError(
            f"{role}: ожидалась схема {TRANSCRIPT_SCHEMA}, в файле: {schema!r}", code=2
        )
    text = data.get("text")
    if not isinstance(text, str):
        raise ToolError(
            f"{role}: в JSON схемы {TRANSCRIPT_SCHEMA} нет строкового поля text", code=2
        )
    return text, f"JSON {TRANSCRIPT_SCHEMA}"


# ------------------------------------------------------------- метрики ---


@dataclass
class Pair:
    """Пара ошибки «сказано → распознано» с частотой."""

    ref: str
    hyp: str
    count: int
    latin: bool

    def as_dict(self) -> dict:
        return {"ref": self.ref, "hyp": self.hyp, "count": self.count, "latin": self.latin}


@dataclass
class Score:
    """Всё, что инструмент насчитал по паре текстов."""

    ref_words: int
    wer: float
    wer_masked: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    pairs: list[Pair] = field(default_factory=list)
    latin_count: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def latin_share(self) -> float:
        """Доля ошибок, пришедшихся на пары «латиница → кириллица», в долях единицы."""
        return self.latin_count / self.errors if self.errors else 0.0


def is_latin_pair(ref_word: str, hyp_word: str) -> bool:
    """Латиница в эталоне против кириллицы в гипотезе: HR → эйчар, DevOps → девопс.

    Счётчик диагностический. Решение владельца: ничего к нормализации не
    добавляем, такое расхождение — обычная ошибка; в отчёте надо лишь показать,
    какая доля ошибок на него приходится.
    """
    return bool(
        LATIN_RE.search(ref_word)
        and CYRILLIC_RE.search(hyp_word)
        and not LATIN_RE.search(hyp_word)
    )


def collect_pairs(output) -> tuple[list[Pair], int]:
    """Собирает пары замен из выравнивания jiwer: (пары по убыванию частоты, латинских).

    Удаления и вставки в пары не идут: у них нет второй половины, а «сказано →
    распознано» имеет смысл только там, где модель слово услышала, но услышала
    не так.
    """
    counts: dict[tuple[str, str], int] = {}
    for alignment, references, hypotheses in zip(
        output.alignments, output.references, output.hypotheses
    ):
        for chunk in alignment:
            if chunk.type != "substitute":
                continue
            ref_slice = references[chunk.ref_start_idx:chunk.ref_end_idx]
            hyp_slice = hypotheses[chunk.hyp_start_idx:chunk.hyp_end_idx]
            for ref_word, hyp_word in zip(ref_slice, hyp_slice):
                key = (ref_word, hyp_word)
                counts[key] = counts.get(key, 0) + 1

    pairs = [
        Pair(ref=ref_word, hyp=hyp_word, count=count, latin=is_latin_pair(ref_word, hyp_word))
        for (ref_word, hyp_word), count in counts.items()
    ]
    # Сортировка по частоте, при равной частоте — по алфавиту: вывод должен быть
    # одинаковым от прогона к прогону, иначе отчёты не сравнить diff'ом.
    pairs.sort(key=lambda pair: (-pair.count, pair.ref, pair.hyp))
    latin_count = sum(pair.count for pair in pairs if pair.latin)
    return pairs, latin_count


def compute(raw_ref: str, raw_hyp: str) -> Score:
    """Считает все метрики по уже подготовленным (но не нормализованным) текстам."""
    jiwer = libs()[0]

    ref, hyp = normalize(raw_ref), normalize(raw_hyp)
    if not ref:
        raise ToolError(
            "эталон пуст после нормализации: сравнивать не с чем "
            "(проверьте --ref: не тот файл, пустой файл или только разметка)",
            code=1,
        )
    ref_masked, hyp_masked = normalize(raw_ref, True), normalize(raw_hyp, True)

    output = jiwer.process_words(ref, hyp)
    pairs, latin_count = collect_pairs(output)
    return Score(
        ref_words=len(ref.split()),
        wer=jiwer.wer(ref, hyp) * 100,
        wer_masked=jiwer.wer(ref_masked, hyp_masked) * 100,
        cer=jiwer.cer(ref, hyp) * 100,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        pairs=pairs,
        latin_count=latin_count,
    )


# -------------------------------------------------------------- вывод ---


def print_report(
    score: Score,
    ref_path: Path,
    hyp_path: Path,
    hyp_kind: str,
    parsed: RefParse,
    notes_ref: int,
    notes_hyp: int,
    top: int,
) -> None:
    print(f"эталон:   {ref_path}")
    print(f"гипотеза: {hyp_path} ({hyp_kind})")
    print(
        f"строк эталона разобрано: {parsed.lines}"
        f" (с меткой времени: {parsed.timecoded}), снято ярлыков роли: {parsed.roles}"
    )
    print(
        f"пометок в квадратных скобках удалено: {notes_ref + notes_hyp}"
        f" (эталон {notes_ref}, гипотеза {notes_hyp})"
    )
    print()
    print(f"WER:            {score.wer:.2f} %")
    print(f"WER без чисел:  {score.wer_masked:.2f} %")
    print(f"CER:            {score.cer:.2f} %")
    print(
        f"замен: {score.substitutions}  удалений: {score.deletions}"
        f"  вставок: {score.insertions}"
    )
    print(f"слов в эталоне: {score.ref_words}")
    if abs(score.wer - score.wer_masked) > MASK_GAP_WARN:
        print(
            f"ВНИМАНИЕ: числа дают {score.wer - score.wer_masked:+.2f} п. п. — "
            f"решение принимать по колонке «без чисел» (раздел 8, шаг 2)"
        )

    print()
    if score.pairs:
        shown = score.pairs[:top]
        print(
            f"пары ошибок «сказано → распознано»: показано {len(shown)}"
            f" из {len(score.pairs)}"
        )
        width = max(len(pair.ref) for pair in shown)
        for pair in shown:
            mark = "  ← латиница" if pair.latin else ""
            print(f"  {pair.count:4d}  {pair.ref:<{width}} → {pair.hyp}{mark}")
    else:
        print("пар ошибок нет: замен в выравнивании не было")

    print(
        f"ошибок на парах «латиница → кириллица»: {score.latin_count}"
        f" из {score.errors} ({score.latin_share * 100:.2f} % ошибок)"
    )


def build_machine_report(
    score: Score,
    ref_path: Path,
    hyp_path: Path,
    hyp_kind: str,
    parsed: RefParse,
    notes_ref: int,
    notes_hyp: int,
) -> dict:
    """Итог схемы score/1. Все три метрики — в процентах, как и в печатном выводе."""
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "units": "percent",
        "reference": PLAN_REFERENCE,
        "ref": str(ref_path),
        "hyp": str(hyp_path),
        "hyp_kind": hyp_kind,
        "ref_words": score.ref_words,
        "wer": round(score.wer, 4),
        "wer_masked": round(score.wer_masked, 4),
        "cer": round(score.cer, 4),
        "substitutions": score.substitutions,
        "deletions": score.deletions,
        "insertions": score.insertions,
        "errors": score.errors,
        "pairs": [pair.as_dict() for pair in score.pairs],
        "latin_pairs": {
            "count": score.latin_count,
            "share_of_errors": round(score.latin_share, 6),
        },
        "ref_lines": {
            "lines": parsed.lines,
            "timecoded": parsed.timecoded,
            "roles_stripped": parsed.roles,
        },
        "notes_removed": {"ref": notes_ref, "hyp": notes_hyp},
    }


def write_machine_report(report: dict, target: str) -> None:
    """Пишет итог в файл или в stdout по «-»."""
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if target == "-":
        sys.stdout.write(text)
        return
    path = Path(target)
    if path.parent and not path.parent.exists():
        raise ToolError(f"--json: каталога нет: {path.parent}", code=2)
    path.write_text(text, encoding="utf-8")
    print(f"машинночитаемый итог: {path}")


# ----------------------------------------------------------------- прогон ---


@dataclass
class Options:
    ref: Path | None = None
    hyp: Path | None = None
    json_path: str | None = None
    top: int = DEFAULT_TOP
    selftest: bool = False
    help: bool = False


def run(opts: Options) -> int:
    assert opts.ref is not None and opts.hyp is not None  # проверено в main()

    raw_ref, _ = read_input(opts.ref, "--ref")
    raw_hyp, hyp_kind = read_input(opts.hyp, "--hyp")

    # Разбор построчно — только для эталона: метки времени и ярлыки ролей есть
    # в нём, а гипотеза приходит сплошным текстом от модели.
    parsed = parse_reference(raw_ref)
    ref_text, notes_ref = strip_notes(parsed.text)
    hyp_text, notes_hyp = strip_notes(raw_hyp)

    score = compute(ref_text, hyp_text)
    print_report(
        score, opts.ref, opts.hyp, hyp_kind, parsed, notes_ref, notes_hyp, opts.top
    )
    if opts.json_path:
        report = build_machine_report(
            score, opts.ref, opts.hyp, hyp_kind, parsed, notes_ref, notes_hyp
        )
        write_machine_report(report, opts.json_path)
    return 0


# ---------------------------------------------------------------- CLI ---


USAGE = f"""Использование: python tools/score.py --ref <эталон> --hyp <гипотеза> [опции]

Считает WER, WER с замаскированными числительными и CER по протоколу
({PLAN_REFERENCE}). Эталон и гипотеза нормализуются одной и той же
функцией: нижний регистр, ё → е, цифры и символы единиц словами, без
пунктуации, дефис — пробелом. Дополнительно печатает разбивку ошибок, топ пар
«сказано → распознано» и долю ошибок на парах «латиница в эталоне → кириллица
в гипотезе».

Опции:
  --ref <путь>     эталонная расшифровка .txt (обязателен, кроме --selftest).
                   Строки вида «[ММ:СС] Роль: текст» разбираются: метка времени
                   и ярлык роли снимаются до нормализации; ярлык — только сразу
                   после метки и только если он не длиннее трёх слов
  --hyp <путь>     гипотеза: .txt или JSON схемы {TRANSCRIPT_SCHEMA}
                   (обязателен, кроме --selftest). У JSON берётся поле text
  --json <путь>    куда писать машинночитаемый итог схемы {SCHEMA}
                   («-» — в stdout)
  --top <N>        сколько пар ошибок печатать (по умолчанию {DEFAULT_TOP})
  --selftest       прогнать встроенные проверки на синтетике и выйти
  -h, --help       показать эту справку

Что ещё снимается с текста: пометки в квадратных скобках ([нрзб], [реплика])
удаляются из обоих текстов, их количество печатается отдельной строкой.

Коды возврата: 0 — метрики посчитаны, 1 — данные непригодны для сравнения
(пустой после нормализации эталон), 2 — вход непригоден (файла нет, битый JSON,
чужая схема, ошибка в аргументах, нет jiwer/num2words).

Инструмент ничего не правит: ни эталон, ни гипотезу. Единственный файл, который
он пишет, — указанный в --json.

Примеры:
  python tools/score.py --ref "data/samples/лекция.ref.txt" \\
      --hyp data/transcripts/v3_rnnt.json
  python tools/score.py --ref эталон.txt --hyp гипотеза.txt --top 30 --json -
  python tools/score.py --selftest
"""


def _need_value(argv: list[str], index: int, name: str) -> str:
    if index + 1 >= len(argv):
        raise UsageError(f"{name} требует значение")
    return argv[index + 1]


def parse_args(argv: list[str]) -> Options:
    """Разбирает аргументы вручную — ради сообщений на русском, как в соседних tools/."""
    opts = Options()
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("-h", "--help"):
            opts.help = True
            return opts
        if arg == "--selftest":
            opts.selftest = True
            index += 1
            continue
        if arg == "--ref":
            opts.ref = Path(_need_value(argv, index, arg))
            index += 2
            continue
        if arg == "--hyp":
            opts.hyp = Path(_need_value(argv, index, arg))
            index += 2
            continue
        if arg == "--json":
            opts.json_path = _need_value(argv, index, arg)
            index += 2
            continue
        if arg == "--top":
            raw = _need_value(argv, index, arg)
            try:
                opts.top = int(raw)
            except ValueError:
                raise UsageError(f"--top ожидает целое число, получено: {raw}") from None
            if opts.top < 1:
                raise UsageError(f"--top ожидает число больше нуля, получено: {raw}")
            index += 2
            continue
        raise UsageError(f"неизвестный аргумент: {arg} (см. --help)")
    return opts


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        opts = parse_args(args)
        if opts.help:
            print(USAGE, end="")
            return 0
        if opts.selftest:
            return selftest()
        if opts.ref is None:
            raise UsageError("не указан --ref (см. --help)")
        if opts.hyp is None:
            raise UsageError("не указан --hyp (см. --help)")
        return run(opts)
    except ToolError as exc:
        return die(str(exc), exc.code)
    except KeyboardInterrupt:
        return die("прервано пользователем")


# ------------------------------------------------------------ селф-тест ---
#
# Фикстуры — только синтетические фразы: ни строчки из реальной записи здесь
# быть не должно, файл лежит в git, а запись — нет.

# Контрольная пара из раздела 4 протокола. Фраза приведена целиком, иначе числа
# не воспроизводятся. Это единственный «не придуманный здесь» текст — он сам
# является частью протокола.
CONTROL_REF = (
    "В две тысячи двадцать четвёртом году мы обсуждали, что доля выросла "
    "на две целых пять десятых процента по сравнению с прошлым годом."
)
CONTROL_HYP = (
    "В 2024 году мы обсуждали, что доля выросла на 2,5% по сравнению с прошлым годом."
)


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Прогон main() с перехватом вывода: возвращает (код возврата, stdout).

    stderr тоже перехватывается и отбрасывается: часть фикстур намеренно
    непригодна, и её сообщения не должны мешать читать селф-тест.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
        code = main(argv)
    return code, buffer.getvalue()


def _fixture(directory: str, name: str, content: str) -> Path:
    path = Path(directory) / name
    path.write_text(content, encoding="utf-8")
    return path


def _check_control_phrase():
    """Числа протокола воспроизводятся: 9,09 % → 0,00 %, CER 5,38 %."""
    score = compute(CONTROL_REF, CONTROL_HYP)
    assert score.ref_words == 22, score.ref_words
    assert f"{score.wer:.2f}" == "9.09", score.wer
    assert f"{score.wer_masked:.2f}" == "0.00", score.wer_masked
    assert f"{score.cer:.2f}" == "5.38", score.cer
    assert (score.substitutions, score.deletions, score.insertions) == (2, 0, 0), score
    # Те самые два расхождения из таблицы протокола, и оба — ложные.
    pairs = {(pair.ref, pair.hyp) for pair in score.pairs}
    assert ("четвертом", "четыре") in pairs, pairs
    assert ("процента", "процентов") in pairs, pairs


def _check_control_phrase_cli():
    """Та же пара через CLI: код 0 и все обязательные строки в выводе."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", CONTROL_REF)
        hyp = _fixture(tmp, "hyp.txt", CONTROL_HYP)
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp)])
        assert code == 0, (code, out)
        for fragment in (
            "WER:            9.09 %",
            "WER без чисел:  0.00 %",
            "CER:            5.38 %",
            "замен: 2  удалений: 0  вставок: 0",
            "слов в эталоне: 22",
            "ВНИМАНИЕ: числа дают +9.09 п. п.",
        ):
            assert fragment in out, (fragment, out)
        # Эталон не изменился: его правка обнуляет сравнение (раздел 3).
        assert ref.read_text(encoding="utf-8") == CONTROL_REF
        assert hyp.read_text(encoding="utf-8") == CONTROL_HYP
        # Ничего, кроме фикстур, инструмент не создал.
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["hyp.txt", "ref.txt"]


def _check_masked_without_unit_words():
    """Слова единиц в словаре маскировки обязательны: без них было бы 6,25 %."""
    assert _is_number_word("процента"), "«процента» должно маскироваться"
    assert _is_number_word("процентов"), "«процентов» должно маскироваться"
    assert _is_number_word("четвертом") and _is_number_word("двадцать")
    # Контрольные наборы детектора из раздела 4: ложных срабатываний быть не должно.
    for word in ("стол", "стало", "тру", "лекция", "который", "момент", "около"):
        assert not _is_number_word(word), word


def _check_reference_lines():
    """Метки времени и ярлыки ролей снимаются, в WER не попадают."""
    parsed = parse_reference(
        "[00:00] Менеджер: Один два три\n"
        "\n"
        "[01:30] Клиент два: Четыре пять\n"
        "[02:00:05] Голос из зала: Шесть\n"
        "[03:10] А теперь смотрите внимательно: семь\n"
        "Строка без метки времени: восемь\n"
    )
    assert parsed.lines == 5, parsed
    assert parsed.timecoded == 4, parsed
    # Четвёртая строка — четыре слова до двоеточия, это уже не ярлык; пятая без
    # метки времени, значит её двоеточие трогать нельзя.
    assert parsed.roles == 3, parsed
    assert parsed.text == (
        "Один два три\n"
        "Четыре пять\n"
        "Шесть\n"
        "А теперь смотрите внимательно: семь\n"
        "Строка без метки времени: восемь"
    ), parsed.text
    # Сквозь CLI счётчики видны человеку.
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", "[00:00] Менеджер: один два\n[00:05] Клиент: три\n")
        hyp = _fixture(tmp, "hyp.txt", "один два три")
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp)])
        assert code == 0, (code, out)
        assert "строк эталона разобрано: 2 (с меткой времени: 2), снято ярлыков роли: 2" in out, out
        assert "WER:            0.00 %" in out, out


def _check_notes_removed():
    """[нрзб] и [реплика] выбрасываются из обоих текстов и считаются."""
    text, count = strip_notes("один [нрзб] два [реплика] три")
    assert count == 2, count
    assert normalize(text) == "один два три", text
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", "[00:00] Лектор: один [нрзб] два три")
        hyp = _fixture(tmp, "hyp.txt", "один два [шум] три")
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp)])
        assert code == 0, (code, out)
        assert "пометок в квадратных скобках удалено: 2 (эталон 1, гипотеза 1)" in out, out
        assert "WER:            0.00 %" in out, out


def _check_empty_reference():
    """Пустой после нормализации эталон — явная ошибка, а не деление на ноль."""
    # Эталон из одной разметки: метка времени, ярлык и [нрзб] снимаются, слов
    # не остаётся. Это самый вероятный способ наткнуться на деление на ноль.
    prepared, _ = strip_notes(parse_reference("[00:00] Ведущий:\n[нрзб]\n").text)
    try:
        compute(prepared, "что-то распознанное")
    except ToolError as exc:
        assert exc.code == 1, exc.code
        assert "эталон пуст после нормализации" in str(exc), exc
    else:
        raise AssertionError("ожидалась ToolError на пустом эталоне")

    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", "   \n\n[нрзб]\n")
        hyp = _fixture(tmp, "hyp.txt", "раз два три четыре пять")
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp)])
        assert code == 1, (code, out)
        assert "WER" not in out, out


def _check_substitution_pairs():
    """Пары «сказано → распознано» извлекаются и сортируются по частоте."""
    ref = "кошка сидела на окне кошка молчала кошка ушла собака лаяла"
    hyp = "кот сидела на окне кот молчала кот ушла пёс лаяла"
    score = compute(ref, hyp)
    assert score.substitutions == 4, score
    assert (score.deletions, score.insertions) == (0, 0), score
    assert [(p.ref, p.hyp, p.count) for p in score.pairs] == [
        ("кошка", "кот", 3),
        ("собака", "пес", 1),  # ё → е снимается нормализацией
    ], score.pairs
    assert all(not pair.latin for pair in score.pairs), score.pairs
    # --top режет печать, но не сам список пар.
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = _fixture(tmp, "ref.txt", ref)
        hyp_path = _fixture(tmp, "hyp.txt", hyp)
        code, out = _run_cli(["--ref", str(ref_path), "--hyp", str(hyp_path), "--top", "1"])
        assert code == 0, (code, out)
        assert "показано 1 из 2" in out, out
        assert "кошка → кот" in out, out
        assert "собака → пес" not in out, out


def _check_latin_pairs():
    """Пары «латиница в эталоне → кириллица в гипотезе» считаются отдельно."""
    assert is_latin_pair("hr", "эйчар")
    assert is_latin_pair("devops", "девопс")
    assert not is_latin_pair("эйчар", "эйчар")
    # Транслит в обратную сторону парой не считается: гипотеза должна быть
    # кириллической, иначе это обычное расхождение написания.
    assert not is_latin_pair("девопс", "devops")
    assert not is_latin_pair("hr", "hr manager")

    ref = "в HR пришёл DevOps и ещё раз HR а потом обычное слово"
    hyp = "в эйчар пришёл девопс и ещё раз эйчар а потом другое слово"
    score = compute(ref, hyp)
    assert score.substitutions == 4, score
    assert score.latin_count == 3, score.latin_count
    assert abs(score.latin_share - 0.75) < 1e-9, score.latin_share
    latin_pairs = {(p.ref, p.hyp, p.count) for p in score.pairs if p.latin}
    assert latin_pairs == {("hr", "эйчар", 2), ("devops", "девопс", 1)}, latin_pairs


def _check_hypothesis_from_json():
    """--hyp читает и .txt, и JSON схемы transcript/1 — с одинаковым результатом."""
    transcript = {
        "schema": TRANSCRIPT_SCHEMA,
        "source": "синтетика.wav",
        "text": "один два три четыре",
        "segments": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", "[00:00] Лектор: один два три четыре")
        hyp_json = _fixture(tmp, "hyp.json", json.dumps(transcript, ensure_ascii=False))
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp_json)])
        assert code == 0, (code, out)
        assert f"(JSON {TRANSCRIPT_SCHEMA})" in out, out
        assert "WER:            0.00 %" in out, out
        # Чужая схема и битый JSON — вход непригоден, код 2.
        alien = _fixture(tmp, "alien.json", json.dumps({"schema": "chunk-plan/1"}))
        code, _ = _run_cli(["--ref", str(ref), "--hyp", str(alien)])
        assert code == 2, code
        broken = _fixture(tmp, "broken.json", "{не json")
        code, _ = _run_cli(["--ref", str(ref), "--hyp", str(broken)])
        assert code == 2, code


def _check_machine_report():
    """--json пишет объект схемы score/1 со всеми обязательными полями."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", "[00:00] Лектор: HR сказал один два три")
        hyp = _fixture(tmp, "hyp.txt", "эйчар сказал один два")
        target = Path(tmp) / "score.json"
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp), "--json", str(target)])
        assert code == 0, (code, out)
        report = json.loads(target.read_text(encoding="utf-8"))
        assert report["schema"] == SCHEMA, report["schema"]
        for key in (
            "ref_words",
            "wer",
            "wer_masked",
            "cer",
            "substitutions",
            "deletions",
            "insertions",
            "pairs",
            "latin_pairs",
        ):
            assert key in report, key
        assert report["ref_words"] == 5, report["ref_words"]
        assert report["deletions"] == 1, report["deletions"]
        assert report["pairs"][0] == {
            "ref": "hr",
            "hyp": "эйчар",
            "count": 1,
            "latin": True,
        }, report["pairs"]
        assert set(report["latin_pairs"]) == {"count", "share_of_errors"}, report["latin_pairs"]
        assert report["latin_pairs"]["count"] == 1, report["latin_pairs"]
        assert abs(report["latin_pairs"]["share_of_errors"] - 0.5) < 1e-9, report["latin_pairs"]
        # Порядок пар — по убыванию count.
        counts = [pair["count"] for pair in report["pairs"]]
        assert counts == sorted(counts, reverse=True), counts

        # «-» отправляет тот же объект в stdout и файла не создаёт.
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp), "--json", "-"])
        assert code == 0, (code, out)
        assert f'"schema": "{SCHEMA}"' in out, out


def _check_bad_arguments():
    """Ошибки в аргументах и отсутствующие файлы дают код 2, --help — код 0."""
    code, out = _run_cli(["--help"])
    assert code == 0, code
    for flag in ("--ref", "--hyp", "--json", "--top", "--selftest", "-h, --help"):
        assert flag in out, flag
    assert "Использование:" in out, out

    for argv in (
        [],
        ["--ref", "нет.txt"],
        ["--hyp", "нет.txt"],
        ["--ref", "нет.txt", "--hyp", "нет.txt"],
        ["--ref"],
        ["--ref", "a", "--hyp", "b", "--top", "ноль"],
        ["--ref", "a", "--hyp", "b", "--top", "0"],
        ["--чепуха"],
    ):
        code, _ = _run_cli(argv)
        assert code == 2, (argv, code)


def _check_default_top():
    """По умолчанию печатается не больше пятнадцати пар."""
    assert DEFAULT_TOP == 15
    # Различать слова буквами, а не цифрами: цифры нормализация развернёт в
    # слова, и двадцати разных пар не получится.
    letters = "абвгдежзиклмнопрстуф"
    assert len(set(letters)) == 20
    words_ref = " ".join(f"слово{letter}" for letter in letters)
    words_hyp = " ".join(f"слава{letter}" for letter in letters)
    with tempfile.TemporaryDirectory() as tmp:
        ref = _fixture(tmp, "ref.txt", words_ref)
        hyp = _fixture(tmp, "hyp.txt", words_hyp)
        code, out = _run_cli(["--ref", str(ref), "--hyp", str(hyp)])
        assert code == 0, (code, out)
        assert "показано 15 из 20" in out, out
        # Строка-заголовок тоже содержит «→», поэтому считаем только пары.
        printed = [line for line in out.splitlines() if line.startswith("  ") and " → " in line]
        assert len(printed) == 15, printed


CHECKS = (
    ("контрольная фраза раздела 4: 9,09 / 0,00 / 5,38", _check_control_phrase),
    ("та же фраза через CLI, эталон не изменён", _check_control_phrase_cli),
    ("детектор числительных: слова единиц и отсутствие ложных", _check_masked_without_unit_words),
    ("разбор строк «[ММ:СС] Роль: текст»", _check_reference_lines),
    ("пометки в квадратных скобках удаляются и считаются", _check_notes_removed),
    ("пустой эталон — ошибка с кодом 1", _check_empty_reference),
    ("извлечение и сортировка пар замен, --top", _check_substitution_pairs),
    ("подсчёт пар «латиница → кириллица»", _check_latin_pairs),
    ("--hyp из JSON схемы transcript/1", _check_hypothesis_from_json),
    ("--json пишет итог схемы score/1", _check_machine_report),
    ("аргументы: --help, пропуски и мусор", _check_bad_arguments),
    ("--top по умолчанию 15", _check_default_top),
)


def selftest() -> int:
    failed = 0
    for name, check_func in CHECKS:
        try:
            check_func()
        except Exception as err:  # noqa: BLE001 — селф-тест обязан дойти до конца
            failed += 1
            print(f"ПРОВАЛ  {name}: {type(err).__name__}: {err}")
        else:
            print(f"ок      {name}")
    print()
    if failed:
        print(f"{PROG}: селф-тест ПРОВАЛЕН, провалено {failed} из {len(CHECKS)} проверок")
        return 1
    print(f"{PROG}: селф-тест пройден, {len(CHECKS)} из {len(CHECKS)} проверок")
    return 0


if __name__ == "__main__":
    sys.exit(main())
