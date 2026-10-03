# Обзор ASR-моделей под русскую лекцию

**Дата:** 2026-10-03
**Дата сверки источников: 2026-10-03.** Все ссылки открывались в этот день, и приведённые
по ним числа сверялись с первоисточником дословно. Одно исключение: страница тарифов
[Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text/pricing) отдаётся
обрезанной и прямой цитаты с неё получить не удалось — числа по Google взяты из
поисковой выдачи по домену `cloud.google.com` и перепроверены по двум формулировкам; сам
механизм скидки за логирование описан на отдельной странице документации
[Data logging](https://cloud.google.com/speech-to-text/docs/data-logging).

**Задача обзора.** Выбрать, чем распознавать русскую лекцию длиной 1,5–2,5 часа на
MacBook с Apple M5 и 16 ГБ унифицированной памяти при нулевом бюджете. Целевой
результат MVP — текст и субтитры с таймкодами (SRT/VTT), поэтому уровень таймкодов
(сегменты или слова) — такой же критерий отбора, как и WER.

**Исходная запись, под которую считаем.** `vidio_example/Лекция 1 01.09.26 (1).webm`:
02:35:29.651, Opus 48 кГц стерео, 20,6 % тишины — по замерам из
[`../analysis/example-recordings.md`](../analysis/example-recordings.md). Два с половиной
часа — это верхняя граница того, что должно обрабатываться за один прогон.

---

## Чему здесь можно верить, а чему нет

Этот документ собран по публичным источникам. **Ни одна модель в рамках задачи не
скачивалась и не запускалась** — это прямое ограничение постановки. Поэтому числа
делятся на три сорта, и сорт указан у каждого:

| Пометка | Что значит |
| --- | --- |
| ссылка на источник | число взято из первоисточника (README модели, карточка на Hugging Face, страница тарифов) |
| **расчёт** | число посчитано мной из параметров модели по явной формуле (она приведена рядом), первоисточник его не публикует |
| **требует проверки** | подтвердить источником не удалось либо источник не первичный (блог, SEO-страница); число оставлено как ориентир и должно быть перемерено |

**Отдельно про скорость.** Публичных замеров ASR на Apple **M5** на 2026-10-03 найти не
удалось: официальный бенчмарк-тред whisper.cpp содержит данные только по M1
([discussions/89](https://github.com/ggml-org/whisper.cpp/discussions/89)), а числа по
M3/M4 попадаются лишь в блогах без методики. Поэтому **вся колонка скорости — «требует
проверки»**: это ориентиры, а не измерения. Их и должен заменить прогон по протоколу из
[`benchmark-plan.md`](benchmark-plan.md).

Единственный найденный публичный замер с внятной методикой, который хоть как-то применим к
CPU, — таблица в README faster-whisper: 13 минут аудио моделью **small** на Intel Core
i7-12700K в 8 потоков, `openai-whisper` (fp32) 6 мин 58 с и 2335 МБ RAM против
faster-whisper (int8) 1 мин 42 с и 1477 МБ, то есть примерно 7,6×RT
([README](https://github.com/SYSTRAN/faster-whisper)). Переносить это число на нашу задачу
напрямую нельзя: там x86-CPU, а не Apple Silicon, и модель small, а не large — large
примерно в 6 раз больше по параметрам. Но порядок величины «быстрее реального времени даже
на CPU» он подтверждает, равно как и то, что int8-рантайм экономит и память, а не только
время.

**Отдельно про WER.** Сравнивать WER между строками таблицы напрямую нельзя: числа
получены на разных наборах и с разной нормализацией текста. Корректно сравнивать только
внутри одного набора и одной колонки. Самая полная сводка по русскому — таблица
[GigaAM/evaluation.md](https://github.com/salute-developers/GigaAM/blob/main/evaluation.md):
она сравнивает GigaAM всех версий, T-One и Whisper large-v3 на **шести** наборах (Golos
Farfield, Golos Crowd, Mozilla Common Voice 19, Russian LibriSpeech, OpenSTT Youtube,
Callcenter), и колонка «Average» в ней — среднее по этим шести. Но её считал вендор
GigaAM, то есть в пользу собственной модели; это учтено в выводах.

---

## Локальные варианты

Колонка «Скорость» — во сколько раз быстрее реального времени (×RT) обрабатывается
запись: 10×RT значит, что 2,5 часа лекции превратятся в текст за 15 минут.

| Вариант | Качество на русском (WER, %) | Веса, ГБ | RAM (оценка) | Скорость на M5/16 ГБ, ×RT | Таймкоды | Лицензия | Способ запуска |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **whisper.cpp**, large-v3-turbo (q5_0) | на публичных русских корпусах (Golos, MCV, RuLS) замеров именно turbo нет. Единственный найденный ориентир: 7,9 у large-v3-turbo на RTX 4090, там же `v3_e2e_rnnt` GigaAM даёт 3,3 на CPU ([Хабр, 5 фрагментов 6–30 с, TTS-озвучка аудиокниг, jiwer, 3 повтора](https://habr.com/ru/articles/1002260/)) — выборка крошечная и синтетическая (автор это признаёт сам), поэтому число **требует проверки**; ценно оно двумя вещами: замер независим от вендора GigaAM и воспроизводит тот же порядок разрыва, и мерялся в нём ровно тот вариант GigaAM, который стоит в нашем шортлисте | ≈0,55 (**расчёт**: 809 М параметров × 5 бит / 8; **требует проверки**) | ≈1,0–1,5 (**расчёт** от официальной таблицы: large = 2,9 ГиБ диска → ~3,9 ГБ памяти, turbo вдвое меньше) | 8–15 (**требует проверки**: M3 ~9,1×, M4 Pro 14–18× по блогам без методики) | сегменты; пословные через `-ml 1` (`--max-len`, по умолчанию 0) и `-dtw`/`--dtw MODEL` — «compute token-level timestamps» ([cli.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/cli/cli.cpp)) | MIT ([README](https://github.com/ggml-org/whisper.cpp)) | один бинарник `whisper-cli`, Metal и Core ML включены ([README](https://github.com/ggml-org/whisper.cpp)) |
| **whisper.cpp**, large-v3 (fp16) | 21,0 в среднем по 6 русским наборам; Golos Crowd 19,0, Golos Farfield 16,4, MCV19 5,5, RuLS 9,4, OpenSTT YouTube 17,8, Callcenter 23,1 ([GigaAM/evaluation.md](https://github.com/salute-developers/GigaAM/blob/main/evaluation.md), замер вендора GigaAM) | 2,9 ГиБ ([официальная таблица](https://github.com/ggml-org/whisper.cpp)) | ~3,9 ([официальная таблица](https://github.com/ggml-org/whisper.cpp)) | 2–4 (**требует проверки**) | то же | MIT | то же |
| **faster-whisper** (CTranslate2), large-v3 int8 | как у large-v3 выше (та же модель, иной рантайм); влияние int8 на русский WER **требует проверки** | 3,1 скачивается (fp16 с HF), в память при `compute_type="int8"` ≈1,6 (**расчёт**: 1550 М × 1 байт) | ≈2–3 (**расчёт**) | 3–6 (**требует проверки**; на Apple Silicon только CPU — Metal в CTranslate2 нет) | пословные, `word_timestamps=True` ([README](https://github.com/SYSTRAN/faster-whisper)) | MIT ([LICENSE](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)) | `pip install faster-whisper`, веса тянутся с HF автоматически ([README](https://github.com/SYSTRAN/faster-whisper)) |
| **WhisperX** (faster-whisper + выравнивание) | WER распознавания = faster-whisper; WhisperX улучшает не текст, а точность границ слов ([README](https://github.com/m-bain/whisperX)) | веса Whisper + ≈1,2 на `wav2vec2-large-xlsr-53-russian` (**расчёт**: 315 М × 4 байта; **требует проверки**) | ≈3,5–4,5 (**расчёт**: Whisper + модель выравнивания одновременно) | ниже faster-whisper: добавляется проход выравнивания (**требует проверки**) | **пословные, форсированным выравниванием** — для `ru` берётся `jonatasgrosman/wav2vec2-large-xlsr-53-russian` ([alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)) | BSD-2-Clause ([LICENSE](https://github.com/m-bain/whisperX/blob/main/LICENSE)) | `pip install whisperx`; на macOS только `--compute_type int8 --device cpu` ([README](https://github.com/m-bain/whisperX)) |
| **mlx-whisper**, large-v3-turbo | = Whisper turbo; отдельных замеров на русском нет, **требует проверки** | ≈1,6 fp16 / ≈0,45 в 4 бита (**расчёт** от 809 М параметров turbo по [официальной таблице openai/whisper](https://github.com/openai/whisper)) | ≈2–3 fp16 (**расчёт**) | нативный MLX на GPU Apple; публичных чисел в репозитории нет — **требует проверки** | пословные, `word_timestamps=True` ([PyPI](https://pypi.org/project/mlx-whisper/)) | **MIT** ([PyPI mlx-whisper](https://pypi.org/project/mlx-whisper/), версия 0.4.3 от 2025-08-29) | `pip install mlx-whisper`, веса с HF (`mlx-community/*`) ([PyPI](https://pypi.org/project/mlx-whisper/)) |
| **GigaAM v2** (RNNT / CTC) | **10,6 / 11,1** в среднем по 6 наборам; Golos Crowd 2,3 / 2,5, Golos Farfield 4,0 / 4,3, MCV19 0,9 / 1,5, RuLS 5,2 / 5,2 ([evaluation.md](https://github.com/salute-developers/GigaAM/blob/main/evaluation.md)) | ≈0,96 (**расчёт**: 0,2 млрд параметров × 4 байта fp32 — параметры и dtype по [карточке модели](https://huggingface.co/waveletdeboshir/gigaam-rnnt); **требует проверки**) | ≈2–3 (**расчёт**), +pyannote VAD для длинных записей | **требует проверки** | пословные, `word_timestamps=True` ([README_ru](https://github.com/salute-developers/GigaAM/blob/main/README_ru.md)) | MIT ([README](https://github.com/salute-developers/GigaAM)) | `pip install -e ".[torch]"` из репозитория; **`transcribe()` принимает не больше 25 с**, длинное — только `transcribe_longform()` ([README](https://github.com/salute-developers/GigaAM)) |
| **GigaAM v3** (RNNT / CTC) | **8,3 / 9,1** в среднем по 6 наборам — лучший результат на русском из найденных; Golos Crowd 2,4 / 2,8, Golos Farfield 3,9 / 4,5, MCV19 0,9 / 1,3, RuLS 4,4 / 4,7, OpenSTT YouTube 10,6 / 11,6, Callcenter 9,5 / 10,3 ([evaluation.md](https://github.com/salute-developers/GigaAM/blob/main/evaluation.md)) | **требует проверки** (энкодеры версий — 220–600 М параметров по [README_ru](https://github.com/salute-developers/GigaAM/blob/main/README_ru.md) → fp32 ≈0,9–2,4) | ≈3–5 (**расчёт**) | **требует проверки** | пословные | MIT | то же |
| **GigaAM v3 e2e** (RNNT / CTC) | 11,2 / 12,0 в среднем по 6 наборам — хуже «чистых» v3, **но выдаёт текст сразу с пунктуацией и заглавными**, и WER в таблице посчитан после их удаления. Проседает в основном на Golos Crowd (9,1 / 9,7 против 2,4 / 2,8 у v3) ([evaluation.md](https://github.com/salute-developers/GigaAM/blob/main/evaluation.md)) | **требует проверки** | ≈3–5 (**расчёт**) | **требует проверки** | пословные | MIT | `gigaam.load_model("v3_e2e_rnnt")` ([README](https://github.com/salute-developers/GigaAM)) |
| **Vosk**, `vosk-model-ru-0.42` | Golos Crowd **4,4**; OpenSTT audiobooks 11,1, YouTube 19,5, звонки 36,0 ([alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)) | **1,8** ([там же](https://alphacephei.com/vosk/models)) | ≈2,5–3,5 (**расчёт**: граф Kaldi грузится в память целиком; **требует проверки**) | потоковый распознаватель, быстрее реального времени; число **требует проверки** | **пословные со значением уверенности**, нативно | Apache 2.0 ([alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)) | `pip install vosk`, модель — zip-архив вручную |
| **Vosk**, `vosk-model-small-ru-0.22` | Golos Crowd 11,79; OpenSTT YouTube 31,97 ([alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)) | **0,045** ([там же](https://alphacephei.com/vosk/models)) | <0,5 (**расчёт**) | самый быстрый вариант обзора; число **требует проверки** | пословные | Apache 2.0 | то же |
| **NeMo**, `stt_ru_fastconformer_hybrid_large_pc` | без пунктуации (RNNT-декодер): Golos Crowd **1,9**, Golos Farfield 5,76, MCV12 test 5,3, RuLS test 11,05, Dusha 4,01. С учётом пунктуации и заглавных заметно хуже: MCV12 test 7,3, а **RuLS test 30,81** — то есть заявленная «бесплатная» пунктуация на книжной речи обходится втрое дороже по WER ([карточка модели](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc)) | ≈0,46 (**расчёт**: 115 М параметров × 4 байта fp32; **требует проверки**) | ≈2–3 (**расчёт**), плюс очень тяжёлая установка зависимостей | **требует проверки** (NeMo ориентирован на CUDA, качество поддержки MPS **требует проверки**) | в карточке модели не заявлены — **требует проверки** | модель CC-BY-4.0 ([карточка](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc)), тулкит Apache 2.0 | `pip install nemo_toolkit['all']` ([карточка](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc)) |

### Что из таблицы следует

**Хватает ли 16 ГБ.** Да, всем строкам таблицы с большим запасом: самый тяжёлый вариант —
whisper.cpp с large-v3 fp16, ~3,9 ГБ. Единственный способ упереться в память на этой
машине — запускать оригинальный `openai-whisper` на PyTorch: официальная таблица заявляет
для large ~10 ГБ, для turbo ~6 ГБ в колонке «Required VRAM»
([openai/whisper](https://github.com/openai/whisper)). На Apple Silicon отдельной
видеопамяти нет — те же 16 ГБ делятся между системой, CPU и GPU, поэтому 10 ГБ под одну
модель здесь уже рискованно. Поэтому
оригинальная реализация в шортлист не входит — у неё нет ни одного преимущества перед
whisper.cpp и faster-whisper, кроме референсности.

**Whisper на русском заметно слабее специализированных моделей.** 21,0 % среднего WER
(по 6 наборам) против 8,3 % у GigaAM v3 — разница в 2,5 раза. Даже с поправкой на то, что замер делал
вендор GigaAM, разрыв слишком велик, чтобы быть артефактом методики: он подтверждается и
независимо — у Vosk на Golos Crowd 4,4 % против 19,0 % у Whisper. Причина понятная:
Whisper многоязычный и на русский у него приходится малая доля обучающих данных, тогда
как GigaAM, Vosk и NeMo-ru обучались на русском целенаправленно.

**Но у специализированных моделей своя цена.** GigaAM не умеет длинные записи сам:
`transcribe()` ограничен 25 секундами, а `transcribe_longform()` тянет pyannote и требует
токен Hugging Face с принятием условий gated-модели
([README](https://github.com/salute-developers/GigaAM)). Vosk и «чистые» GigaAM v2/v3
отдают текст без пунктуации и заглавных — для расшифровки лекции это значит, что текст
придётся восстанавливать отдельным шагом. У NeMo этой проблемы нет (`_pc` = punctuation &
capitalization), но установка `nemo_toolkit['all']` непропорционально тяжёлая, а поддержка
Apple Silicon неясна.

**Пунктуация — скрытый критерий.** Цель MVP — читаемый текст и субтитры. Whisper выдаёт
пунктуацию и заглавные «из коробки», и это его главное практическое преимущество на этой
задаче. Из русских моделей так умеют NeMo `_pc` и GigaAM `v3_e2e_*` — но обе платят за это
качеством, и видно это только если смотреть на правильные колонки: у GigaAM `v3_e2e_rnnt`
средний WER 11,2 против 8,3 у `v3_rnnt`, у NeMo `_pc` на RuLS 30,81 против 11,05. То есть
«модель с пунктуацией» — это не бесплатная добавка, а обмен нескольких процентов точности
на готовность текста к чтению. Наш протокол замера этот обмен оценивает честно: WER
считается после снятия пунктуации у всех (см. нормализацию в
[`benchmark-plan.md`](benchmark-plan.md)), а наличие пунктуации учитывается отдельным
критерием при близких WER.

---

## Облачные сервисы с бесплатными квотами

**Это альтернатива, а не план по умолчанию.** Владелец выбрал локальное распознавание,
чтобы записи лекций не уходили наружу, и это решение остаётся в силе. Таблица ниже
существует потому, что владелец отдельно просил «изучить, какие есть варианты»: она нужна
как план Б, если локальные замеры покажут неприемлемое качество или скорость.

Главное, что стоит знать до любого обращения к облаку: **загрузка записи в сторонний
сервис — это разглашение содержимого лекции**, и бесплатные тарифы почти всегда дают
худшие гарантии по данным, чем платные. У [Gladia](https://www.gladia.io/pricing),
например, автоматический отказ от использования данных для обучения заявлен только на
старших тарифах — то есть на бесплатном его нет.

| Сервис | Что именно бесплатно | Лимиты | Нужна карта | Политика по данным |
| --- | --- | --- | --- | --- |
| [Deepgram](https://deepgram.com/pricing) | **200 $ стартового кредита** на pay-as-you-go, дальше оплата по мере использования | до 50 одновременных соединений на REST STT, 150 на WSS, 5 на Deepgram Whisper Cloud | **нет** — «No credit card required» на странице тарифов | заявлены SOC 2, HIPAA, GDPR, но **отдельной политики для бесплатного тарифа на странице нет** — **требует проверки** по Privacy Policy |
| [AssemblyAI](https://www.assemblyai.com/pricing) | **50 $ стартового кредита** — «$50 in free credits, no credit card required» | 5 новых потоков в минуту (против 100 на платном). Отдельная ловушка: «Default model selection may differ between free-tier and paid accounts» — модель по умолчанию на бесплатном тарифе другая, то есть замер на free-tier может не совпасть с платным | **нет** — «no credit card required» | на странице тарифов не описана — **требует проверки**. Поддержка русского в тарифах не указана и тоже **требует проверки** |
| [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text/pricing) | **60 минут аудио в месяц бессрочно** (не разовый триал), плюс разовые 300 $ кредитов новым аккаунтам GCP | 60 мин/мес — это ~4 % одной лекции на 2,5 часа. Сверх квоты: 0,016 $/мин **с логированием данных** против 0,024 $/мин без него | **да** — нужен биллинг-аккаунт GCP | **худшая из пятёрки по сути конструкции**: дешёвый тариф прямо оплачивается согласием на логирование записи, приватность здесь предмет доплаты в 1,5 раза |
| [Azure AI Speech](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/), тариф F0 | **5 часов аудио в месяц** (Speech to Text, общие на Standard и Custom) | только потоковая расшифровка: на странице прямо сказано **«Batch is not supported»** для F0, Fast Transcription тоже недоступна — то есть файл с лекцией этим тарифом не обработать без переписывания под стриминг | нужна подписка Azure; про карту на странице не сказано — **требует проверки** | на странице тарифов не описана — **требует проверки** |
| [Gladia](https://www.gladia.io/pricing) | **50 € кредитов**: «a one-time grant with no monthly reset. That's roughly 80+ hours of pre-recorded transcription» | 25 одновременных асинхронных запросов, 30 real-time; кредиты разовые, не обновляются | на странице прямо не сказано — **требует проверки** | GDPR / HIPAA / SOC 2 Type 2 на всех тарифах, но **«Automatic model training opt-out» появляется только с тарифа Growth**, а «Zero data retention» — только на Enterprise. То есть на бесплатных кредитах отказа от обучения на ваших данных нет |

**Вывод по облаку.** Из пяти вариантов для разовой расшифровки 2,5-часовой лекции реально
подходят Deepgram и Gladia: их кредитов хватает на десятки часов, и карта на входе не
нужна (у Deepgram и AssemblyAI это заявлено прямо). Google с 60 минутами в месяц и Azure F0
без пакетного режима на эту задачу не ложатся технически, а Google ещё и хуже всех по
конструкции приватности — у него скидка прямо привязана к разрешению логировать аудио.

Общая закономерность, которую стоит зафиксировать: **на бесплатных тарифах гарантии по
данным систематически слабее**, и это не случайность, а бизнес-модель. У Gladia отказ от
обучения на ваших данных начинается с платного Growth, у Google дешёвый тариф оплачивается
логированием, у Deepgram и AssemblyAI политика для free-tier на странице тарифов просто не
описана. Поэтому облако здесь — это не «то же самое, но в сети», а другой размен: вы
платите содержимым лекции вместо денег. Путь по умолчанию остаётся локальным.

---

## Длинные записи

Сама по себе ни одна модель обзора не рассчитана на непрерывные 2,5 часа: Whisper
обрабатывает окна по 30 секунд, GigaAM ограничен 25 секундами на вызов, Vosk потоковый.
Склейка окон — именно то место, где длинные записи ломаются, причём тремя разными
способами.

### 1. Нарезка по VAD, а не по таймеру

Резать запись на равные куски нельзя: граница почти наверняка попадёт в середину слова, и
на стыке появятся обрывки и повторы. Нарезать нужно по тишине — детектором речевой
активности (VAD).

У faster-whisper VAD встроен: интегрирован Silero VAD, по умолчанию выбрасываются паузы
длиннее 2 секунд, параметры настраиваются через `vad_parameters`
([README](https://github.com/SYSTRAN/faster-whisper)):

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    "data/samples/Лекция_1_01.09.26_1_t4514_300s_16k_mono.wav",
    language="ru",
    vad_filter=True,
    condition_on_previous_text=False,
    word_timestamps=True,
)
```

У whisper.cpp VAD подключается отдельной моделью, которую надо скачать заранее. В скрипте
`models/download-vad-model.sh` доступны две версии — `silero-v5.1.2` и `silero-v6.2.0`
([download-vad-model.sh](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-vad-model.sh)):

```bash
sh ./models/download-vad-model.sh silero-v6.2.0
./build/bin/whisper-cli -m models/ggml-large-v3-turbo-q5_0.bin \
  -f sample.wav -l ru --vad -vm models/ggml-silero-v6.2.0.bin
```

Флаги — `--vad` (включить) и `-vm` / `--vad-model` (путь к модели), оба по умолчанию
пустые ([cli.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/cli/cli.cpp)).

У GigaAM роль VAD играет pyannote внутри `transcribe_longform()`. Это не просто
зависимость: README требует сгенерировать токен Hugging Face **и** отдельно принять условия
доступа к `pyannote/segmentation-3.0`
([README](https://github.com/salute-developers/GigaAM)) — то есть без аккаунта на Hugging
Face длинные записи этим кандидатом не обработать вообще.

Запас по тишине в нашей записи для этого большой: 20,6 % длительности и 1103 интервала
тишины дольше секунды по замерам из
[`../analysis/example-recordings.md`](../analysis/example-recordings.md) — то есть
естественных мест для разреза более чем достаточно.

### 2. Галлюцинации и зацикливание

На длинной записи Whisper умеет уходить в повтор одной фразы или досочинять текст в
тишине. Механика известная: модель обусловливается на собственный предыдущий вывод и
начинает его продолжать вместо распознавания.

**Сначала главное, что экономит время: почти все пороги уже стоят в нужных значениях по
умолчанию.** Значения и формулировки ниже взяты из докстрингов
[openai/whisper `transcribe.py`](https://github.com/openai/whisper/blob/main/whisper/transcribe.py)
(faster-whisper повторяет их имена, whisper.cpp — одноимённые флаги). Колонка «по
умолчанию» здесь важнее колонки «значение»: менять руками нужно **ровно один параметр**.

| Параметр | По умолчанию | Ставить | Зачем |
| --- | --- | --- | --- |
| `condition_on_previous_text` | `True` | **`False`** | **единственный реальный рычаг.** Первоисточник прямо описывает последствия: «disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync» — то есть отключение лечит сразу и зацикливание, и разъезжающиеся таймкоды, а платим мы связностью текста на стыках окон |
| `compression_ratio_threshold` | `2.4` | оставить | «If the gzip compression ratio is above this value, treat as failed» — повторяющийся текст сжимается слишком хорошо, это и ловится |
| `logprob_threshold` | `-1.0` | оставить | «If the average log probability over sampled tokens is below this value, treat as failed» |
| `no_speech_threshold` | `0.6` | оставить | срабатывает **только вместе** с `logprob_threshold`: «if the no_speech probability is higher than this value AND the average log probability … is below `logprob_threshold`, consider the segment as silent». Поэтому крутить его в одиночку бессмысленно |
| `temperature` | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` | оставить | при срабатывании порогов декодирование повторяется с ростом температуры вместо того, чтобы отдать мусор |
| `vad_filter` (faster-whisper) | `False` | **`True`** | убирает тишину до распознавания — галлюцинации рождаются именно в ней |
| `-et` / `--entropy-thold` (whisper.cpp) | `2.40` | оставить | аналог порога повторов. 2.40 — уже значение по умолчанию ([cli.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/cli/cli.cpp)), поэтому передавать его явно бессмысленно; рычагом он становится только если его **снижать** |

Практический вывод: если кандидат на замере сыплет повторами, не надо перебирать все семь
параметров — надо выключить `condition_on_previous_text` и включить VAD, а остальное не
трогать, потому что оно уже включено.

Важная тонкость: при `vad_filter=True` пороги и VAD начинают дублировать друг друга. В
обсуждениях faster-whisper поднимался вопрос, не стоит ли отключать все три порога
(`compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`), когда VAD уже
включён: автор сообщает, что пороги выбрасывают сегменты с **корректной** расшифровкой
([Discussion #349](https://github.com/SYSTRAN/faster-whisper/discussions/349)).

Это нужно читать с двумя оговорками: это **вопрос пользователя, а не рекомендация
проекта**, и симптом описан на китайском и японском, не на русском. Но для нас важен сам
механизм: отброшенный верный сегмент — это удаления в WER, и на замере он будет выглядеть
как «модель проглатывает речь», а не как «пороги слишком строгие». Поэтому в протоколе и
печатается разбивка на замены/удаления/вставки — чтобы эти два случая не перепутать
([`benchmark-plan.md`](benchmark-plan.md), раздел 4).

У GigaAM и Vosk этой проблемы структурно нет: они не авторегрессионные языковые
декодеры, им нечего зацикливать. Это их недооценённое преимущество на длинных записях.

### 3. Монотонность таймкодов

После склейки окон таймкоды обязаны монотонно возрастать и не пересекаться — иначе SRT
будет невалидным, а плеер покажет субтитры рывками. Что проверять в собранном результате:

- время каждого сегмента пересчитано в абсолютное — к локальному времени внутри окна
  прибавлено смещение окна. Самая частая ошибка при ручной нарезке: VAD выбросил паузы, и
  если смещения считать по длине распознанных кусков, а не по исходным позициям, весь
  хвост расшифровки уезжает;
- `start` каждого следующего сегмента не меньше `end` предыдущего;
- `end > start` у каждого сегмента, пустых сегментов нет;
- последний `end` не превышает длительность записи из `ffprobe`.

Это стоит оформить как автоматическую проверку результата, а не как ручной просмотр: на
двух с половиной часах в SRT будет порядка тысячи сегментов.

**Диаризация** (кто говорит) — за рамками MVP. Если понадобится после, стандартный путь —
`pyannote.audio`, в WhisperX он уже встроен
([README](https://github.com/m-bain/whisperX)).

---

## Шортлист

Три кандидата: «быстрый», «точный» и страховка между ними. Все три — локальные,
бесплатные, с открытой лицензией и помещаются в 16 ГБ с запасом. Команды ниже **не
выполнялись** (постановка запрещает скачивать веса) — это инструкция на будущее.

### 1. «Быстрый»: whisper.cpp + large-v3-turbo

**Почему он.** Самый короткий путь к результату: один бинарник без Python-окружения,
Metal и Core ML поддерживаются как first-class
([README](https://github.com/ggml-org/whisper.cpp)), пунктуация и заглавные идут из
коробки, лицензия MIT и у кода, и у весов. Квантованные веса занимают около половины
гигабайта, памяти нужно порядка 1–1,5 ГБ — на 16 ГБ это даже не предмет обсуждения.
Качество на русском у Whisper заметно хуже специализированных моделей, и именно поэтому он
здесь в роли «быстрого», а не «точного».

```bash
brew install ffmpeg cmake
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
sh ./models/download-ggml-model.sh large-v3-turbo-q5_0
sh ./models/download-vad-model.sh silero-v6.2.0
```

Имена `large-v3-turbo-q5_0` и `silero-v6.2.0` взяты из списков в самих скриптах загрузки
([download-ggml-model.sh](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-ggml-model.sh),
[download-vad-model.sh](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-vad-model.sh)) —
это важно, потому что произвольное имя модели скрипт не примет.

**Размер скачиваемого:** веса ≈0,55 ГБ (**расчёт** от 809 М параметров turbo в q5_0,
**требует проверки** — официальная таблица whisper.cpp строки для turbo не содержит) плюс
VAD-модель (единицы МБ) и исходники репозитория. Для варианта без квантования — large-v3 на
2,9 ГиБ ([официальная таблица](https://github.com/ggml-org/whisper.cpp)).

Прогон:

```bash
./build/bin/whisper-cli -m models/ggml-large-v3-turbo-q5_0.bin \
  -f "data/samples/Лекция_1_01.09.26_1_t4514_300s_16k_mono.wav" \
  -l ru --output-srt --output-txt \
  --vad -vm models/ggml-silero-v6.2.0.bin
```

### 2. «Точный»: GigaAM v3 (e2e RNNT)

**Почему он.** Лучшее найденное качество на русском: 8,3 % среднего WER у `v3_rnnt` против
21,0 % у Whisper на тех же шести наборах
([evaluation.md](https://github.com/salute-developers/GigaAM/blob/main/evaluation.md)).
В шортлист берётся вариант `v3_e2e_rnnt`, а не `v3_rnnt`, и это осознанный размен, а не
мелочь: 11,2 против 8,3 — это на треть больше ошибок. Платим мы за то, что `e2e` выдаёт
текст сразу с пунктуацией и заглавными, а без них расшифровка лекции нечитаема и
непригодна для суммаризации, то есть восстанавливать их пришлось бы отдельной моделью —
ещё одна зависимость и ещё один источник ошибок. **Если замер покажет, что разрыв в WER
больше заявленного, правильное решение — вернуться к `v3_rnnt` и восстанавливать
пунктуацию постобработкой.** Лицензия MIT, пословные таймкоды есть
([README](https://github.com/salute-developers/GigaAM)).

**Чем придётся заплатить.** `transcribe()` принимает максимум 25 секунд («applicable for
audio only up to 25 seconds»), поэтому длинная запись идёт только через
`transcribe_longform()`. А он тянет pyannote и требует двух отдельных действий руками:
сгенерировать токен Hugging Face и **принять условия доступа к
`pyannote/segmentation-3.0`** — это gated-модель
([README](https://github.com/salute-developers/GigaAM)). Это единственный кандидат
шортлиста, которому нужна регистрация во внешнем сервисе — бесплатная, но обязательная, и
без неё двухчасовую лекцию он не обработает вообще. Для автоматизации это существенный
минус: пайплайн перестаёт быть self-contained.

```bash
brew install ffmpeg
git clone https://github.com/salute-developers/GigaAM.git
cd GigaAM
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[torch]"
pip install -e ".[longform]"
export HF_TOKEN="<токен Hugging Face>"
python -c "import gigaam; gigaam.load_model('v3_e2e_rnnt')"
```

Имена extras (`[torch]`, `[longform]`) и имя модели `v3_e2e_rnnt` сверены с
[README](https://github.com/salute-developers/GigaAM) 2026-10-03.

**Важно для оценки трудозатрат:** в `.venv` этого проекта **уже установлены** `gigaam`
0.2.0, `pyannote.audio` 4.0.7, `torch` 2.10.0 и `onnxruntime` 1.23.2 — то есть установка
для этого кандидата фактически уже сделана, не хватает только весов и токена Hugging Face.
Это стоит учитывать, сравнивая «стоимость входа» кандидатов: для GigaAM она ниже, чем
выглядит по списку команд. Веса в рамках этой задачи не скачивались.

**Размер скачиваемого:** веса — **требует проверки** (энкодеры версий GigaAM заявлены как
220–600 М параметров, что в fp32 даёт ≈0,9–2,4 ГБ; для v2 карточка модели указывает
0,2 млрд параметров в F32, то есть ≈0,96 ГБ). Плюс PyTorch и pyannote в окружение —
порядка 2,5–3 ГБ (**расчёт**, **требует проверки**).

### 3. Страховка: faster-whisper, large-v3 int8

**Почему он в списке.** Это тот же Whisper, что и в первом пункте, но с самым зрелым
пайплайном для длинных записей: встроенный Silero VAD, пословные таймкоды, все пороги
против галлюцинаций — штатные параметры, а не самодельные обвязки
([README](https://github.com/SYSTRAN/faster-whisper)). Если whisper.cpp окажется быстрым,
но будет сыпать повторами на стыках окон, а GigaAM упрётся в `HF_TOKEN` или в возню с
pyannote, faster-whisper закроет задачу без сюрпризов. На Apple Silicon у него есть
понятный минус: CTranslate2 не использует Metal, работает только на CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install faster-whisper
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
```

**Размер скачиваемого:** ≈3,1 ГБ — с Hugging Face тянутся веса в float16, а квантование в
int8 происходит при загрузке в память, поэтому на диске лежит полный размер, а в RAM
попадает ≈1,6 ГБ (**расчёт** от 1550 М параметров; **требует проверки**).

### Что в шортлист не вошло и почему

| Вариант | Причина |
| --- | --- |
| Vosk `ru-0.42` | хорошие числа на Golos (4,4 %), но нет пунктуации и заглавных, а на спонтанной речи заметно слабее: 19,5 % на OpenSTT YouTube ([alphacephei](https://alphacephei.com/vosk/models)). Остаётся хорошим вариантом для черновика и для машин слабее этой |
| NeMo `stt_ru_fastconformer_hybrid_large_pc` | числа без пунктуации отличные (1,9 на Golos Crowd), но с пунктуацией сильно хуже (RuLS 30,81), `nemo_toolkit['all']` — несоразмерно тяжёлая установка под одну модель, тулкит ориентирован на CUDA, работа на MPS **требует проверки**, а таймкоды в карточке модели не заявлены вообще. Кандидат «на потом», если понадобится ещё точность |
| mlx-whisper | самый интересный по идее (нативный MLX на GPU Apple, лицензия MIT — [PyPI](https://pypi.org/project/mlx-whisper/)), но публичных замеров скорости на Apple Silicon у проекта нет вообще, то есть главный его аргумент ничем не подтверждён. Стоит перемерить по протоколу и, возможно, заменить им первый пункт шортлиста |
| WhisperX | нужен, только если потребуются очень точные границы слов или диаризация; на macOS работает лишь в режиме `--compute_type int8 --device cpu` ([README](https://github.com/m-bain/whisperX)) |
| оригинальный `openai-whisper` | для large заявлено ~10 ГБ VRAM ([официальная таблица](https://github.com/openai/whisper)), что на 16 ГБ унифицированной памяти рискованно, и при этом медленнее обоих портов: по замерам faster-whisper на CPU он проигрывает в 2,5–4 раза ([README](https://github.com/SYSTRAN/faster-whisper)) |

---

## Что делать дальше

Таблицы выше дают ранжирование по публичным числам, но главные для нас колонки — скорость
на M5 и WER на нашей записи — не заполнены ничем, кроме оценок. Поэтому следующий шаг не
«выбрать по таблице», а замерить: протокол в
[`benchmark-plan.md`](benchmark-plan.md), эталонный фрагмент готовит
`scripts/make_sample.sh`.
