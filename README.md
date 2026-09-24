# My LLM V1 — decoder-only Transformer с нуля

Небольшой, понятный фундамент для собственной языковой модели. Архитектура
реализована на PyTorch: decoder-only Transformer, causal self-attention, RMSNorm,
RoPE, SwiGLU, residual connections, связанные embeddings, KV-cache, обучение
следующего токена, evaluation и генерация.

## Что используется

- Все веса модели создаются случайно при `DecoderOnlyTransformer(...)`.
- Готовые веса Llama, Qwen, Mistral, Gemma, GPT, Phi, DeepSeek и других моделей
  не используются и не загружаются. В проекте нет `from_pretrained()`.
- `tokenizers` используется только для обучения byte-level BPE токенизатора на
  ваших текстах; это не модель и не набор готовых весов.
- Поддерживаются CUDA и CPU. На CUDA precision выбирается автоматически:
  BF16, если видеокарта его поддерживает, иначе FP16. На CPU используется FP32.

## Стартовая конфигурация

В `configs/model_v1.yaml` задано 42 082 816 параметров: словарь 32 000,
контекст 1 024, 8 слоёв, hidden size 512, 8 голов, SwiGLU intermediate size
1 408. Архитектурные и тренировочные параметры настраиваются в YAML. После
обучения токенизатора скрипт автоматически подгоняет `model.vocab_size` под
фактический размер его словаря.

Это не обученная языковая модель. До обучения генерация будет выдавать случайный
бессмысленный текст.

## Установка

```bash
git clone https://github.com/<ваш-логин>/my-llm.git
cd my-llm
python -m venv .venv
```

Активируйте окружение в PowerShell или Linux/macOS:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

Затем установите зависимости:

```bash
python -m pip install -r requirements.txt
```

## Проверка

```bash
python scripts/sanity_check.py
pytest -q
```

Sanity check делает один optimizer step только на случайных токенах и маленькой
временной модели, сохраняет checkpoint во временную папку и сравнивает выходы
после загрузки. Он не обучает V1 и не скачивает данные.

## Подготовка собственных данных

Положите `.txt` и/или `.jsonl` файлы в `data/raw/`. Для JSONL подходят строки
вида `{"text":"Ваш текст"}` или JSON-строки. Пустые записи пропускаются,
дубликаты удаляются, текст нормализуется в Unicode NFC. Разбиение на train и
validation происходит до токенизации. Если вход — один большой текстовый файл,
его непересекающиеся части делятся по границе слов.

Сначала обучите BPE токенизатор только на ваших данных:

```bash
python scripts/train_tokenizer.py --input data/raw
```

По умолчанию будут добавлены `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<|system|>`,
`<|user|>`, `<|assistant|>`. Если корпуса пока мало, фактический vocab может
получиться меньше 32 000; конфиг модели обновится до этого размера.

Затем создайте бинарные блоки для next-token prediction:

```bash
python scripts/prepare_dataset.py \
  --input data/raw \
  --tokenizer tokenizer/tokenizer.json \
  --output data/processed \
  --context-length 1024 \
  --validation-fraction 0.01
```

`data/processed/metadata.json` содержит число блоков, длину контекста и
количество удалённых точных дубликатов. Если корпус слишком мал для полного
блока из 1 025 токенов, уменьшите `context_length` в команде и в конфиге модели.

## Обучение

Настройте `training` и `model` в `configs/model_v1.yaml`, затем сами запустите:

```bash
python scripts/train_v1.py --config configs/model_v1.yaml
```

Обучение в этой подготовительной сессии не запускалось. Training loop поддерживает
AdamW, warmup + cosine scheduler, gradient accumulation, clipping, validation
loss/perplexity, логирование и периодические checkpoints. Данные, конфигурация и
чекпойнты не включаются в Git.

Продолжение с checkpoint:

```bash
python scripts/train_v1.py \
  --config configs/model_v1.yaml \
  --resume-from checkpoints/step_00500
```

Каждый checkpoint хранит `model.pt`, `optimizer.pt`, `scheduler.pt`,
`training_state.json`, а также конфигурацию модели и RNG state. Значение
`max_steps` в конфиге — общий номер целевого шага, а не число дополнительных
шагов после resume.

## Evaluation и генерация

```bash
python -m src.evaluate \
  --checkpoint checkpoints/step_00500 \
  --data data/processed

python scripts/chat.py \
  --checkpoint checkpoints/step_00500 \
  --tokenizer tokenizer/tokenizer.json
```

Однократная генерация:

```bash
python -m src.generate \
  --checkpoint checkpoints/step_00500 \
  --tokenizer tokenizer/tokenizer.json \
  --prompt "Привет" \
  --max-new-tokens 80 \
  --temperature 0.8 --top-k 50 --top-p 0.95 --seed 42
```

Численные результаты до обучения случайные и не имеют языкового смысла.

## Структура

```text
configs/       конфигурация модели и обучения
src/           модель, tokenizer/data API, train/evaluate/generation
scripts/       команды для пользователя и sanity check
tests/         проверки модели, attention, tokenizer, data и training step
data/raw/      ваши исходные .txt/.jsonl (не коммитятся)
data/processed/ подготовленные блоки (не коммитятся)
checkpoints/   локальные checkpoints (не коммитятся)
```

## Ограничения V1

Один GPU/CPU, один процесс, обычный `uint16` token-block формат (словарь должен
быть меньше 65 536 токенов), без distributed training. Для первого запуска
начните с небольшого контекста и маленького числа шагов, затем проверяйте
validation loss перед масштабированием.
