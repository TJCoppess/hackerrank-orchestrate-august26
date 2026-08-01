# WhatsApp Message Notification Router

This Python 3.10+ CLI processes WhatsApp messages sequentially through a
LangGraph workflow backed by GPT-5.6 Sol. Phase 2 adds deterministic scam
analysis, user-scoped Pandas/RapidFuzz retrieval, GPT-5.6 Luna image extraction,
and `gpt-transcribe` voice transcription. Sol remains the sole classifier; media
models only extract evidence.

## Setup

From the repository root:

```powershell
python -m venv code/.venv
code/.venv/Scripts/python -m pip install -r code/requirements.txt
$env:OPENAI_API_KEY = "your-key"
```

On macOS or Linux, activate with `source code/.venv/bin/activate`, install with
`python -m pip install -r code/requirements.txt`, and export the key with
`export OPENAI_API_KEY="your-key"`.

Never commit API keys or `.env` files.

## Run

Smoke-test one text message:

```powershell
code/.venv/Scripts/python code/main.py --message-id msg_023
```

After activating the virtual environment on macOS or Linux, use
`python code/main.py --message-id msg_023` instead.

Process all unfinished rows:

```powershell
code/.venv/Scripts/python code/main.py
```

Useful options:

- `--limit N` processes at most `N` selected messages.
- `--force` replaces already completed predictions.
- `--dataset-dir PATH` changes the input directory.
- `--output PATH` changes the output CSV path.

Configuration is read from environment variables:

- `OPENAI_API_KEY` (required when processing is needed)
- `ORCHESTRATOR_MODEL` (default: `gpt-5.6-sol`)
- `REASONING_EFFORT` (default: `low`)
- `VISION_MODEL` (default: `gpt-5.6-luna`)
- `TRANSCRIBE_MODEL` (default: `gpt-transcribe`)

The default output is `dataset/output.csv`. Writes are atomic and idempotent by
`message_id`, so rerunning a message replaces its row rather than adding a duplicate.

## Test

Tests use only temporary output files and fake model responses:

```powershell
code/.venv/Scripts/python -m unittest discover -s code/tests -v
```

No test writes predictions to the real `dataset/output.csv`.

## Phase 2 workflow and failure behavior

For image and voice messages, the matching extractor must run before analysis.
Every message then receives a local scam scan; Sol may additionally retrieve up
to five prior messages belonging strictly to the active user. A final prediction
may cite only IDs returned by that retrieval call.

Media API failures are returned to Sol as structured evidence and cap final
confidence at `0.60`. Exhausted graph retries and system exceptions still write
a conservative `digest/unknown` row so the output remains complete. These are
marked as system failures in `RoutingResult.diagnostics`; a full CLI run exits
nonzero if any occur. The diagnostics object is serializable for the future
`evaluation/main.py` accuracy, F1, evidence, and failure-segmentation pipeline.
