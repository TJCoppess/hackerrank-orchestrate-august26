# Phase 1 Message Notification Router

This Python 3.10+ CLI processes WhatsApp messages sequentially through a small
LangGraph workflow backed by GPT-5.6 Sol. Phase 1 classifies message text and
carries image/audio paths as metadata; media inspection and historical retrieval
are intentionally deferred.

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

The default output is `dataset/output.csv`. Writes are atomic and idempotent by
`message_id`, so rerunning a message replaces its row rather than adding a duplicate.

## Test

Tests use only temporary output files and fake model responses:

```powershell
code/.venv/Scripts/python -m unittest discover -s code/tests -v
```

No test writes predictions to the real `dataset/output.csv`.
