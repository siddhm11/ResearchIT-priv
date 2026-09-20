# Contributing to ResearchIT

ResearchIT is a FastAPI application with local SQLite state and optional cloud
backends for retrieval, metadata, and summaries. Small, focused pull requests
are easiest to review.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env.local
make dev
```

The app starts at <http://127.0.0.1:7860>. Most tests use fakes and temporary
databases, so they do not need cloud credentials.

## Before opening a pull request

Run the offline checks:

```bash
make check
```

Tests marked `live` call external services and are intentionally excluded from
the default command. Run `make test-live` only when the matching credentials
are configured.

Keep secrets in `.env.local`; every `.env*` file except `.env.example` is
ignored. Add configuration names and safe placeholder values to
`.env.example` when introducing a new setting.

Use a focused conventional commit such as `fix:`, `feat:`, `test:`, `docs:`,
or `chore:`. Describe the behavior change and include the validation command in
the pull request.
