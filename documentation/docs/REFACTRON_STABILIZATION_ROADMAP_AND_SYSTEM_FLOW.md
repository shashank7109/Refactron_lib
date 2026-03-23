# Refactron Stabilization Roadmap and System Flow

This document explains how Refactron works end-to-end, where AI and backend APIs are used, how reports/refactors are generated, and what to do to stabilize the package for production-level developer usage.

## 1) What Refactron Is

Refactron is a Python package and CLI that:

- analyzes Python files for quality, security, complexity, and performance issues,
- suggests and applies refactors,
- supports AI-assisted suggestions and documentation,
- generates analysis reports,
- learns from developer feedback to improve ranking of future suggestions.

Core package entrypoint is:

- Python API: `refactron.Refactron`
- CLI entrypoint: `refactron.cli:main` (installed as `refactron`)

---

## 2) High-Level Architecture



Main directories:

- `refactron/core/` -> orchestration, config, models, results, auth credentials, backups, metrics
- `refactron/analyzers/` -> static analysis modules
- `refactron/refactorers/` -> transformation proposal modules
- `refactron/llm/` -> LLM clients (backend proxy + Groq) and orchestration
- `refactron/rag/` -> code indexing + semantic retrieval (ChromaDB + embeddings)
- `refactron/patterns/` -> feedback-driven pattern learning and ranking
- `refactron/cli.py` -> command-line workflows

Important backend-facing integration points:

- `POST /oauth/device` and `POST /oauth/token` for login device flow
- `GET /api/auth/verify-key` for API key validation
- `POST /api/llm/generate` for backend-proxied LLM generation
- `GET /api/github/repositories` for connected repository listing

Frontend touchpoint:

- Login URL shown/opened by CLI: `https://app.refactron.dev/login?code=<user_code>`

---

## 3) Local Install and First-Run (Windows-focused)

From repo root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
refactron --version
```

Optional automated setup:

```powershell
.\setup_dev.bat
```

Basic local usage:

```powershell
refactron init
refactron analyze .
refactron refactor . --preview
refactron report . -o report.txt
```

Authentication (needed for cloud-backed features):

```powershell
refactron login
refactron auth status
```

Credentials are currently stored in:

- `~/.refactron/credentials.json`

Note: this is file-based storage with restrictive permissions, not OS keychain storage.

---

## 4) End-to-End Runtime Flows

## 4.1 Analyze Flow

Command path:

1. User runs `refactron analyze <target>`
2. CLI loads config (`RefactronConfig`) and initializes `Refactron`
3. `Refactron.analyze()` discovers Python files
4. Optional optimizations run:
   - incremental filtering,
   - parallel file processing,
   - AST cache usage,
   - metrics and memory profiling hooks
5. For each file, `_analyze_file()`:
   - reads source,
   - computes base metrics (LOC/comments/blanks),
   - runs each enabled analyzer,
   - aggregates `CodeIssue` entries
6. Results aggregate into `AnalysisResult`
7. CLI prints summary, optionally detailed issues, and exits non-zero if critical issues exist

Default analyzers enabled:

- complexity
- code_smells
- security
- dependency
- dead_code
- type_hints
- performance

## 4.2 Report Generation Flow

Command path:

1. User runs `refactron report <target> --format <...> -o <file>`
2. CLI runs the same analysis engine (`Refactron.analyze()`)
3. CLI calls `AnalysisResult.report(detailed=True)` and writes content to output path

Current behavior note:

- `--format` is accepted by CLI and set in config, but `AnalysisResult.report()` currently generates text output only.
- So JSON/HTML report formats are not yet implemented in the core report renderer.

## 4.3 Refactor Flow

Command path:

1. User runs `refactron refactor <target> --preview` or `--apply`
2. CLI optionally creates a backup session before apply mode (`BackupRollbackSystem`)
3. `Refactron.refactor()` runs enabled refactorers on each file and returns `RefactorResult`
4. Pattern ranking (if enabled) may reorder operations based on learned patterns
5. CLI shows diff-like preview via `RefactorResult.show_diff()`
6. In apply mode, `RefactorResult.apply()` writes changes to files

Current apply implementation detail:

- apply is string-replacement based (`old_code` -> `new_code`, first match),
- best for non-overlapping and exact-match edits,
- may need a more robust AST or CST patching engine for higher reliability at scale.

## 4.4 AI Suggestion Flow (`suggest`)

Command path:

1. User runs `refactron suggest <file> [--line N]`
2. CLI tries to load RAG context from project `.rag` index (`ContextRetriever`)
3. CLI builds `LLMOrchestrator`, which chooses client:
   - `GroqClient` if `GROQ_API_KEY` exists and is valid
   - otherwise `BackendLLMClient` (Refactron backend proxy)
4. Orchestrator builds prompt (issue + code + retrieved context)
5. LLM returns a structured response
6. `SafetyGate` validates response and assigns safety score/status
7. CLI prints explanation + proposed code + confidence/safety info
8. If `--apply`, CLI creates backup and writes updated content

## 4.5 AI Documentation Flow (`document`)

Command path:

1. User runs `refactron document <file> [--apply]`
2. Similar retriever + orchestrator setup
3. Orchestrator generates markdown-style documentation output
4. In apply mode, CLI writes a new sibling file: `<original_stem>_doc.md`

## 4.6 RAG Flow

Index:

1. `refactron rag index` parses project Python files
2. chunks code into modules/classes/functions/methods
3. creates embeddings (SentenceTransformer)
4. stores vectors + metadata in ChromaDB under `<workspace>/.rag/chroma`

Retrieve:

1. `ContextRetriever.retrieve_similar(query)`
2. query embedding generated
3. nearest chunks returned with distance and metadata
4. optional rerank in CLI using LLM scoring

## 4.7 Pattern Learning Flow

1. Refactor operations are fingerprinted (`PatternFingerprinter`)
2. Feedback can be recorded (`accepted`/`rejected`/`ignored`)
3. `PatternStorage` persists feedback/patterns/profiles under `.refactron/patterns` (or user home fallback)
4. ranking can prioritize future operations based on learned patterns and project profile

---

## 5) Where Your Frontend/Backend APIs Fit

Your product model ("developer installs locally, authenticates using key/token from your platform") maps directly to current implementation:

- Frontend role:
  - user completes browser login and obtains/handles plan + API key UX
- Backend role:
  - device authorization endpoints for CLI login
  - API key verification endpoint
  - LLM proxy endpoint
  - repository APIs
- Local package role:
  - performs local analysis/refactor
  - calls backend only when using authenticated/cloud features

This split is good for startup product architecture because local value exists even if cloud features are unavailable.

---

## 6) Current Stabilization Risks (Priority View)

P0 (fix first):

1. Report format mismatch:
   - CLI exposes `text|json|html`, but renderer outputs text only.
2. Refactor apply robustness:
   - exact string replacement can fail on overlapping/moved code.
3. Documentation drift:
   - some docs describe behavior not fully matching implementation (example: secure keychain claim vs file-based credentials).

P1:

4. End-to-end contract tests for backend endpoints (`/oauth/*`, `/api/auth/verify-key`, `/api/llm/generate`).
5. AI command behavior consistency when RAG index is missing or models unavailable.
6. Better failure taxonomy and user-facing troubleshooting across CLI commands.

P2:

7. Performance benchmarks for large codebases under parallel + incremental modes.
8. Telemetry/metrics dashboards and SLO tracking for startup operations.
9. Hardening around Windows/macOS/Linux path and permission differences.

---

## 7) 90-Day Stabilization Roadmap

## Phase 1 (Weeks 1-2): Reliability Baseline

- lock and validate all command contracts (`analyze`, `refactor`, `report`, `suggest`, `document`, `rag`)
- add golden tests for CLI output and exit codes
- implement true report format backends:
  - text renderer
  - json renderer
  - html renderer
- add regression tests for report formats

Deliverables:

- `report --format json/html` works and is tested
- command behavior matrix documented in one place

## Phase 2 (Weeks 3-5): Safe Refactoring Engine

- replace or augment string-based apply with CST/AST patch application strategy
- detect operation overlap/conflict before apply
- improve rollback metadata and recovery messaging
- add high-confidence integration tests with real-world fixture repos

Deliverables:

- deterministic apply behavior with conflict handling
- rollback recovery validated by tests

## Phase 3 (Weeks 6-8): AI + RAG Production Hardening

- enforce clear provider fallback order (Groq vs backend proxy) with explicit logs
- standardize prompt/result schema validation
- add retry/backoff and typed error mapping for LLM/backend failures
- add RAG index health checks and stale-index warnings

Deliverables:

- stable AI command UX under normal failure conditions
- measurable AI quality gates (pass rate, safety reject rate)

## Phase 4 (Weeks 9-12): Productization and Ops

- align all docs with real behavior and deprecations
- add CI profile for "startup release gate":
  - unit + integration + CLI smoke tests
  - minimum coverage threshold
  - package install smoke test on Windows/Linux/macOS
- create release checklist and runbook
- define SLOs and monitoring for backend dependencies

Deliverables:

- repeatable release process
- clear operational readiness for customer onboarding

---

## 8) Recommended "Start Using Locally Now" Path

1. Install editable package and verify CLI.
2. Run `refactron init` in your project.
3. Run `refactron analyze . --detailed`.
4. Run `refactron refactor . --preview` and inspect suggestions.
5. Apply on a small subset first: `refactron refactor <file> --apply`.
6. Use backups/rollback for safety validation.
7. If using AI:
   - run `refactron rag index` in connected workspace,
   - run `refactron suggest <file> --line <n>`,
   - apply only after tests pass.

---

## 9) Suggested Engineering KPIs for Stabilization

- CLI command success rate (by command and platform)
- report generation correctness (format validation pass rate)
- refactor apply success rate (no manual repair needed)
- rollback success rate
- AI suggestion acceptance rate and safety reject rate
- median analyze runtime per KLOC
- backend dependency failure rate (auth + llm endpoints)

---

## 10) Immediate Action Plan for Your Team (Next 7 Days)

1. Implement JSON/HTML report renderers and tests.
2. Add integration tests for login and API key verification path.
3. Add refactor apply conflict detection tests.
4. Update docs to match credential storage and current feature reality.
5. Create a single "developer onboarding script" for Windows and Linux/macOS.
6. Run a real repo pilot (at least 10k LOC) and log all failures in a stabilization board.

If you want, the next step can be a follow-up implementation pass where we directly build:

- JSON/HTML report generation,
- a safer refactor apply engine,
- and a "production readiness checklist" command in CLI.
