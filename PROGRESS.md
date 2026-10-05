# Development and evaluation record

Updated October 4, 2026. This is a learning-project record, not a production
certification. Personal PDFs, screenshots, extracted text, credentials, and private
evaluation outputs are intentionally excluded from Git.

## Scope and completed work

- Shared RAG module for the CLI and FastAPI; local Ollama generation and embeddings,
  persistent Chroma index, MMR retrieval, document selection, and retrieved-page labels.
- Custom HTML/CSS/JavaScript frontend separated from backend code. Responsive styling,
  private-key connection, uploads, conversation clearing, and confirmed PDF deletion.
- API authentication before upload parsing, strong-key validation, filename/path checks,
  PDF signature and size checks, bounded conversation storage, and lazy integrations.
- Optional S3 backup through the standard credential chain. Local use needs no AWS keys.
  Backup failures are visible; cloud deletion failure preserves the local document.
- Docker configuration and test/build CI. Automatic cloud deployment is disabled.
- Separate scikit-learn/MLflow exercise: six model configurations and eight Naive Bayes
  alpha settings. Revised training code uses stratified train/validation/test splitting,
  validation-based selection, and one final held-out evaluation. Revised scores have
  not been rerun or claimed. These classifiers do not train the chatbot's language model.
- User-confirmed scope: selectable-text PDFs. OCR, scanned forms, checkmark interpretation,
  and image understanding are out of scope for now.

## Failures, improvements, and evidence

| Failure or limitation observed | Change or decision | Evidence / remaining caveat |
| --- | --- | --- |
| Earlier training category was unavailable; Windows output encoding failed; MLflow filesystem tracking was deprecated | Corrected dataset category, ASCII console output, SQLite tracking | Historical fixes; current training results still need a fresh run |
| Credentials and SSH material were exposed during development | Strong local key, ignored secret files, removal of local static AWS credentials, deployment disabled | User showed old IAM key inactive and removed SSH secret; no claim of complete account-wide security audit |
| Embedded/default frontend key was unsafe | Key entered privately and retained only in tab memory; HTTPS required outside localhost | Authentication tests; single-owner, not multi-user authorization |
| Receipt currency guessed from dollar symbol/payment method | Explicit grounding rules forbid unsupported currency inference | Browser check correctly abstained |
| Receipt quantity omitted or confused with product-name multiplier | Narrow single-item table extraction helper | Browser check returned the actual quantity; not a universal table parser |
| Different invoice/receipt identifiers called identical | Extract labeled identifiers and compare exact strings conservatively | Browser comparison passed |
| Broken Unicode characters damaged receipt identifiers | PDFium fallback when primary extraction is corrupted | Extraction tests and fresh upload evaluation passed |
| Restart left stale corrupted chunks in the existing index | Detect corrupted retrieved text and request re-upload | Re-upload required; no automatic index migration |
| Period and totals answers inherited unrelated item/quantity formatting | Removed global formatting example; restrict prior-question context to explicit references | Full API evaluation reproduced two failures before changes, then passed |
| False amount treated as missing information | Distinguish contradiction from absent evidence in prompt | Browser correctly rejected false amount |
| Upload returned generic 503 without useful terminal diagnostics | Log stage, exception class, and code locations without exception contents | Tests verify no sensitive exception text leaks; original intermittent cause was not established; later upload succeeded |
| Filename containing tilde rejected | User renamed file using allowed characters | CV upload succeeded with four indexed chunks; filename rules unchanged |
| Enter-to-send did not work in user's browser despite mocked tests | Enter and submit button share send function; cache-busted script and no-store UI responses; Shift+Enter preserves newline | User subsequently confirmed working; initial mock tests were insufficient browser evidence |
| Delete action hard to identify | Explicit Delete PDF label, confirmation, deleting state, error recovery | API success/failure tests; no claim of browser-tested deletion of personal data |
| Scanned survey values detached from labels; second page absent from extracted text | Identified extraction limitation; chose text-only scope instead of OCR/model tuning | First question had three wrong fields; second could not find visible totals. Unresolved for scanned documents |

## Manual browser evaluation

Receipt: 8/8 questions passed after re-upload: item/quantity, plan period, subtotal/tax/
total, payment date, identifier comparison, missing currency, missing refund policy,
and correction of a false amount.

Independent two-page CV: 5/5 questions passed without CV-specific code changes:
university and dates, programming languages, spoken languages/proficiency, project
technology stack, and refusal to invent an unstated salary.

These 13 checks cover two small text-based documents only. The receipt was used while
developing fixes and is not a held-out benchmark. The CV offers a small independent
check, not statistical proof of general accuracy. Retrieved pages are not claim-level
citations; listing both pages does not prove both support each answer.

## Automated verification

- October 4 Docker verification: fresh image built; isolated Linux container reached
  host Ollama, rejected unauthenticated requests, indexed a synthetic PDF, answered its
  known project-code question with the expected page citation, retained the index after
  container restart, and deleted the test document. Cloud backup was disabled. The
  disposable container was removed; user documents and existing containers were untouched.
  This tested restart persistence, not volume recovery after container replacement.
  Reproduce with `python scripts/smoke_docker.py PATH_TO_SYNTHETIC_PDF` after building
  `doc-bot:local-check`; fixture must contain the project code BLUEBIRD-742.
- Pre-push run on September 29: 52 Python tests passed, 5 optional live-model tests
  skipped; all 3 JavaScript tests passed.
- Prior real local-model run: 55 Python tests passed at that revision.
- Receipt upload/index/retrieval/chat evaluation: 8/8 in a shared conversation and
  8/8 in isolated sessions, using an ephemeral index and no cloud backup.
- Latest keyboard work: 3 JavaScript tests and 27 API tests passed. Keyboard tests
  mock browser elements; user's subsequent browser confirmation is separate evidence.
- Reproduce: `python -m pytest -q` and `node --test tests/frontend.test.cjs`.
  Set `RUN_OLLAMA_TESTS=1` for optional live-model tests. Private PDF evaluation uses
  `scripts/evaluate_receipt.py`; keep cases/results under ignored `tmp/`.

## Known limitations and next work

- Scanned/mixed-image PDFs may yield partial text. Empty-text rejection exists, but
  comprehensive partial-extraction detection/warnings are not implemented.
- No OCR, vision, arbitrary table reasoning, or clinical interpretation.
- Prompt/history heuristics and narrow deterministic helpers need broader evaluation.
- No atomic transaction across filesystem and Chroma; use one API worker.
- Dependencies not fully locked; Docker inference passed the local synthetic smoke test,
  but current cloud deployment remains unverified. Automatic AWS deployment is disabled.
- Roadmap gaps: fresh classifier evaluation, broader RAG benchmark, production HTTPS,
  rate limiting, backup recovery validation, Kubernetes, agents, external monitoring.
- Next: add more independent text-document tests, improve partial-text warnings within
  the chosen scope, and verify Docker locally before any cloud deployment.
