# Deployment, Demo, and Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the repository-level deployment, Demo runbook, and LaTeX report deliverables without embedding private images/models or attributing teammates' M0--M5 work to Zhang Tianyi.

**Architecture:** Keep the existing Pixi/Conda Python application unchanged and add a thin container/deployment layer around `scripts/launch_app.py`. Bring the existing personal report from `report/zhang-tianyi` into `main`, update it only with auditable current evidence, and freeze team-report result tables only after formal candidate qrels exist.

**Tech Stack:** Docker 29, NVIDIA Container Toolkit, Conda/Pixi, Python 3.11, Gradio 6, pytest, XeLaTeX/ctex.

## Global Constraints

- Develop on `main`; the user explicitly approved direct development on `main`.
- Do not rewrite teammates' M0--M5 implementations or claim them as Zhang Tianyi's independent contribution.
- Do not add course images, model weights, FAISS indexes, generated images, or API keys to Git or Docker image layers.
- Container paths for models, data, and indexes are read-only volume mounts; generated outputs use a separate writable mount.
- Formal A5/A6 numbers are written only after `qrels_validation.json` reports `valid=true` and all 50 `graded_query_ids` are present.
- The personal report filename remains `张添翼_U202315231_个人报告.tex/.pdf`.
- Team-report body must remain at most 20 pages; appendices hold prompts and long schemas.
- Pytest commands unset host proxy variables and disable host pytest plugin auto-loading.

---

### Task 1: Add container and deployment contracts

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`
- Create: `docs/DEPLOY.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `pixi.toml`, `pixi.lock`, `configs/default.yaml`, `scripts/launch_app.py`.
- Produces: a GPU-capable image whose default process is `python scripts/launch_app.py --host 0.0.0.0 --port 7860`, plus documented host/Pixi/Conda/Docker launch paths.

- [ ] **Step 1: Record the configuration-only verification exception**

Docker and Markdown are configuration/documentation rather than Python behavior. Validate them with Docker's parser and Compose interpolation instead of source-grep unit tests.

- [ ] **Step 2: Create the minimal image contract**

Use a pinned Pixi base image, copy `pixi.toml`, `pixi.lock`, and `pyproject.toml` before source code for layer caching, run `pixi install --locked`, then copy `src/`, `scripts/`, `configs/`, `run.py`, and `README.md`. Do not copy ignored runtime directories.

The default entry point must expose Gradio on `0.0.0.0:7860`; `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` remain opt-in because first-time model installation is documented as a host preparation step.

- [ ] **Step 3: Define mounts in Compose**

Map these host paths explicitly:

```text
../Val                         -> /data/Val:ro
./artifacts/indexes            -> /app/artifacts/indexes:ro
./models                       -> /app/models:ro
./Qwen--Qwen3-VL-2B-Instruct  -> /app/Qwen--Qwen3-VL-2B-Instruct:ro
./stablediffusion              -> /app/stablediffusion:ro
./artifacts/generated          -> /app/artifacts/generated:rw
```

Reserve the NVIDIA GPU through `deploy.resources.reservations.devices` and publish `7860:7860`.

- [ ] **Step 4: Write deployment documentation**

Document prerequisites, model/index validation, local Conda launch, local Pixi launch, Docker build/run, Compose launch, Windows notes, 8GB settings, API-key handling, health checks, and recovery commands. Every command must use repository-relative paths and must state whether it writes artifacts.

- [ ] **Step 5: Verify configuration**

Run:

```bash
docker build --check .
docker compose config --quiet
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q
```

Expected: Docker checks exit 0 and the Python suite reports at least 248 passed.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml docs/DEPLOY.md README.md
git commit -m "docs: add reproducible GPU deployment"
```

### Task 2: Add a deterministic final Demo runbook

**Files:**
- Create: `configs/final_demo_queries.jsonl`
- Create: `docs/FINAL_DEMO_RUNBOOK.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the existing Gradio application, full Val indexes, M6/M7 service methods, and formal query IDs.
- Produces: one reproducible script for simple/compositional/negative/count/OCR retrieval, grounded QA refusal, 3--8 image story ordering, and AI-marked gap filling.

- [ ] **Step 1: Select five audited query examples**

Choose one reviewed query per category from `evaluation/formal_val_100/queries.jsonl`. Store `query_id`, `text`, `category`, and expected source image ID; do not store the image itself.

- [ ] **Step 2: Write the operator sequence**

The runbook must contain: environment check, Gradio start, five retrieval searches, candidate selection, grounded question and refusal example, story generation with and without gap filling, AI-label inspection, expected output fields, and a three-minute recording timeline.

- [ ] **Step 3: Run local smoke checks**

Run:

```bash
conda run -n vlm-course python run.py --input_dir ../Val --dry-run --launch
env -u ALL_PROXY -u all_proxy conda run -n vlm-course \
  python scripts/launch_app.py --config configs/default.yaml --split val --port 7860
```

Confirm `http://127.0.0.1:7860/` returns HTTP 200, then stop the temporary server.

- [ ] **Step 4: Commit**

```bash
git add configs/final_demo_queries.jsonl docs/FINAL_DEMO_RUNBOOK.md README.md
git commit -m "docs: add final M3-M7 demo runbook"
```

### Task 3: Bring the existing personal report into main

**Files:**
- Create from branch: `docs/personal_report/evidence.md`
- Create from branch: `docs/personal_report/references.bib`
- Create from branch: `docs/personal_report/张添翼_U202315231_个人报告.tex`
- Create from branch: `docs/personal_report/张添翼_U202315231_个人报告.pdf`

**Interfaces:**
- Consumes: commits `8f6de3d`, `4377b3d`, `2c236a2`, `c009f91` from `report/zhang-tianyi`.
- Produces: the existing six-page personal report and its evidence ledger on `main` without re-creating it from scratch.

- [ ] **Step 1: Confirm clean scoped status**

Run `git status --short` and confirm no uncommitted report paths exist.

- [ ] **Step 2: Cherry-pick the four report commits in order**

```bash
git cherry-pick 8f6de3d 4377b3d 2c236a2 c009f91
```

If `.gitignore` conflicts, preserve current runtime ignore rules and add only the report-specific PDF exception; do not replace the current file wholesale.

- [ ] **Step 3: Verify report identity and attribution**

Confirm the title page contains `张添翼` and `U202315231`, the teammate M3--M5 baseline is explicitly attributed, and all personal statements map to evidence commits.

- [ ] **Step 4: Verify the existing PDF**

Run:

```bash
pdfinfo docs/personal_report/张添翼_U202315231_个人报告.pdf
```

Expected: valid PDF metadata and six pages before the final evidence update.

### Task 4: Update the personal report with current auditable evidence

**Files:**
- Modify: `docs/personal_report/evidence.md`
- Modify: `docs/personal_report/张添翼_U202315231_个人报告.tex`
- Modify: `docs/personal_report/张添翼_U202315231_个人报告.pdf`

**Interfaces:**
- Consumes: formal 100-query merge report, 50-query candidate pool summary, A6 one-query smoke, current test output, and later frozen A5/A6 artifacts.
- Produces: a report that distinguishes engineering smoke results from final quality results and retains the personal contribution boundary.

- [ ] **Step 1: Replace obsolete stage claims**

Update: 122 tests to the fresh verified count; “full annotations pending” to the actual Qwen3.5 canonical/full-index status; 3-query M6-only evidence to include the new fixed-candidate quality smoke; and relevance status to 100 reviewed source queries plus 50 candidate queries awaiting/after explicit review.

- [ ] **Step 2: Add current contribution commits**

Record Tasks 1--7 of formal evaluation (`c64a958` through `6b2401a`) as Zhang Tianyi's integration/evaluation work. Do not describe Qwen3.5 annotation production or teammates' original M3--M5 implementation as personal work.

- [ ] **Step 3: Add formal results only after qrels freeze**

Read values from:

```text
artifacts/evaluation/formal/a5/a5_formal_results.json
artifacts/evaluation/formal/a6/formal_quality.json
artifacts/evaluation/formal/qrels_validation.json
```

If these files are absent, retain an explicit “candidate-level review pending” limitation and do not manufacture a table.

- [ ] **Step 4: Compile and verify**

Use XeLaTeX twice and BibTeX once in the report directory. Run `pdfinfo`; confirm the PDF opens, fonts are embedded, references resolve, and the filename still contains the student's name and ID.

- [ ] **Step 5: Commit**

```bash
git add docs/personal_report
git commit -m "docs: update Zhang Tianyi personal report evidence"
```

### Task 5: Create and freeze the team LaTeX report

**Files:**
- Create: `docs/team_report/AskAlbum_课程设计报告.tex`
- Create: `docs/team_report/references.bib`
- Create: `docs/team_report/Makefile`
- Create: `docs/team_report/AskAlbum_课程设计报告.pdf`

**Interfaces:**
- Consumes: `VLM_Final_Project_Technical_Proposal.md`, module evidence docs, formal A5/A6 artifacts, A7 resource comparison, Demo screenshots, and teammate-provided M0--M2 evidence.
- Produces: a ctex academic report with Abstract, Introduction, Related Work, Method, Experiment, Result, Discussion, Conclusion, contribution statement, and appendices.

- [ ] **Step 1: Draft stable sections before qrels freeze**

Write the system definition, M0--M7 method, schema/interfaces, environment, failure handling, and contribution split using only repository evidence. Keep M0--M5 ownership assigned to the responsible teammates.

- [ ] **Step 2: Import generated tables after qrels freeze**

Use `\input{}` for the frozen A5/A6 LaTeX tables and reproduce A7 resource values from its JSON artifact. Caption each table with sample count and label source.

- [ ] **Step 3: Write limitations and migration discussion**

State the actual A9 domains/sample counts, prohibit medical diagnosis claims, document single-reviewer/second-reviewer status, and distinguish source-positive metrics from 50-query graded nDCG.

- [ ] **Step 4: Compile and enforce page limit**

Run XeLaTeX/BibTeX and `pdfinfo`. Keep the main body at or below 20 pages by moving prompts, schemas, and long examples to appendices.

- [ ] **Step 5: Commit**

```bash
git add docs/team_report
git commit -m "docs: add final AskAlbum technical report"
```

### Task 6: Final post-qrels experiment and repository release

**Files:**
- Create: `docs/A5_FORMAL_RESULTS_2026-08-16.md`
- Create: `docs/A6_FORMAL_RESULTS_2026-08-16.md`
- Create: `docs/FORMAL_EVALUATION_HANDOFF_2026-08-16.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 50 completed candidate reviews and Tasks 1--5.
- Produces: final A5/A6 evidence, updated reports, clean Git status, and pushed `main`.

- [ ] **Step 1: Finalize qrels**

Run `scripts/finalize_formal_qrels.py`; require `valid=true`, 100 queries, 50 explicit `graded_query_ids`, and grade counts preserving zero rows.

- [ ] **Step 2: Run A5 and A6 formal experiments**

Run `scripts/run_ablation.py` on the finalized files, then `scripts/benchmark_listwise_top20.py --graded-only` with the same qrels validation. Record failures, degradation, latency, VRAM, MRR, and nDCG@10.

- [ ] **Step 3: Update both reports and evidence docs**

Every numeric claim must name its artifact, query count, date, and limitation. State the second-reviewer gap until another teammate actually completes it.

- [ ] **Step 4: Run release verification**

```bash
env -u ALL_PROXY -u all_proxy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n vlm-course python -m pytest -q
conda run -n vlm-course python -m compileall -q src scripts tests
conda run -n vlm-course python scripts/validate_manual_eval_set.py \
  --queries evaluation/formal_val_100/queries.jsonl \
  --relevance evaluation/formal_val_100/relevance.csv --expected-count 100
git diff --check
git status --short
```

- [ ] **Step 5: Push main**

```bash
git push origin main
```

Expected: local `main` and `origin/main` point to the same verified release commit.
