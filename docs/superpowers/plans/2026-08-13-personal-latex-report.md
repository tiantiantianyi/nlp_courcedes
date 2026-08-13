# 张添翼个人 LaTeX 报告 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份可独立提交、贡献归属准确、证据可追溯并已在本机实际编译验证的中文个人技术报告 PDF。

**Architecture:** 报告以 Git 提交和阶段实测文档为唯一事实源，先建立贡献与数值证据台账，再编写 BibTeX 和 `ctexart` 正文。编译环节使用 XeLaTeX 中文工具链，最后同时做日志、PDF 元信息、文本提取和页面视觉检查，避免把队友基线或未完成实验写成本人结果。

**Tech Stack:** LaTeX `ctexart`、XeLaTeX、latexmk、BibTeX、Poppler (`pdfinfo`/`pdftotext`/`pdftoppm`)、Git。

## Global Constraints

- 报告作者固定为“张添翼”，学号固定为“U202315231”。
- 初始提交 `222f232` 的 M3--M5 多模态检索流水线必须归属队友。
- 从 `4457459` 起的扩展、补全、联调、验证和修复可列为本人工作。
- M1 全量标注、人工 relevance、正式 A5--A7 质量实验和团队最终结果属于共同依赖或后续工作。
- 不使用 mock、image-only、20/64 图资源冒烟推导 Recall、MRR、mAP、nDCG 或质量提升。
- 报告目标长度为 5--7 页，不含可选参考文献页；若超过 7 页，先删减背景与代码细节，不删除贡献边界和结论限制。
- 使用 `ctexart + XeLaTeX`；PDF、`.tex`、`.bib` 提交，辅助文件不提交。
- 模型、原图、索引、实验原始 JSON 和 API Key 不进入 Git。

---

## File Map

- Create: `docs/personal_report/evidence.md` — 贡献归属、提交、实测数字和允许措辞的内部台账，不作为最终提交正文。
- Create: `docs/personal_report/references.bib` — Chinese-CLIP、jina-clip-v2、RRF、VIST、VLM reranking 等实际引用条目。
- Create: `docs/personal_report/zhang_tianyi_personal_report.tex` — 个人报告唯一 LaTeX 主文件。
- Create: `docs/personal_report/zhang_tianyi_personal_report.pdf` — 经本机编译和检查的最终 PDF。
- Modify: `.gitignore` — 忽略个人报告目录下的 LaTeX 辅助文件，不忽略 `.tex/.bib/.pdf`。
- Reference only: `docs/M4_QUERY_BACKENDS_2026-08-11.md`、`docs/M5_FUSION_COMPARISON_2026-08-11.md`、`docs/M6_LISTWISE_TOP20_2026-08-11.md`、`docs/M7_AUTO_STORY_UI_2026-08-11.md`、`docs/A7_JINA_CLIP_COMPARISON_2026-08-11.md`。

---

### Task 1: 建立贡献和实验数字证据台账

**Files:**
- Create: `docs/personal_report/evidence.md`
- Reference: Git commits `222f232..a8f3a05`
- Reference: `docs/M4_QUERY_BACKENDS_2026-08-11.md`
- Reference: `docs/M5_FUSION_COMPARISON_2026-08-11.md`
- Reference: `docs/M6_LISTWISE_TOP20_2026-08-11.md`
- Reference: `docs/M7_AUTO_STORY_UI_2026-08-11.md`
- Reference: `docs/A7_JINA_CLIP_COMPARISON_2026-08-11.md`

**Interfaces:**
- Consumes: Git commit hashes, file statistics, exact experiment values and conclusion-boundary statements.
- Produces: A four-column ledger `claim | ownership | evidence | allowed wording` used by Tasks 2--4.

- [ ] **Step 1: Create the evidence directory and empty ledger structure**

Create `docs/personal_report/evidence.md` with these exact headings:

```markdown
# 个人报告证据台账

## 队友已有基线
## 本人扩展提交
## 可引用的实测数字
## 共同依赖与禁止声明
## 提交到报告章节映射
```

- [ ] **Step 2: Record the teammate baseline**

Under “队友已有基线”, record:

```markdown
| 声明 | 归属 | 证据 | 报告允许措辞 |
|---|---|---|---|
| 原始 M3--M5 多模态检索流水线 | 队友 | `222f232` | “队友已经提交多路索引、检索与 RRF 基线；本人以此为开发起点。” |
```

- [ ] **Step 3: Record each personal extension commit**

Run:

```bash
git show --stat --oneline 4457459
git show --stat --oneline 9c5d54e
git show --stat --oneline 33551fe
git show --stat --oneline 6550040
git show --stat --oneline c4d329c
git show --stat --oneline 0c54bf8
git show --stat --oneline 223f3e0
git show --stat --oneline a8f3a05
```

Expected: each commit exists and its files match mock/image-only, evaluation/UI, directory pipeline, M5, M6, M7, M4 and A7 respectively.

Add one ledger row per commit. Use only these verbs: “扩展”“补全”“联调”“验证”“修复”。

- [ ] **Step 4: Record the exact numeric evidence**

Copy the following exact values into “可引用的实测数字”:

```text
M4: local Qwen 3/3; first cold 10.085 s; two warm mean 2.697 s.
M5: 20 images, 17 valid annotations, 12 queries, Top-8 overlap 92.71%, mean rank change 0.446, RRF 10.58 ms, weighted 6.62 ms.
M6: 3 queries, Top-20; pointwise 44.377 s/query and 4.039 GiB; listwise 9.143 s/query and 4.092 GiB; degraded 1/3.
A7 Chinese-CLIP: 64 images, 6.060 s build, 0.394 GiB peak, 2.251 ms warm query.
A7 jina-clip-v2: 64 images, 10.510 s build, 2.577 GiB peak, 47.701 ms warm query.
Regression at A7 commit: 122 passed.
```

For M4/M5/M6/A7 add the boundary “engineering/resource evidence only; not retrieval-quality evidence.”

- [ ] **Step 5: Scan the ledger for ownership violations**

Run:

```bash
rg -n '本人从零实现|本人完成全部|显著提升|优于|准确率提高' docs/personal_report/evidence.md
```

Expected: no matches. If a match exists, replace it with evidence-scoped wording before proceeding.

- [ ] **Step 6: Commit the evidence ledger**

```bash
git add docs/personal_report/evidence.md
git commit -m "docs: add personal report evidence ledger"
```

Expected: one documentation-only commit.

---

### Task 2: 编写可验证的参考文献数据库

**Files:**
- Create: `docs/personal_report/references.bib`
- Reference: `VLM_Final_Project_Technical_Proposal.md:556-589`

**Interfaces:**
- Consumes: citation keys used by the report.
- Produces: BibTeX keys `chineseclip2022`, `jinaclipv2`, `rrf2009`, `vist2016`, `ragvl`, `qwen3vl`, `faiss2017`, `gradio`.

- [ ] **Step 1: Define the citation key contract**

The report may cite only these keys:

```text
chineseclip2022
jinaclipv2
rrf2009
vist2016
ragvl
qwen3vl
faiss2017
gradio
```

- [ ] **Step 2: Write complete BibTeX entries**

Create `references.bib` with author, title, year and URL/venue for every key. Use the proposal’s cited paper URLs as the source; do not invent DOI values.

- [ ] **Step 3: Validate keys and duplicate entries**

Run:

```bash
rg -n '^@' docs/personal_report/references.bib
rg -o '^@[^{]+\{[^,]+' docs/personal_report/references.bib | sort
```

Expected: exactly eight unique entries.

- [ ] **Step 4: Commit the bibliography**

```bash
git add docs/personal_report/references.bib
git commit -m "docs: add personal report bibliography"
```

---

### Task 3: 编写个人报告 LaTeX 正文

**Files:**
- Create: `docs/personal_report/zhang_tianyi_personal_report.tex`
- Consume: `docs/personal_report/evidence.md`
- Consume: `docs/personal_report/references.bib`

**Interfaces:**
- Consumes: evidence ledger claims and the eight citation keys.
- Produces: a self-contained `ctexart` document that compiles with `latexmk -xelatex`.

- [ ] **Step 1: Write the preamble and title block**

Use this package contract:

```latex
\documentclass[UTF8,11pt,a4paper]{ctexart}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{booktabs,longtable,tabularx,array}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage[numbers,sort&compress]{natbib}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.25em}
\title{视觉语言相册检索项目个人技术总结}
\author{张添翼\\学号：U202315231}
\date{2026年8月}
```

- [ ] **Step 2: Write the abstract and ownership statement**

The abstract must contain all four ideas:

```text
队友先提交 M3--M5 基线；本人从该基线继续扩展；本机为 RTX 4060 Laptop 8GB；
当前报告只给工程/资源证据，正式质量结论等待结构化标注和人工 relevance。
```

Immediately after the abstract, add a boxed ownership note stating that `222f232` belongs to the teammate baseline.

- [ ] **Step 3: Write Sections 1--2**

Section 1 “项目背景与个人任务边界” must include a table with columns:

```text
模块 | 队友已有/共同起点 | 本人扩展 | 当前结论边界
```

Section 2 “代码基线分析” must be no longer than 500 Chinese characters and cite the initial commit as repository evidence, not as the author’s implementation.

- [ ] **Step 4: Write Section 3 personal technical work**

Use six subsections in this exact order:

```latex
\subsection{无标注开发模式与任意目录流水线}
\subsection{M5：归一化加权融合对照}
\subsection{M6：Top-20 Listwise 视觉重排}
\subsection{M7：自动故事与缺图补全}
\subsection{M4：三后端查询理解}
\subsection{A7：jina-clip-v2 适配与稳定性修复}
```

Each subsection must contain “问题”“本人工作”“验证”“限制” as bold inline labels.

- [ ] **Step 5: Write Section 4 local experiment tables**

Add one M6 table and one A7 table using the exact values from Task 1. Add a paragraph for M4/M5/M7. Every table caption must include sample scope such as “3 queries” or “64 images”.

- [ ] **Step 6: Write Sections 5--7**

Section 5 must explain Qwen schema drift, listwise missing IDs, Jina RoPE NaN and serialized 8GB model loading.

Section 6 must list the future order:

```text
M1 delivery validation -> 200-image integration -> full indexing -> human qrels validation -> A5--A7 formal evaluation -> report freeze
```

Section 7 must end with the audit table:

```text
工作项 | 队友已有 | 本人新增 | 共同依赖/后续
```

- [ ] **Step 7: Add bibliography and appendix evidence mapping**

End the file with:

```latex
\bibliographystyle{plainnat}
\bibliography{references}
\appendix
\section{提交与证据映射}
```

List all relevant commit hashes in a compact longtable; do not print local absolute paths.

- [ ] **Step 8: Perform a source-level ownership scan**

Run:

```bash
rg -n '从零实现|完成全部M3|完成全部 M3|正式质量提升|显著优于|准确率提高' \
  docs/personal_report/zhang_tianyi_personal_report.tex
```

Expected: no matches.

- [ ] **Step 9: Check citation consistency before compilation**

Run:

```bash
rg -o '\\cite[a-zA-Z]*\{[^}]+\}' docs/personal_report/zhang_tianyi_personal_report.tex
rg -o '^@[^{]+\{[^,]+' docs/personal_report/references.bib
```

Expected: every cited key exists in `references.bib`.

- [ ] **Step 10: Commit the LaTeX source**

```bash
git add docs/personal_report/zhang_tianyi_personal_report.tex
git commit -m "docs: draft Zhang Tianyi personal report"
```

---

### Task 4: 安装或配置中文 LaTeX 环境

**Files:**
- Modify: none in repository
- System packages: `texlive-xetex`, `texlive-lang-chinese`, `texlive-latex-extra`, `latexmk`

**Interfaces:**
- Consumes: a Linux host with package manager access.
- Produces: commands `xelatex`, `bibtex`, `latexmk` available on `PATH`.

- [ ] **Step 1: Detect existing tools**

Run:

```bash
command -v xelatex
command -v bibtex
command -v latexmk
```

Expected now: missing `xelatex` and `latexmk`. If all three exist, skip Steps 2--3.

- [ ] **Step 2: Install the minimum Chinese XeLaTeX toolchain**

On Ubuntu/Debian run:

```bash
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-lang-chinese texlive-latex-extra latexmk
```

Expected: package installation succeeds without changing repository files.

- [ ] **Step 3: Verify Chinese font and package availability**

Run:

```bash
kpsewhich ctexart.cls
kpsewhich xeCJK.sty
kpsewhich booktabs.sty
latexmk -v
```

Expected: each `kpsewhich` prints a path and latexmk prints a version.

---

### Task 5: 编译报告并修复所有 LaTeX 错误

**Files:**
- Modify: `docs/personal_report/zhang_tianyi_personal_report.tex`
- Modify: `docs/personal_report/references.bib`
- Create: `docs/personal_report/zhang_tianyi_personal_report.pdf`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: XeLaTeX toolchain and Task 3 source.
- Produces: final PDF with resolved citations and no fatal layout errors.

- [ ] **Step 1: Add auxiliary-file ignore rules**

Append scoped patterns:

```gitignore
docs/personal_report/*.aux
docs/personal_report/*.bbl
docs/personal_report/*.blg
docs/personal_report/*.fdb_latexmk
docs/personal_report/*.fls
docs/personal_report/*.log
docs/personal_report/*.out
docs/personal_report/*.toc
docs/personal_report/*.xdv
```

- [ ] **Step 2: Run the first full build**

From `docs/personal_report` run:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error zhang_tianyi_personal_report.tex
```

Expected: first successful build creates `.pdf`; if it fails, use the first LaTeX error in `.log`, fix only that cause and rerun.

- [ ] **Step 3: Check fatal warnings and unresolved references**

Run:

```bash
rg -n 'Undefined control sequence|LaTeX Error|Citation.*undefined|Reference.*undefined|There were undefined' \
  docs/personal_report/zhang_tianyi_personal_report.log
```

Expected: no matches.

- [ ] **Step 4: Check overfull boxes**

Run:

```bash
rg -n 'Overfull \\hbox|Overfull \\vbox' docs/personal_report/zhang_tianyi_personal_report.log
```

Expected: no matches wider than 5 pt. Fix long URLs with `\url{}`, long code with `\path{}`, and wide tables with `tabularx`; do not shrink all text globally.

- [ ] **Step 5: Verify page count and metadata**

Run:

```bash
pdfinfo docs/personal_report/zhang_tianyi_personal_report.pdf | rg 'Pages|Page size|File size'
```

Expected: A4 and 5--7 pages. If above 7, shorten background and appendix prose. If below 5, expand technical diagnosis/reflection using existing evidence, not invented results.

- [ ] **Step 6: Extract and audit PDF text**

Run:

```bash
pdftotext docs/personal_report/zhang_tianyi_personal_report.pdf /tmp/zhang-tianyi-report.txt
rg -n '张添翼|U202315231|队友|共同依赖|relevance|122 passed' /tmp/zhang-tianyi-report.txt
rg -n 'TBD|TODO|PLACEHOLDER|undefined|从零实现|显著优于' /tmp/zhang-tianyi-report.txt
```

Expected: first command finds all required identity/boundary terms; second finds no matches.

- [ ] **Step 7: Commit compiled source and PDF**

```bash
git add .gitignore docs/personal_report/zhang_tianyi_personal_report.tex \
  docs/personal_report/references.bib docs/personal_report/zhang_tianyi_personal_report.pdf
git commit -m "docs: compile Zhang Tianyi personal report"
```

Expected: no `.aux/.log/.out/.toc` files staged.

---

### Task 6: 最终视觉检查、仓库验证与交付

**Files:**
- Inspect: `docs/personal_report/zhang_tianyi_personal_report.pdf`
- Inspect: `docs/personal_report/zhang_tianyi_personal_report.tex`
- Inspect: repository status

**Interfaces:**
- Consumes: compiled PDF and clean report commit.
- Produces: verified final handoff with absolute clickable paths and exact build/test evidence.

- [ ] **Step 1: Render all PDF pages to images**

Run:

```bash
mkdir -p /tmp/zhang-tianyi-report-pages
pdftoppm -png -r 110 docs/personal_report/zhang_tianyi_personal_report.pdf \
  /tmp/zhang-tianyi-report-pages/page
```

Expected: one PNG per PDF page.

- [ ] **Step 2: Visually inspect every rendered page**

Check each page for:

```text
clipped tables, orphan headings, overlapping footer/page numbers,
missing Chinese glyphs, tiny text, excessive whitespace, broken URLs,
incorrect name/student ID, and ownership statements separated from their tables.
```

If any defect exists, edit `.tex`, rebuild with Task 5 Step 2, and repeat all Task 5 checks.

- [ ] **Step 3: Run final repository checks**

Run:

```bash
git diff --check
git status --short
git ls-files docs/personal_report
```

Expected: clean worktree; tracked files are `evidence.md`, `references.bib`, `.tex`, `.pdf`; no LaTeX auxiliary files.

- [ ] **Step 4: Confirm the final commit and PDF hash**

Run:

```bash
git log -3 --oneline
sha256sum docs/personal_report/zhang_tianyi_personal_report.pdf
```

Expected: report commits visible and a stable PDF SHA256 printed for handoff.

- [ ] **Step 5: Deliver the report**

Final response must state:

```text
PDF path, TeX path, page count, compiler used, contribution-boundary summary,
LaTeX installation result, validation commands, latest commit, and whether pushed.
```

Do not claim the report is complete before Tasks 5--6 have produced and inspected the PDF.
