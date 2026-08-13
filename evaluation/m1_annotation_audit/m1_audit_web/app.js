"use strict";

const state = {
  reviewer: "",
  summary: null,
  currentTask: null,
  currentImageId: null,
  activeCandidate: "candidate_1",
  filter: "all",
  candidateDrafts: {},
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast show${error ? " error" : ""}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.className = "toast"; }, 2800);
}

async function api(path, options = {}) {
  const separator = path.includes("?") ? "&" : "?";
  const response = await fetch(`${path}${separator}reviewer=${encodeURIComponent(state.reviewer)}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function setView(loggedIn) {
  $("#loginView").classList.toggle("hidden", loggedIn);
  $("#appView").classList.toggle("hidden", !loggedIn);
}

async function login() {
  const reviewer = $("#reviewerInput").value.trim();
  if (!reviewer) return showToast("请输入评审者名称", true);
  state.reviewer = reviewer;
  localStorage.setItem("m1_audit_reviewer", reviewer);
  $("#reviewerLabel").textContent = reviewer;
  try {
    await loadSummary();
    setView(true);
    const firstPending = state.summary.tasks.find(task => task.phase !== "submitted") || state.summary.tasks[0];
    if (firstPending) await loadTask(firstPending.image_id);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function loadSummary() {
  state.summary = await api("/api/summary");
  const { submitted, gold_saved: drafts } = state.summary.counts;
  $("#progressText").textContent = `${submitted} / ${state.summary.sample_size} 已提交`;
  $("#draftText").textContent = `${drafts} 份进行中`;
  $("#progressBar").style.width = `${(submitted / state.summary.sample_size) * 100}%`;
  renderTaskList();
}

function renderTaskList() {
  const tasks = state.summary.tasks.filter(task => {
    if (state.filter === "pending") return task.phase !== "submitted";
    if (state.filter === "submitted") return task.phase === "submitted";
    return true;
  });
  $("#taskList").innerHTML = tasks.map(task => `
    <button class="task-button ${task.phase} ${task.image_id === state.currentImageId ? "active" : ""}" data-image-id="${escapeHtml(task.image_id)}">
      <span class="task-number">${String(task.sample_index).padStart(2, "0")}</span>
      <span class="task-id">图片 ${escapeHtml(task.image_id)}</span>
      <span class="task-dot" title="${task.phase}"></span>
    </button>
  `).join("");
  $$(".task-button", $("#taskList")).forEach(button => {
    button.addEventListener("click", () => loadTask(button.dataset.imageId));
  });
}

async function loadTask(imageId) {
  try {
    state.currentImageId = imageId;
    state.currentTask = await api(`/api/task/${encodeURIComponent(imageId)}`);
    state.activeCandidate = "candidate_1";
    state.candidateDrafts = structuredClone(state.currentTask.review?.candidate_reviews || {});
    $("#emptyState").classList.add("hidden");
    $("#auditImage").src = `${state.currentTask.image_url}?v=${encodeURIComponent(state.currentTask.image_id)}`;
    $("#imageTitle").textContent = `图片 ${state.currentTask.image_id}`;
    $("#sampleIndex").textContent = `SAMPLE ${state.currentTask.sample_index} / ${state.currentTask.sample_size}`;
    $("#imageStage").classList.remove("actual");
    if (state.currentTask.review?.gold) {
      fillGoldForm(state.currentTask.review.gold);
      showCandidatePhase();
    } else {
      resetGoldForm();
      showGoldPhase();
    }
    renderTaskList();
  } catch (error) {
    showToast(error.message, true);
  }
}

function showGoldPhase() {
  $("#goldPhase").classList.remove("hidden");
  $("#candidatePhase").classList.add("hidden");
}

function showCandidatePhase() {
  $("#goldPhase").classList.add("hidden");
  $("#candidatePhase").classList.remove("hidden");
  ensureCandidateDrafts();
  renderCandidateTabs();
  renderCandidateContent();
  updateSubmissionControls();
}

function updateSubmissionControls() {
  const submitted = state.currentTask.review?.phase === "submitted";
  $("#reopenReviewButton").classList.toggle("hidden", !submitted);
  $("#editableActions").classList.toggle("hidden", submitted);
  $("#backToGoldButton").classList.toggle("hidden", submitted);
  $("#candidateContent").querySelectorAll("input, select, textarea, button").forEach(control => {
    control.disabled = submitted;
  });
}

function resetGoldForm() {
  $("#goldForm").reset();
  $("#goldEntities").innerHTML = "";
  $("#goldOcr").innerHTML = "";
  updateGoldEmptyStates();
}

function fillGoldForm(gold) {
  resetGoldForm();
  const assessability = $(`input[name="assessability"][value="${gold.assessability}"]`);
  if (assessability) assessability.checked = true;
  $("#scenePrimary").value = gold.scene_primary;
  $("#environment").value = gold.environment;
  gold.salient_entities.forEach(item => addGoldEntityRow(item));
  gold.clear_ocr.forEach(item => addGoldOcrRow(item));
  $("#goldNotes").value = gold.notes || "";
  updateGoldEmptyStates();
}

function addGoldEntityRow(value = {}) {
  const index = $$(".entry-row", $("#goldEntities")).length + 1;
  const row = document.createElement("div");
  row.className = "entry-row gold-entity-row";
  row.innerHTML = `
    <span class="row-id">g${index}</span>
    <input class="entity-name" maxlength="80" placeholder="实体名称" value="${escapeHtml(value.name || "")}">
    <label class="count-toggle"><input class="count-evaluable" type="checkbox" ${value.count_evaluable ? "checked" : ""}> 可精确计数</label>
    <input class="count-input" type="number" min="1" max="999" placeholder="数量" value="${value.count ?? ""}" ${value.count_evaluable ? "" : "disabled"}>
    <button type="button" class="remove-entry" title="删除" aria-label="删除">×</button>`;
  $(".count-evaluable", row).addEventListener("change", event => {
    $(".count-input", row).disabled = !event.target.checked;
    if (!event.target.checked) $(".count-input", row).value = "";
  });
  $(".remove-entry", row).addEventListener("click", () => { row.remove(); renumberGoldRows(); });
  $("#goldEntities").append(row);
  updateGoldEmptyStates();
}

function addGoldOcrRow(value = {}) {
  const index = $$(".entry-row", $("#goldOcr")).length + 1;
  const row = document.createElement("div");
  row.className = "entry-row ocr-row gold-ocr-row";
  row.innerHTML = `
    <span class="row-id">o${index}</span>
    <input class="ocr-text" maxlength="300" placeholder="按原图转写文字" value="${escapeHtml(value.text || "")}">
    <button type="button" class="remove-entry" title="删除" aria-label="删除">×</button>`;
  $(".remove-entry", row).addEventListener("click", () => { row.remove(); renumberGoldRows(); });
  $("#goldOcr").append(row);
  updateGoldEmptyStates();
}

function renumberGoldRows() {
  $$(".gold-entity-row").forEach((row, index) => { $(".row-id", row).textContent = `g${index + 1}`; });
  $$(".gold-ocr-row").forEach((row, index) => { $(".row-id", row).textContent = `o${index + 1}`; });
  updateGoldEmptyStates();
}

function updateGoldEmptyStates() {
  $("#goldEntityEmpty").classList.toggle("hidden", $$(".gold-entity-row").length > 0);
  $("#goldOcrEmpty").classList.toggle("hidden", $$(".gold-ocr-row").length > 0);
}

function collectGold() {
  const assessability = $("input[name=" + '"assessability"' + "]:checked")?.value;
  if (!assessability) throw new Error("请选择图片是否足以判断");
  if (!$("#scenePrimary").value || !$("#environment").value) throw new Error("请选择场景主类和环境");
  const salientEntities = $$(".gold-entity-row").map((row, index) => {
    const name = $(".entity-name", row).value.trim();
    if (!name) throw new Error(`请填写第 ${index + 1} 个主要实体名称`);
    const countEvaluable = $(".count-evaluable", row).checked;
    const count = countEvaluable ? Number($(".count-input", row).value) : null;
    if (countEvaluable && (!Number.isInteger(count) || count < 1)) throw new Error(`请填写 ${name} 的精确数量`);
    return { gold_id: `g${index + 1}`, name, count_evaluable: countEvaluable, count };
  });
  const clearOcr = $$(".gold-ocr-row").map((row, index) => {
    const text = $(".ocr-text", row).value.trim();
    if (!text) throw new Error(`请填写第 ${index + 1} 条 OCR`);
    return { gold_id: `o${index + 1}`, text };
  });
  return {
    assessability,
    scene_primary: $("#scenePrimary").value,
    environment: $("#environment").value,
    salient_entities: salientEntities,
    clear_ocr: clearOcr,
    notes: $("#goldNotes").value.trim(),
  };
}

async function saveGold(event) {
  event.preventDefault();
  try {
    const body = await api(`/api/review/${encodeURIComponent(state.currentImageId)}/gold`, {
      method: "PUT", body: JSON.stringify(collectGold()),
    });
    state.currentTask.review = body.review;
    state.candidateDrafts = structuredClone(body.review.candidate_reviews || {});
    await loadSummary();
    showCandidatePhase();
    showToast("人工事实已保存");
  } catch (error) { showToast(error.message, true); }
}

function defaultCandidateReview(candidate) {
  const gold = state.currentTask.review.gold;
  return {
    entity_judgments: {},
    salient_coverage: Object.fromEntries(gold.salient_entities.map(item => [item.gold_id, false])),
    ocr_judgments: {},
    ocr_coverage: Object.fromEntries(gold.clear_ocr.map(item => [item.gold_id, false])),
    relation_judgments: {},
    caption_new_fact_count: 0,
    caption_new_fact_notes: "",
    caption_correctness: 3,
    caption_completeness: 3,
    privacy_violation: false,
    privacy_notes: "",
    notes: "",
  };
}

function ensureCandidateDrafts() {
  const gold = state.currentTask.review.gold;
  state.currentTask.candidates.forEach(candidate => {
    if (!state.candidateDrafts[candidate.slot]) state.candidateDrafts[candidate.slot] = defaultCandidateReview(candidate);
    const review = state.candidateDrafts[candidate.slot];
    review.salient_coverage = Object.fromEntries(
      gold.salient_entities.map(item => [item.gold_id, Boolean(review.salient_coverage?.[item.gold_id])])
    );
    review.ocr_coverage = Object.fromEntries(
      gold.clear_ocr.map(item => [item.gold_id, Boolean(review.ocr_coverage?.[item.gold_id])])
    );
  });
}

function candidateLetter(slot) {
  return String.fromCharCode(64 + Number(slot.split("_")[1]));
}

function renderCandidateTabs() {
  $("#candidateTabs").innerHTML = state.currentTask.candidates.map(candidate => {
    const review = state.candidateDrafts[candidate.slot];
    const stateLabel = candidate.available ? (review ? "已载入" : "未评价") : "无合法候选";
    return `<button class="candidate-tab ${candidate.slot === state.activeCandidate ? "active" : ""}" data-slot="${candidate.slot}" role="tab">
      候选 ${candidateLetter(candidate.slot)}<span class="tab-state">${stateLabel}</span>
    </button>`;
  }).join("");
  $$(".candidate-tab").forEach(tab => tab.addEventListener("click", () => {
    syncCandidateFromDom();
    state.activeCandidate = tab.dataset.slot;
    renderCandidateTabs();
    renderCandidateContent();
  }));
}

function annotationSummary(annotation) {
  const scene = annotation.scene || {};
  const capture = annotation.capture_visual || {};
  return `
    <p><strong>场景：</strong>${escapeHtml(scene.primary_type || "空")} · ${escapeHtml(scene.environment || "空")} · ${escapeHtml(scene.media_type || "空")}</p>
    <p><strong>画面：</strong>${escapeHtml(capture.time_of_day || "空")} · ${escapeHtml(capture.weather || "空")} · ${escapeHtml(capture.lighting || "空")}</p>`;
}

function entityLabel(entity) {
  const count = entity.count_exact && entity.count != null ? ` ×${entity.count}` : " ×?";
  const attributes = entity.attributes || {};
  const details = [...(attributes.colors_zh || []), ...(attributes.states_zh || [])].slice(0, 3);
  return `${entity.name_zh || "未命名"}${count}${details.length ? ` · ${details.join("/")}` : ""}`;
}

function relationLabel(relation, entities) {
  const names = Object.fromEntries(entities.map(entity => [entity.entity_id, entity.name_zh || entity.entity_id]));
  return `${names[relation.subject_id] || relation.subject_id} · ${relation.predicate} · ${names[relation.object_id] || relation.object_id}`;
}

function renderCandidateContent() {
  const candidate = state.currentTask.candidates.find(item => item.slot === state.activeCandidate);
  const review = state.candidateDrafts[candidate.slot];
  const annotation = candidate.annotation || {};
  const entities = annotation.entities || [];
  const ocr = annotation.ocr || [];
  const relations = annotation.relations || [];
  const gold = state.currentTask.review.gold;

  const unavailableNotice = candidate.available ? "" : `<div class="unavailable">该位置没有合法候选。按空标注评价覆盖率，并对描述评分。</div>`;
  const entityRows = entities.length ? entities.map(entity => `
    <tr data-entity-id="${escapeHtml(entity.entity_id)}">
      <td><strong>${escapeHtml(entityLabel(entity))}</strong><br><span class="muted">${escapeHtml(entity.entity_type)} · ${escapeHtml(entity.position_zone)}</span></td>
      <td><select class="compact-select entity-support">
        <option value="" ${!review.entity_judgments[entity.entity_id]?.support ? "selected" : ""}>请选择</option>
        <option value="supported" ${review.entity_judgments[entity.entity_id]?.support === "supported" ? "selected" : ""}>有图像依据</option>
        <option value="unsupported" ${review.entity_judgments[entity.entity_id]?.support === "unsupported" ? "selected" : ""}>不存在/类别错误</option>
        <option value="uncertain" ${review.entity_judgments[entity.entity_id]?.support === "uncertain" ? "selected" : ""}>无法判断</option>
      </select></td>
      <td>${entity.count_exact && entity.count != null ? `<select class="compact-select entity-count">
        <option value="" ${!review.entity_judgments[entity.entity_id]?.count ? "selected" : ""}>请选择</option>
        <option value="correct" ${review.entity_judgments[entity.entity_id]?.count === "correct" ? "selected" : ""}>数量正确</option>
        <option value="incorrect" ${review.entity_judgments[entity.entity_id]?.count === "incorrect" ? "selected" : ""}>数量错误</option>
        <option value="not_evaluable" ${review.entity_judgments[entity.entity_id]?.count === "not_evaluable" ? "selected" : ""}>不可精确计数</option>
      </select>` : `<span class="muted">未声称精确数量</span>`}</td>
    </tr>`).join("") : `<tr><td colspan="3" class="muted">没有实体</td></tr>`;

  const goldCoverage = gold.salient_entities.length ? gold.salient_entities.map(item => `
    <label class="check-row"><input type="checkbox" class="salient-coverage" data-gold-id="${item.gold_id}" ${review.salient_coverage[item.gold_id] ? "checked" : ""}><span>${escapeHtml(item.name)}${item.count_evaluable ? ` ×${item.count}` : ""}</span></label>`).join("") : `<p class="muted">人工参考没有主要实体</p>`;

  const ocrRows = ocr.length ? ocr.map(item => `
    <tr data-text-id="${escapeHtml(item.text_id)}">
      <td><strong>${escapeHtml(item.text_raw)}</strong><br><span class="muted">${escapeHtml(item.legibility)} · ${escapeHtml(item.language)}</span></td>
      <td><select class="compact-select ocr-status">
        <option value="" ${!review.ocr_judgments[item.text_id]?.status ? "selected" : ""}>请选择</option>
        <option value="correct" ${review.ocr_judgments[item.text_id]?.status === "correct" ? "selected" : ""}>完全正确</option>
        <option value="partial" ${review.ocr_judgments[item.text_id]?.status === "partial" ? "selected" : ""}>部分正确</option>
        <option value="invented" ${review.ocr_judgments[item.text_id]?.status === "invented" ? "selected" : ""}>凭空文字</option>
        <option value="unreadable" ${review.ocr_judgments[item.text_id]?.status === "unreadable" ? "selected" : ""}>原图不可读</option>
      </select></td>
      <td><input class="ocr-correction" maxlength="300" placeholder="部分正确时填写人工转写" value="${escapeHtml(review.ocr_judgments[item.text_id]?.corrected_text || "")}"></td>
    </tr>`).join("") : `<tr><td colspan="3" class="muted">没有 OCR</td></tr>`;

  const ocrCoverage = gold.clear_ocr.length ? gold.clear_ocr.map(item => `
    <label class="check-row"><input type="checkbox" class="ocr-coverage" data-gold-id="${item.gold_id}" ${review.ocr_coverage[item.gold_id] ? "checked" : ""}><span>“${escapeHtml(item.text)}”</span></label>`).join("") : `<p class="muted">人工参考没有清晰 OCR</p>`;

  const relationRows = relations.length ? relations.map((relation, index) => {
    const relationId = `r${index + 1}`;
    return `<tr data-relation-id="${relationId}"><td>${escapeHtml(relationLabel(relation, entities))}</td><td><select class="compact-select relation-status">
      <option value="" ${!review.relation_judgments[relationId] ? "selected" : ""}>请选择</option>
      <option value="correct" ${review.relation_judgments[relationId] === "correct" ? "selected" : ""}>正确</option>
      <option value="incorrect" ${review.relation_judgments[relationId] === "incorrect" ? "selected" : ""}>错误</option>
      <option value="uncertain" ${review.relation_judgments[relationId] === "uncertain" ? "selected" : ""}>无法判断</option>
    </select></td></tr>`;
  }).join("") : `<tr><td colspan="2" class="muted">没有关系</td></tr>`;

  const captions = annotation.captions || {};
  $("#candidateContent").innerHTML = `
    <div class="candidate-heading"><h3>候选 ${candidateLetter(candidate.slot)}</h3><span class="status-chip ${candidate.available ? "pending" : ""}">${candidate.available ? "盲评" : "无候选"}</span></div>
    ${unavailableNotice}
    <div class="candidate-summary">${annotationSummary(annotation)}</div>

    <section class="entry-section">
      <div class="section-line"><div><h3>实体存在性与计数</h3><p>存在性与颜色、bbox 分开判断。</p></div></div>
      <div class="quick-actions"><button type="button" class="secondary" id="entitiesAllSupported">实体全部有依据</button><button type="button" class="quiet" id="countsAllCorrect">可计数项全部正确</button></div>
      <table class="judgment-table"><thead><tr><th>候选实体</th><th>存在性</th><th>精确数量</th></tr></thead><tbody>${entityRows}</tbody></table>
    </section>

    <section class="entry-section">
      <div class="section-line"><div><h3>主要实体覆盖</h3><p>勾选这份候选已经覆盖的人工参考实体。</p></div><button type="button" class="quiet" id="allGoldCovered">全部覆盖</button></div>
      <div class="coverage-list">${goldCoverage}</div>
    </section>

    <section class="entry-section">
      <div class="section-line"><div><h3>OCR</h3><p>部分正确时填写人工转写，用于计算字符错误率。</p></div><button type="button" class="quiet" id="ocrAllCorrect">候选 OCR 全部正确</button></div>
      <table class="judgment-table"><thead><tr><th>候选文字</th><th>判断</th><th>人工转写</th></tr></thead><tbody>${ocrRows}</tbody></table>
      <div class="section-line" style="margin-top:16px"><div><h3>清晰 OCR 覆盖</h3></div><button type="button" class="quiet" id="allGoldOcrCovered">全部覆盖</button></div>
      <div class="coverage-list">${ocrCoverage}</div>
    </section>

    <section class="entry-section">
      <div class="section-line"><div><h3>关系</h3><p>主语、宾语、方向全部正确才算正确。</p></div><button type="button" class="quiet" id="relationsAllCorrect">全部正确</button></div>
      <table class="judgment-table"><thead><tr><th>关系三元组</th><th>判断</th></tr></thead><tbody>${relationRows}</tbody></table>
    </section>

    <section class="entry-section">
      <div class="section-line"><div><h3>Caption</h3><p>新增事实只计算具体、无图像依据的硬断言。</p></div></div>
      <div class="caption-box">${escapeHtml(captions.dense_zh || captions.short_zh || "没有可用描述")}</div>
      <div class="two-column-fields" style="margin-top:13px">
        <label class="field-control">新增硬事实数量<input id="captionNewFactCount" type="number" min="0" max="50" value="${review.caption_new_fact_count}"></label>
        <label class="field-control">新增事实说明<input id="captionNewFactNotes" maxlength="2000" value="${escapeHtml(review.caption_new_fact_notes)}" placeholder="数量大于 0 时填写"></label>
      </div>
      <div class="score-row"><label for="captionCorrectness">事实正确性</label><input id="captionCorrectness" type="range" min="1" max="5" value="${review.caption_correctness}"><span class="score-value">${review.caption_correctness}</span></div>
      <div class="score-row"><label for="captionCompleteness">主要内容完整性</label><input id="captionCompleteness" type="range" min="1" max="5" value="${review.caption_completeness}"><span class="score-value">${review.caption_completeness}</span></div>
      <label class="privacy-row"><input id="privacyViolation" type="checkbox" ${review.privacy_violation ? "checked" : ""}> 存在敏感身份推断</label>
      <label class="field-control">隐私问题说明<input id="privacyNotes" maxlength="1000" value="${escapeHtml(review.privacy_notes)}" placeholder="勾选时填写"></label>
      <label class="field-control notes-field">候选备注（可选）<textarea id="candidateNotes" maxlength="2000" rows="2">${escapeHtml(review.notes)}</textarea></label>
    </section>`;

  bindCandidateControls();
}

function bindCandidateControls() {
  $("#entitiesAllSupported")?.addEventListener("click", () => { $$(".entity-support").forEach(select => { select.value = "supported"; }); });
  $("#countsAllCorrect")?.addEventListener("click", () => { $$(".entity-count").forEach(select => { select.value = "correct"; }); });
  $("#allGoldCovered")?.addEventListener("click", () => { $$(".salient-coverage").forEach(input => { input.checked = true; }); });
  $("#ocrAllCorrect")?.addEventListener("click", () => { $$(".ocr-status").forEach(select => { select.value = "correct"; }); });
  $("#allGoldOcrCovered")?.addEventListener("click", () => { $$(".ocr-coverage").forEach(input => { input.checked = true; }); });
  $("#relationsAllCorrect")?.addEventListener("click", () => { $$(".relation-status").forEach(select => { select.value = "correct"; }); });
  [$("#captionCorrectness"), $("#captionCompleteness")].forEach(input => input?.addEventListener("input", () => {
    input.nextElementSibling.textContent = input.value;
  }));
}

function syncCandidateFromDom() {
  if (!state.currentTask || $("#candidatePhase").classList.contains("hidden")) return;
  const candidate = state.currentTask.candidates.find(item => item.slot === state.activeCandidate);
  const review = state.candidateDrafts[candidate.slot] || defaultCandidateReview(candidate);
  $$("tr[data-entity-id]").forEach(row => {
    const id = row.dataset.entityId;
    const support = $(".entity-support", row).value;
    const countSelect = $(".entity-count", row);
    const count = countSelect ? countSelect.value : "not_evaluable";
    if (support && count) review.entity_judgments[id] = { support, count };
    else delete review.entity_judgments[id];
  });
  $$(".salient-coverage").forEach(input => { review.salient_coverage[input.dataset.goldId] = input.checked; });
  $$("tr[data-text-id]").forEach(row => {
    const id = row.dataset.textId;
    const status = $(".ocr-status", row).value;
    if (status) review.ocr_judgments[id] = { status, corrected_text: $(".ocr-correction", row).value.trim() };
    else delete review.ocr_judgments[id];
  });
  $$(".ocr-coverage").forEach(input => { review.ocr_coverage[input.dataset.goldId] = input.checked; });
  $$("tr[data-relation-id]").forEach(row => {
    const status = $(".relation-status", row).value;
    if (status) review.relation_judgments[row.dataset.relationId] = status;
    else delete review.relation_judgments[row.dataset.relationId];
  });
  review.caption_new_fact_count = Number($("#captionNewFactCount")?.value || 0);
  review.caption_new_fact_notes = $("#captionNewFactNotes")?.value.trim() || "";
  review.caption_correctness = Number($("#captionCorrectness")?.value || 3);
  review.caption_completeness = Number($("#captionCompleteness")?.value || 3);
  review.privacy_violation = Boolean($("#privacyViolation")?.checked);
  review.privacy_notes = $("#privacyNotes")?.value.trim() || "";
  review.notes = $("#candidateNotes")?.value.trim() || "";
  state.candidateDrafts[candidate.slot] = review;
}

function validateBeforeSubmit() {
  for (const candidate of state.currentTask.candidates) {
    const letter = candidateLetter(candidate.slot);
    const review = state.candidateDrafts[candidate.slot];
    for (const item of Object.values(review.ocr_judgments)) {
      if (item.status === "partial" && !item.corrected_text) throw new Error(`候选 ${letter} 有部分正确 OCR，但未填写人工转写`);
    }
    if (review.caption_new_fact_count > 0 && !review.caption_new_fact_notes) throw new Error(`候选 ${letter} 有新增事实，请填写说明`);
    if (review.privacy_violation && !review.privacy_notes) throw new Error(`候选 ${letter} 勾选了隐私违规，请填写说明`);
  }
}

async function saveCandidates(submit) {
  try {
    syncCandidateFromDom();
    if (submit) validateBeforeSubmit();
    const body = await api(`/api/review/${encodeURIComponent(state.currentImageId)}/candidates?submit=${submit}`, {
      method: "PUT", body: JSON.stringify(state.candidateDrafts),
    });
    state.currentTask.review = body.review;
    await loadSummary();
    showToast(submit ? "本图已提交" : "草稿已保存");
    if (submit) {
      const next = state.summary.tasks.find(task => task.phase !== "submitted");
      if (next) await loadTask(next.image_id);
    }
  } catch (error) { showToast(error.message, true); }
}

async function reopenReview() {
  if (!confirm("重新打开后，这张图将从已提交退回草稿，导出时暂不计入。继续吗？")) return;
  try {
    const body = await api(`/api/review/${encodeURIComponent(state.currentImageId)}/reopen`, {
      method: "PUT", body: JSON.stringify({}),
    });
    state.currentTask.review = body.review;
    await loadSummary();
    renderCandidateContent();
    updateSubmissionControls();
    showToast("本图已重新打开，可以修改");
  } catch (error) { showToast(error.message, true); }
}

function bindStaticEvents() {
  $("#loginButton").addEventListener("click", login);
  $("#reviewerInput").addEventListener("keydown", event => { if (event.key === "Enter") login(); });
  $("#switchReviewerButton").addEventListener("click", () => { state.reviewer = ""; setView(false); });
  $$(".filter").forEach(button => button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    $$(".filter").forEach(item => item.classList.toggle("active", item === button));
    renderTaskList();
  }));
  $("#fitImageButton").addEventListener("click", () => $("#imageStage").classList.toggle("actual"));
  $("#addGoldEntity").addEventListener("click", () => addGoldEntityRow());
  $("#addGoldOcr").addEventListener("click", () => addGoldOcrRow());
  $("#goldForm").addEventListener("submit", saveGold);
  $("#backToGoldButton").addEventListener("click", () => {
    syncCandidateFromDom(); fillGoldForm(state.currentTask.review.gold); showGoldPhase();
  });
  $("#saveDraftButton").addEventListener("click", () => saveCandidates(false));
  $("#submitReviewButton").addEventListener("click", () => saveCandidates(true));
  $("#reopenReviewButton").addEventListener("click", reopenReview);
}

document.addEventListener("DOMContentLoaded", () => {
  bindStaticEvents();
  const saved = localStorage.getItem("m1_audit_reviewer") || "";
  $("#reviewerInput").value = saved;
  if (saved) login();
});
