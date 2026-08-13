"use strict";

const state = {
  reviewer: "",
  summary: null,
  currentTask: null,
  currentImageId: null,
  filter: "all",
  dirty: false,
  saving: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast show${error ? " error" : ""}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.className = "toast"; }, 2600);
}

async function api(path, options = {}) {
  const separator = path.includes("?") ? "&" : "?";
  const url = `${path}${separator}reviewer=${encodeURIComponent(state.reviewer)}`;
  const response = await fetch(url, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json" } : {},
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败：${response.status}`);
  return data;
}

function candidateLetter(slot) {
  return String.fromCharCode(64 + Number(slot.split("_")[1]));
}

function scoreOptions(slot, field, selected) {
  return [1, 2, 3, 4, 5].map(score => `
    <label title="${score} 分">
      <input type="radio" name="${slot}-${field}" value="${score}" ${selected === score ? "checked" : ""}>
      <span>${score}</span>
    </label>`).join("");
}

function entityText(entity) {
  const count = entity.count_exact && entity.count != null ? ` ×${entity.count}` : "";
  const attributes = entity.attributes || {};
  const details = [
    ...(attributes.colors_zh || []),
    ...(attributes.states_zh || []),
    attributes.action_zh,
  ].filter(Boolean).slice(0, 3);
  return `${entity.name_zh || "未命名"}${count}${details.length ? `（${details.join("、")}）` : ""}`;
}

function relationText(relation, entityNames) {
  const subject = entityNames[relation.subject_id] || relation.subject_id;
  const object = entityNames[relation.object_id] || relation.object_id;
  const labels = {
    on: "在...上", inside: "在...里面", holding: "拿着", wearing: "穿着",
    left_of: "在...左侧", right_of: "在...右侧", above: "在...上方", below: "在...下方",
    in_front_of: "在...前方", behind: "在...后方", next_to: "靠近", looking_at: "看向",
  };
  const predicate = relation.predicate === "other"
    ? (relation.predicate_other_zh || "其他关系")
    : (labels[relation.predicate] || relation.predicate);
  return `${subject} ${predicate} ${object}`;
}

function annotationHtml(annotation) {
  const scene = annotation.scene || {};
  const capture = annotation.capture_visual || {};
  const captions = annotation.captions || {};
  const entities = annotation.entities || [];
  const ocr = annotation.ocr || [];
  const relations = annotation.relations || [];
  const names = Object.fromEntries(entities.map(entity => [entity.entity_id, entity.name_zh || entity.entity_id]));
  const salientEntities = [...entities]
    .sort((left, right) => ({ primary: 0, secondary: 1, background: 2 }[left.salience] ?? 3) - ({ primary: 0, secondary: 1, background: 2 }[right.salience] ?? 3))
    .slice(0, 8)
    .map(entity => entityText(entity));
  const entityItems = entities.length
    ? entities.map(entity => `<li>${escapeHtml(entityText(entity))}</li>`).join("")
    : `<li class="muted">未标实体</li>`;
  const ocrItems = ocr.length
    ? ocr.map(item => `<li>“${escapeHtml(item.text_raw)}”</li>`).join("")
    : `<li class="muted">未标 OCR</li>`;
  const relationItems = relations.length
    ? relations.map(item => `<li>${escapeHtml(relationText(item, names))}</li>`).join("")
    : `<li class="muted">未标关系</li>`;
  return `
    <div class="caption-block">
      <strong>${escapeHtml(captions.short_zh || "无短描述")}</strong>
      <p>${escapeHtml(captions.dense_zh || "无详细描述")}</p>
      <p class="key-entities"><strong>主要实体：</strong>${escapeHtml(salientEntities.join("；") || "未标实体")}</p>
    </div>
    <dl class="scene-grid">
      <div><dt>场景</dt><dd>${escapeHtml(scene.primary_type || "空")} / ${escapeHtml(scene.environment || "空")} / ${escapeHtml(scene.media_type || "空")}</dd></div>
      <div><dt>拍摄</dt><dd>${escapeHtml(capture.time_of_day || "空")} / ${escapeHtml(capture.weather || "空")} / ${escapeHtml(capture.viewpoint || "空")} / ${escapeHtml(capture.shot_scale || "空")}</dd></div>
    </dl>
    <details><summary>查看全部实体 ${entities.length}</summary><ul>${entityItems}</ul></details>
    <details><summary>OCR ${ocr.length}</summary><ul>${ocrItems}</ul></details>
    <details><summary>关系 ${relations.length}</summary><ul>${relationItems}</ul></details>`;
}

function defaultRating() {
  return { ratings: {}, best_choice: null, notes: "" };
}

function renderCandidate(candidate, saved) {
  const slot = candidate.slot;
  const rating = saved.ratings?.[slot] || {};
  return `<article class="candidate-card" data-slot="${slot}">
    <header><span class="candidate-letter">${candidateLetter(slot)}</span><strong>候选 ${candidateLetter(slot)}</strong></header>
    <div class="annotation-content">${annotationHtml(candidate.annotation || {})}</div>
    <div class="rating-fields">
      <div class="rating-row"><div><strong>准确性</strong><span>事实是否有图像依据</span></div><div class="score-options">${scoreOptions(slot, "accuracy", rating.accuracy)}</div></div>
      <div class="rating-row"><div><strong>完整性</strong><span>主要内容是否覆盖</span></div><div class="score-options">${scoreOptions(slot, "completeness", rating.completeness)}</div></div>
      <div class="rating-row"><div><strong>整体可用性</strong><span>需要多少修改才能使用</span></div><div class="score-options">${scoreOptions(slot, "usability", rating.usability)}</div></div>
      <div class="severe-row"><strong>存在严重错误</strong><div class="binary-options">
        <label><input type="radio" name="${slot}-severe" value="false" ${rating.severe_error === false ? "checked" : ""}><span>没有</span></label>
        <label><input type="radio" name="${slot}-severe" value="true" ${rating.severe_error === true ? "checked" : ""}><span>有</span></label>
      </div></div>
    </div>
  </article>`;
}

function readRadio(name) {
  return $(`input[name="${name}"]:checked`)?.value ?? null;
}

function collectRating() {
  const ratings = {};
  for (const candidate of state.currentTask.candidates) {
    const slot = candidate.slot;
    const accuracy = readRadio(`${slot}-accuracy`);
    const completeness = readRadio(`${slot}-completeness`);
    const usability = readRadio(`${slot}-usability`);
    const severe = readRadio(`${slot}-severe`);
    if ([accuracy, completeness, usability, severe].some(value => value !== null)) {
      ratings[slot] = {
        accuracy: accuracy === null ? null : Number(accuracy),
        completeness: completeness === null ? null : Number(completeness),
        usability: usability === null ? null : Number(usability),
        severe_error: severe === null ? null : severe === "true",
      };
    }
  }
  return {
    ratings,
    best_choice: readRadio("best-choice"),
    notes: $("#notes").value.trim(),
  };
}

function setEditable(editable) {
  $$("#ratingForm input, #ratingForm textarea").forEach(input => { input.disabled = !editable; });
  $("#saveButton").classList.toggle("hidden", !editable);
  $("#submitButton").classList.toggle("hidden", !editable);
  $("#reopenButton").classList.toggle("hidden", editable);
}

function renderTask(task) {
  const saved = task.review?.rating || defaultRating();
  $("#emptyState").classList.add("hidden");
  $("#ratingForm").classList.remove("hidden");
  $("#sampleIndex").textContent = `样本 ${task.sample_index} / ${task.sample_size}`;
  $("#imageTitle").textContent = `图片 ${task.image_id}`;
  $("#ratingImage").src = task.image_url;
  $("#candidateList").innerHTML = task.candidates.map(candidate => renderCandidate(candidate, saved)).join("");
  $("#bestChoice").innerHTML = [
    ...task.candidates.map(candidate => [candidate.slot, `候选 ${candidateLetter(candidate.slot)}`]),
    ["tie", "并列"],
    ["all_unacceptable", "都不合格"],
  ].map(([value, label]) => `<label><input type="radio" name="best-choice" value="${value}" ${saved.best_choice === value ? "checked" : ""}><span>${label}</span></label>`).join("");
  $("#notes").value = saved.notes || "";
  const submitted = task.review?.phase === "submitted";
  $("#statusChip").textContent = submitted ? "已提交" : task.review ? "草稿" : "未开始";
  $("#statusChip").className = `status-chip ${submitted ? "submitted" : "draft"}`;
  $("#saveState").textContent = task.review ? `已保存 ${new Date(task.review.updated_at_utc).toLocaleString()}` : "尚未保存";
  setEditable(!submitted);
  state.dirty = false;
  $$("#ratingForm input, #ratingForm textarea").forEach(input => input.addEventListener("change", markDirty));
  $("#notes").addEventListener("input", markDirty);
}

function markDirty() {
  if (state.currentTask?.review?.phase === "submitted") return;
  state.dirty = true;
  $("#saveState").textContent = "有未保存修改";
}

function renderSummary() {
  const { counts, sample_size: total } = state.summary;
  $("#progressText").textContent = `${counts.submitted} / ${total} 已提交`;
  $("#draftText").textContent = `${counts.draft} 份草稿`;
  $("#progressBar").style.width = `${total ? counts.submitted / total * 100 : 0}%`;
  const tasks = state.summary.tasks.filter(task => {
    if (state.filter === "submitted") return task.phase === "submitted";
    if (state.filter === "pending") return task.phase !== "submitted";
    return true;
  });
  $("#taskList").innerHTML = tasks.map(task => `
    <button class="task-button ${task.phase} ${task.image_id === state.currentImageId ? "active" : ""}" data-image-id="${task.image_id}">
      <span class="task-number">${String(task.sample_index).padStart(2, "0")}</span>
      <span class="task-id">图片 ${escapeHtml(task.image_id)}</span>
      <span class="task-dot"></span>
    </button>`).join("");
  $$(".task-button").forEach(button => button.addEventListener("click", () => loadTask(button.dataset.imageId)));
}

async function refreshSummary() {
  state.summary = await api("/api/summary");
  renderSummary();
}

async function saveCurrent(submit = false, quiet = false) {
  if (!state.currentTask || state.saving || state.currentTask.review?.phase === "submitted") return true;
  if (!state.dirty && !submit) return true;
  state.saving = true;
  try {
    const data = await api(`/api/review/${state.currentImageId}/rating?submit=${submit}`, {
      method: "PUT", body: JSON.stringify(collectRating()),
    });
    state.currentTask.review = data.review;
    state.dirty = false;
    $("#saveState").textContent = `已保存 ${new Date(data.review.updated_at_utc).toLocaleString()}`;
    await refreshSummary();
    if (!quiet) showToast(submit ? "本图已提交" : "草稿已保存");
    return true;
  } catch (error) {
    showToast(error.message, true);
    return false;
  } finally {
    state.saving = false;
  }
}

async function loadTask(imageId) {
  if (state.currentImageId === imageId) return;
  if (state.currentTask && state.dirty) {
    const saved = await saveCurrent(false, true);
    if (!saved) return;
  }
  try {
    state.currentImageId = imageId;
    state.currentTask = await api(`/api/task/${imageId}`);
    renderSummary();
    renderTask(state.currentTask);
  } catch (error) {
    showToast(error.message, true);
  }
}

function nextPendingImage() {
  const currentIndex = state.summary.tasks.findIndex(task => task.image_id === state.currentImageId);
  const after = state.summary.tasks.slice(currentIndex + 1).find(task => task.phase !== "submitted");
  const before = state.summary.tasks.slice(0, currentIndex).find(task => task.phase !== "submitted");
  return (after || before)?.image_id || null;
}

async function login() {
  const reviewer = $("#reviewerInput").value.replace(/\s+/g, " ").trim();
  if (!reviewer) return showToast("请输入评审者名称", true);
  state.reviewer = reviewer;
  localStorage.setItem("m1BlindRatingReviewer", reviewer);
  try {
    await refreshSummary();
    $("#reviewerLabel").textContent = reviewer;
    $("#loginView").classList.add("hidden");
    $("#appView").classList.remove("hidden");
    const first = state.summary.tasks.find(task => task.phase !== "submitted") || state.summary.tasks[0];
    if (first) await loadTask(first.image_id);
  } catch (error) {
    showToast(error.message, true);
  }
}

$("#loginButton").addEventListener("click", login);
$("#reviewerInput").addEventListener("keydown", event => { if (event.key === "Enter") login(); });
$("#switchReviewerButton").addEventListener("click", async () => {
  if (state.dirty) await saveCurrent(false, true);
  state.currentTask = null;
  state.currentImageId = null;
  $("#appView").classList.add("hidden");
  $("#loginView").classList.remove("hidden");
});
$("#fitImageButton").addEventListener("click", () => $("#imageStage").classList.toggle("actual"));
$("#saveButton").addEventListener("click", () => saveCurrent(false));
$("#ratingForm").addEventListener("submit", async event => {
  event.preventDefault();
  const submitted = await saveCurrent(true);
  if (!submitted) return;
  state.currentTask = await api(`/api/task/${state.currentImageId}`);
  renderTask(state.currentTask);
  const next = nextPendingImage();
  if (next) await loadTask(next); else showToast("50 张评审已全部完成");
});
$("#reopenButton").addEventListener("click", async () => {
  try {
    const data = await api(`/api/review/${state.currentImageId}/reopen`, { method: "PUT" });
    state.currentTask.review = data.review;
    renderTask(state.currentTask);
    await refreshSummary();
    showToast("本图已重新打开");
  } catch (error) { showToast(error.message, true); }
});
$$('.filter').forEach(button => button.addEventListener("click", () => {
  state.filter = button.dataset.filter;
  $$('.filter').forEach(item => item.classList.toggle("active", item === button));
  renderSummary();
}));

window.addEventListener("beforeunload", event => {
  if (state.dirty) { event.preventDefault(); event.returnValue = ""; }
});

const remembered = localStorage.getItem("m1BlindRatingReviewer");
if (remembered) $("#reviewerInput").value = remembered;
