let CURRENT_VIEW = "predict"; // "predict" | "explain" | "explain_pred"
let LAST_PREDICT = null;
let LAST_EXPLAIN = null;
let LAST_EXPLAIN_PRED = null;

// ---------- utilities ----------
function $(id){ return document.getElementById(id); }

function selectedCustIds() {
  const sel = $("custSelect");
  return Array.from(sel.selectedOptions).map(o => o.value);
}

function setSelectedCount(){
  $("selectedCount").textContent = String(selectedCustIds().length);
}

function setStatus(msg){
  $("status").textContent = msg || "";
}

function toast(title, msg){
  $("toastT1").textContent = title || "Info";
  $("toastT2").textContent = msg || "";
  const t = $("toast");
  t.classList.add("show");
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(()=>t.classList.remove("show"), 2800);
}

function setLoading(which, on){
  const map = {
    predict: ["spinPredict", "btnPredict"],
    explain: ["spinExplain", "btnExplain"],
    explainPred: ["spinExplainPred", "btnExplainPred"]
  };
  const pair = map[which];
  if(!pair) return;
  $(pair[0]).style.display = on ? "inline-block" : "none";
  $(pair[1]).disabled = on;
}

function fmtMoney(x){
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  return Number(x).toFixed(2);
}

function fmtPctPoints(x){
  // x is "percentage points", e.g. 0.172 means 0.172%
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  return Number(x).toFixed(3);
}

function actionTag(action){
  const a = (action || "HOLD").toUpperCase();
  const cls = a === "CLI" ? "cli" : (a === "CLD" ? "cld" : "hold");
  return `<span class="tag ${cls}"><span class="tDot"></span>${a}</span>`;
}

// ---------- health ----------
async function checkHealth(){
  try{
    const res = await fetch("/health");
    if(!res.ok) throw new Error("health not ok");
    $("healthDot").style.background = "var(--good)";
    $("healthText").textContent = "Service online";
  }catch(e){
    $("healthDot").style.background = "var(--bad)";
    $("healthText").textContent = "Service offline";
  }
}

// ---------- load customers ----------
async function loadCustomers() {
  const res = await fetch("/customers");
  if(!res.ok) throw new Error(`Failed /customers: ${res.status}`);
  const data = await res.json();

  const sel = $("custSelect");
  sel.innerHTML = "";
  for (const cid of data.customer_ids) {
    const opt = document.createElement("option");
    opt.value = cid;
    opt.textContent = cid;
    sel.appendChild(opt);
  }
  setSelectedCount();
}

// ---------- table render (sortable) ----------
function renderPredictTable(items){
  const rows = (items || []).map(r => ({
    cust_id: r.cust_id,
    action_taken: r.action_taken,
    mag: r.magnitude_percentage,
    prev: r.prev_credit_limit,
    updated: r.updated_credit_limit,
    gate_prob: r.gate_prob,
    dir_prob: r.dir_prob
  }));

  const sortState = { key: "cust_id", asc: true };

  function sortBy(key){
    if(sortState.key === key) sortState.asc = !sortState.asc;
    else { sortState.key = key; sortState.asc = true; }

    rows.sort((a,b)=>{
      const va = a[key], vb = b[key];
      const na = Number(va), nb = Number(vb);
      let cmp = 0;
      if(!Number.isNaN(na) && !Number.isNaN(nb)) cmp = na - nb;
      else cmp = String(va).localeCompare(String(vb));
      return sortState.asc ? cmp : -cmp;
    });

    mount();
  }

  function spotlight(selectors){
    const all = document.querySelectorAll(".panel, .card, .tableWrap");
    all.forEach(el => el.classList.add("dimmed"));

    selectors.forEach(sel => {
      document.querySelectorAll(sel).forEach(el => {
        el.classList.remove("dimmed");
        el.classList.add("highlight", "focus-ring");
      });
    });

    setTimeout(() => {
      all.forEach(el => el.classList.remove("dimmed"));
      document.querySelectorAll(".highlight").forEach(el => {
        el.classList.remove("highlight", "focus-ring");
      });
    }, 4200);
  }

  function mount(){
    let html = `
      <div class="tableWrap">
        <table>
          <thead>
            <tr>
              <th data-k="cust_id">Customer ID</th>
              <th data-k="action_taken">Action</th>
              <th data-k="mag" class="num">Magnitude %</th>
              <th data-k="prev" class="num">Current Limit</th>
              <th data-k="updated" class="num">Updated Limit</th>
              <th data-k="gate_prob" class="num">Gate Prob</th>
              <th data-k="dir_prob" class="num">Dir Prob</th>
            </tr>
          </thead>
          <tbody>
    `;

    for(const r of rows){
      html += `
        <tr>
          <td class="mono">${r.cust_id}</td>
          <td>${actionTag(r.action_taken)}</td>
          <td class="num mono">${fmtPctPoints(r.mag)}</td>
          <td class="num mono">${fmtMoney(r.prev)}</td>
          <td class="num mono">${fmtMoney(r.updated)}</td>
          <td class="num mono">${(r.gate_prob === null || r.gate_prob === undefined) ? "-" : Number(r.gate_prob).toFixed(3)}</td>
          <td class="num mono">${(r.dir_prob === null || r.dir_prob === undefined) ? "-" : Number(r.dir_prob).toFixed(3)}</td>
        </tr>
      `;
    }

    html += `</tbody></table></div>`;
    $("results").innerHTML = html;

    document.querySelectorAll("th[data-k]").forEach(th=>{
      th.addEventListener("click", ()=> sortBy(th.getAttribute("data-k")));
    });
  }

  mount();
}

function parseImpactLine(line){
  const m = line.match(/impact=([+-]?\d+(\.\d+)?(e[+-]?\d+)?)/i);
  if(!m) return { text: line, sign: 0 };
  const v = Number(m[1]);
  if(Number.isNaN(v)) return { text: line, sign: 0 };
  return { text: line, sign: v };
}

function renderExplainCards(items){
  const cards = (items || []).map(r=>{
    const cid = r.cust_id;
    const action = r.action_taken;
    const meta = r.meta || {};

    const prev = meta.prev_credit_limit;
    const updated = meta.updated_limit;
    const mag = meta.magnitude_pct;
    const gate = meta.gate_prob;
    const dir = meta.dir_prob;

    const updatedDisplay = (String(action).toUpperCase() === "HOLD") ? "-" : fmtMoney(updated);

    const lines = (r.explanation_lines || []).map(l=>{
      const p = parseImpactLine(l);
      if(p.sign > 0) return `<li><span class="impactPos">▲</span> ${p.text}</li>`;
      if(p.sign < 0) return `<li><span class="impactNeg">▼</span> ${p.text}</li>`;
      return `<li>${p.text}</li>`;
    }).join("");

    return `
      <div class="card">
        <div class="cardHd">
          <div class="cardTitle">
            <div class="cid mono">cust_id ${cid}</div>
            ${actionTag(action)}
          </div>
          <div class="pill">Top K: <b>${(r.attributions && r.attributions.length) ? r.attributions.length : (r.explanation_lines ? r.explanation_lines.length : 0)}</b></div>
        </div>

        <div class="kv">
          <div class="pill">Magnitude: <b class="mono">${(mag===null||mag===undefined)? "-" : fmtPctPoints(mag)}%</b></div>
          <div class="pill">Current: <b class="mono">${(prev===null||prev===undefined)? "-" : fmtMoney(prev)}</b></div>
          <div class="pill">Updated: <b class="mono">${updatedDisplay}</b></div>
          <div class="pill">Gate: <b class="mono">${(gate===null||gate===undefined)? "-" : Number(gate).toFixed(3)}</b></div>
          <div class="pill">Dir: <b class="mono">${(dir===null||dir===undefined)? "-" : Number(dir).toFixed(3)}</b></div>
        </div>

        <div class="explain">
          <b>Why this decision? ↓(Model view)</b>
          <ul>${lines}</ul>
        </div>
      </div>
    `;
  });

  $("results").innerHTML = `<div class="cards">${cards.join("")}</div>`;
}

function renderExplainPredTable(items, ollamaInfo){
  const rows = (items || []).map(r => ({
    cust_id: r.cust_id,
    action_taken: r.action_taken,
    mag: r.magnitude_percentage,
    prev: r.prev_credit_limit,
    updated: r.updated_credit_limit,
    gate_prob: r.gate_prob,
    dir_prob: r.dir_prob,
    explanation_customer: r.customer_explanation,
    recourse: r.recourse || [],
    policy: r.disclosure || null
  }));

  let html = `
    <div class="hint" style="margin-bottom:12px;">
      Customer Friendly Explanations
    </div>
    <div class="tableWrap">
      <table style="min-width:1200px;">
        <thead>
          <tr>
            <th>Customer ID</th>
            <th>Action</th>
            <th class="num">Magnitude %</th>
            <th class="num">Current Limit</th>
            <th class="num">Updated Limit</th>
            <th>Customer Explanation</th>
            <th>What can improve</th>
          </tr>
        </thead>
        <tbody>
  `;

  for(const r of rows){
    const action = String(r.action_taken || "HOLD").toUpperCase();
    const updatedDisplay = (action === "HOLD") ? "-" : fmtMoney(r.updated);

    const rec = (r.recourse || []).map(x=>`<li class="small">${x}</li>`).join("");
    const pol = r.policy ? `
      <details>
        <summary>View</summary>
        <div class="small">
          <b>Policy disclosure</b>
          <ul>${(r.policy.policy_disclosure || []).map(x=>`<li>${x}</li>`).join("")}</ul>
          <b>Fairness notes</b>
          <ul>${(r.policy.fairness_notes || []).map(x=>`<li>${x}</li>`).join("")}</ul>
        </div>
      </details>
    ` : "-";

    html += `
      <tr>
        <td class="mono">${r.cust_id}</td>
        <td>${actionTag(r.action_taken)}</td>
        <td class="num mono">${fmtPctPoints(r.mag)}</td>
        <td class="num mono">${fmtMoney(r.prev)}</td>
        <td class="num mono">${updatedDisplay}</td>
        <td style="white-space:normal; min-width:320px;">${(r.explanation_customer || "-")}</td>
        <td style="white-space:normal; min-width:320px;">
          <ul style="margin:0; padding-left:18px;">${rec || "<li class='small'>-</li>"}</ul>
        </td>
      </tr>
    `;
  }

  html += `</tbody></table></div>`;
  $("results").innerHTML = html;
}

// ---------- view toggle ----------
function setTab(active){
  CURRENT_VIEW = active;

  $("tabPredict").classList.toggle("active", active === "predict");
  $("tabExplain").classList.toggle("active", active === "explain");
  $("tabExplainPred").classList.toggle("active", active === "explain_pred");

  $("viewModeLabel").textContent =
    active === "predict" ? "Predictions" :
    active === "explain" ? "Model Explanations" :
    "Customer Explanation";

  if(active === "predict" && LAST_PREDICT) renderPredictTable(LAST_PREDICT.items);
  else if(active === "explain" && LAST_EXPLAIN) renderExplainCards(LAST_EXPLAIN.items);
  else if(active === "explain_pred" && LAST_EXPLAIN_PRED) renderExplainPredTable(LAST_EXPLAIN_PRED.items, LAST_EXPLAIN_PRED.ollama);
}

// ---------- API calls ----------
async function doPredict(){
  const ids = selectedCustIds();
  if(!ids.length) return toast("Select customers", "Please select at least one customer ID.");

  setLoading("predict", true);
  setStatus("Running prediction…");
  try{
    const res = await fetch("/predict_customer_limit", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ cust_ids: ids })
    });

    if(!res.ok){
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const data = await res.json();
    LAST_PREDICT = data;

    setStatus("");
    setTab("predict");
    renderPredictTable(data.items);
    toast("Prediction complete", `Processed ${data.items?.length || 0} customers.`);
  }catch(e){
    setStatus("Prediction failed: " + e);
    toast("Prediction failed", String(e));
  }finally{
    setLoading("predict", false);
  }
}

async function doExplain(){
  const ids = selectedCustIds();
  if(!ids.length) return toast("Select customers", "Please select at least one customer ID.");

  setLoading("explain", true);
  setStatus("Generating model explanations…");
  try{
    const res = await fetch("/explain_customer_limit", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ cust_ids: ids, stage:"auto", top_k: 5 })
    });

    if(!res.ok){
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const data = await res.json();
    LAST_EXPLAIN = data;

    setStatus("");
    setTab("explain");
    renderExplainCards(data.items);
    toast("Explanation ready", `Generated model explanations for ${data.items?.length || 0} customers.`);
  }catch(e){
    setStatus("Explain failed: " + e);
    toast("Explain failed", String(e));
  }finally{
    setLoading("explain", false);
  }
}

async function doExplainPredicted(){
  const ids = selectedCustIds();
  if(!ids.length) return toast("Select customers", "Please select at least one customer ID.");

  setLoading("explainPred", true);
  setStatus("Generating customer explanations…");
  try{
    const res = await fetch("/explain_predicted_limit", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ cust_ids: ids, top_k: 5, use_llm: true })
    });

    if(!res.ok){
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const data = await res.json();
    LAST_EXPLAIN_PRED = data;

    setStatus("");
    setTab("explain_pred");
    renderExplainPredTable(data.items, data.ollama);
    toast("Customer explanation ready", `Generated explanations for ${data.items?.length || 0} customers.`);
  }catch(e){
    setStatus("Explain predicted failed: " + e);
    toast("Explain predicted failed", String(e));
  }finally{
    setLoading("explainPred", false);
  }
}

function clearOutput(){
  LAST_PREDICT = null;
  LAST_EXPLAIN = null;
  LAST_EXPLAIN_PRED = null;
  $("results").innerHTML = `<div class="hint">Select customers on the left and run <b>Predict</b> or <b>Explain</b>. Results will appear here.</div>`;
  setStatus("");
  toast("Cleared", "Output has been cleared.");
}

// ---------- wire up ----------
$("btnPredict").addEventListener("click", doPredict);
$("btnExplain").addEventListener("click", doExplain);
$("btnExplainPred").addEventListener("click", doExplainPredicted);
$("btnClear").addEventListener("click", clearOutput);

$("custSelect").addEventListener("change", setSelectedCount);

$("tabPredict").addEventListener("click", ()=> setTab("predict"));
$("tabExplain").addEventListener("click", ()=> setTab("explain"));
$("tabExplainPred").addEventListener("click", ()=> setTab("explain_pred"));

// ---------- boot ----------
(async function boot(){
  await checkHealth();
  try{
    await loadCustomers();
    toast("Ready", "Customers loaded. Select IDs to begin.");
  }catch(e){
    setStatus("Failed to load customers: " + e);
    toast("Startup error", String(e));
  }
})();
