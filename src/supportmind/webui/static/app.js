(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // ---------- Toast ----------
  const toastEl = $("#toast");
  let toastTimer = null;
  function showToast(message) {
    toastEl.textContent = message;
    toastEl.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("is-visible"), 4200);
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch (_) {
        /* ignore parse failure */
      }
      throw new Error(detail);
    }
    return res.json();
  }

  // ---------- Tabs ----------
  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");
      $$(".panel").forEach((p) => p.classList.remove("is-active"));
      $(`#tab-${btn.dataset.tab}`).classList.add("is-active");
    });
  });

  // ---------- Health status ----------
  async function pollHealth() {
    const pill = $("#statusPill");
    try {
      await api("/health");
      pill.classList.remove("is-down");
      pill.lastChild.textContent = " live";
    } catch (_) {
      pill.classList.add("is-down");
      pill.lastChild.textContent = " unreachable";
    }
  }
  pollHealth();
  setInterval(pollHealth, 30000);

  // ---------- Ask tab ----------
  const modeSegment = $("#modeSegment");
  $$("button", modeSegment).forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("button", modeSegment).forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      modeSegment.dataset.value = btn.dataset.value;
    });
  });

  function setBusy(button, busy) {
    button.disabled = busy;
    $(".btn-label", button).style.visibility = busy ? "hidden" : "visible";
    $(".spinner", button).hidden = !busy;
  }

  $("#askSubmit").addEventListener("click", async () => {
    const query = $("#askInput").value.trim();
    if (!query) return;
    const button = $("#askSubmit");
    setBusy(button, true);
    try {
      const payload = {
        query,
        mode: modeSegment.dataset.value,
        use_reranker: $("#rerankerToggle").checked,
      };
      const data = await api("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderAskResult(data);
    } catch (err) {
      showToast(`Query failed: ${err.message}`);
    } finally {
      setBusy(button, false);
    }
  });

  $("#askInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#askSubmit").click();
  });

  function renderAskResult(data) {
    $("#askEmpty").hidden = true;
    const result = $("#askResult");
    result.hidden = false;

    $("#answerText").textContent = data.answer;

    const stats = [
      ["retrieval", data.latency.retrieval_ms],
      ["rerank", data.latency.rerank_ms],
      ["generation", data.latency.generation_ms],
      ["total", data.latency.total_ms],
    ];
    $("#latencyStats").innerHTML = stats
      .map(
        ([label, value]) => `
        <div class="stat-tile">
          <div class="stat-value">${Math.round(value)}ms</div>
          <div class="stat-label">${label}</div>
        </div>`
      )
      .join("");

    const citationsEl = $("#citationChips");
    citationsEl.innerHTML = data.citations.length
      ? data.citations.map((c) => `<span class="chip">${c.chunk_id}</span>`).join("")
      : `<span class="muted">No citations for this answer.</span>`;

    const chunksEl = $("#retrievedChunks");
    chunksEl.innerHTML = data.retrieved_chunks
      .map((chunk) => `<div class="chunk-item">[${chunk.chunk_id}] (${chunk.source}, score=${chunk.score.toFixed(2)})\n${escapeHtml(chunk.text)}</div>`)
      .join("");
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ---------- Resolve tab (chat) ----------
  let threadId = null;
  const chatWindow = $("#chatWindow");

  function scrollChatToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function clearEmptyState() {
    const empty = $("#chatEmptyState");
    if (empty) empty.remove();
  }

  function appendUserMessage(text) {
    clearEmptyState();
    const div = document.createElement("div");
    div.className = "msg msg-user";
    div.textContent = text;
    chatWindow.appendChild(div);
    scrollChatToBottom();
  }

  function appendAgentMessage(text, { escalated = false } = {}) {
    clearEmptyState();
    const div = document.createElement("div");
    div.className = "msg msg-agent" + (escalated ? " is-escalated" : "");
    div.textContent = text;
    chatWindow.appendChild(div);
    scrollChatToBottom();
  }

  function appendTypingIndicator() {
    clearEmptyState();
    const div = document.createElement("div");
    div.className = "typing-dots";
    div.id = "typingIndicator";
    div.innerHTML = "<span></span><span></span><span></span>";
    chatWindow.appendChild(div);
    scrollChatToBottom();
    return div;
  }

  function appendApprovalCard(pending, replyText) {
    clearEmptyState();
    const wrap = document.createElement("div");
    wrap.className = "msg-approval";
    wrap.innerHTML = `
      <div class="approval-title">Approval needed &mdash; order #${pending.order_id}</div>
      <div class="approval-body">${escapeHtml(replyText)}</div>
      <div class="approval-actions">
        <button class="btn btn-approve btn-sm" data-approval="${pending.approval_id}" data-decision="approve">Approve</button>
        <button class="btn btn-reject btn-sm" data-approval="${pending.approval_id}" data-decision="reject">Reject</button>
      </div>
    `;
    chatWindow.appendChild(wrap);
    scrollChatToBottom();

    wrap.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", async () => {
        wrap.querySelectorAll("button").forEach((b) => (b.disabled = true));
        await decideApproval(pending.approval_id, btn.dataset.decision === "approve", wrap);
      });
    });
  }

  $("#chatForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#chatInput");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    appendUserMessage(message);

    const sendBtn = $("#chatSend");
    setBusy(sendBtn, true);
    const typing = appendTypingIndicator();

    try {
      const data = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: threadId }),
      });
      threadId = data.thread_id;
      typing.remove();

      if (data.pending_approval) {
        appendApprovalCard(data.pending_approval, data.reply);
      } else {
        appendAgentMessage(data.reply, { escalated: data.escalated });
      }
    } catch (err) {
      typing.remove();
      appendAgentMessage(`Something went wrong: ${err.message}`, { escalated: true });
    } finally {
      setBusy(sendBtn, false);
      input.focus();
    }
  });

  $("#newConversation").addEventListener("click", () => {
    threadId = null;
    chatWindow.innerHTML = `
      <div class="chat-empty" id="chatEmptyState">
        <div class="empty-glyph">💬</div>
        <p>Try: <em>"Refund order 1, it arrived damaged"</em> or <em>"Cancel order 5"</em></p>
      </div>`;
  });

  // ---------- Approvals panel ----------
  async function refreshApprovals() {
    try {
      const approvals = await api("/approvals");
      renderApprovals(approvals);
    } catch (err) {
      showToast(`Could not load approvals: ${err.message}`);
    }
  }

  function renderApprovals(approvals) {
    const list = $("#approvalsList");
    if (!approvals.length) {
      list.innerHTML = `<p class="muted approvals-empty">No pending approvals.</p>`;
      return;
    }
    list.innerHTML = approvals
      .map(
        (a) => `
        <div class="approval-item" data-id="${a.id}">
          <div class="row">
            <span class="order-id">Order #${a.order_id}</span>
            <span class="amount">$${a.amount.toFixed(2)}</span>
          </div>
          <div class="meta">expires ${new Date(a.expires_at).toLocaleString()}</div>
          <div class="actions">
            <button class="btn btn-approve btn-sm" data-decision="approve">Approve</button>
            <button class="btn btn-reject btn-sm" data-decision="reject">Reject</button>
          </div>
        </div>`
      )
      .join("");

    $$(".approval-item", list).forEach((item) => {
      const id = item.dataset.id;
      $$(".btn", item).forEach((btn) => {
        btn.addEventListener("click", async () => {
          $$(".btn", item).forEach((b) => (b.disabled = true));
          await decideApproval(id, btn.dataset.decision === "approve", item);
        });
      });
    });
  }

  async function decideApproval(id, approved, elementToRemove) {
    try {
      const result = await api(`/approvals/${id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved }),
      });
      showToast(approved ? "Refund approved" : "Refund rejected");
      if (elementToRemove) {
        elementToRemove.style.opacity = "0.5";
      }
      appendAgentMessage(result.reply);
      refreshApprovals();
    } catch (err) {
      showToast(`Decision failed: ${err.message}`);
    }
  }

  $("#refreshApprovals").addEventListener("click", refreshApprovals);
  refreshApprovals();
  setInterval(refreshApprovals, 20000);
})();
