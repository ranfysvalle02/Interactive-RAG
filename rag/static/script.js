// ---------------------------
// Global state and references
// ---------------------------
let currentSessionId = "default";
let allSessions = [];
let availableModels = [];

// Dom references
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const chatForm = document.getElementById("chat-form");
const sessionSelector = document.getElementById("session-selector");
const newSessionBtn = document.getElementById("new-session-btn");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const toolButtonsContainer = document.getElementById("tool-buttons");

const embeddingModelSelector = document.getElementById("embedding-model-selector");
const numSourcesInput = document.getElementById("num-sources-input");
const minScoreInput = document.getElementById("min-score-input");
const minScoreValue = document.getElementById("min-score-value");
const maxCharsInput = document.getElementById("max-chunk-length-input");
const maxCharsValue = document.getElementById("max-chars-value");

const previewRagBtn = document.getElementById("preview-rag-btn");

// Modal references
const modalOverlay = document.getElementById("modal-overlay");
const modalContainer = document.getElementById("modal-container");
const modalTitle = document.getElementById("modal-title");
const modalText = document.getElementById("modal-text");
const modalContentHost = document.getElementById("modal-content-host");
const modalCancelBtn = document.getElementById("modal-btn-cancel");
const modalSubmitBtn = document.getElementById("modal-btn-submit");

// Source browser references
const sourceBrowserOverlay = document.getElementById("source-browser-overlay");
const sourceBrowserContainer = document.getElementById("source-browser-container");
const sourceBrowserCloseBtn = document.getElementById("source-browser-close-btn");
const sourceListEl = document.getElementById("source-list");
const chunkListEl = document.getElementById("chunk-list");
const chunkListPlaceholder = document.getElementById("chunk-list-placeholder");

// -----------
// Modal Logic
// -----------
function showModal({ title, text, contentHTML, onSubmit, onCancel }) {
  modalTitle.textContent = title || "Modal Title";
  modalText.textContent = text || "";
  modalContentHost.innerHTML = contentHTML || "";
  if (onSubmit) {
    modalSubmitBtn.onclick = () => {
      onSubmit();
    };
  } else {
    modalSubmitBtn.onclick = () => {
      hideModal();
    };
  }
  if (onCancel) {
    modalCancelBtn.onclick = () => {
      onCancel();
    };
  } else {
    modalCancelBtn.onclick = () => {
      hideModal();
    };
  }
  modalOverlay.classList.remove("invisible", "opacity-0");
  modalContainer.classList.remove("scale-95", "opacity-0");
}

function hideModal() {
  modalOverlay.classList.add("opacity-0", "invisible");
  modalContainer.classList.add("scale-95", "opacity-0");
  // Reset content
  modalTitle.textContent = "";
  modalText.textContent = "";
  modalContentHost.innerHTML = "";
  modalSubmitBtn.onclick = null;
  modalCancelBtn.onclick = null;
}

modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) {
    hideModal();
  }
});

// ------------------------
// Source Browser Functions
// ------------------------
function openSourceBrowser() {
  // Show overlay
  sourceBrowserOverlay.classList.remove("opacity-0", "invisible");
  sourceBrowserContainer.classList.remove("scale-95", "opacity-0");

  // Load sources
  fetch(`/sources?session_id=${encodeURIComponent(currentSessionId)}`)
    .then((r) => r.json())
    .then((data) => {
      sourceListEl.innerHTML = "";
      chunkListEl.innerHTML = "";
      chunkListPlaceholder.style.display = "block";
      if (!data || data.length === 0) {
        sourceListEl.innerHTML = "<p class='text-gray-400 text-sm'>No sources found.</p>";
        return;
      }
      data.forEach((src) => {
        const btn = document.createElement("button");
        btn.className = "w-full text-left hover:bg-gray-700 text-gray-100 px-2 py-1 rounded";
        btn.textContent = src.name + (src.type ? ` (${src.type})` : "");
        btn.onclick = () => loadChunksForSource(src.name);
        sourceListEl.appendChild(btn);
      });
    })
    .catch((err) => {
      console.error("Failed to list sources:", err);
      sourceListEl.innerHTML = `<p class='text-red-500'>Error: ${err.message}</p>`;
    });
}

function loadChunksForSource(sourceUrl) {
  fetch(`/chunks?session_id=${encodeURIComponent(currentSessionId)}&source_url=` + encodeURIComponent(sourceUrl))
    .then((r) => r.json())
    .then((data) => {
      chunkListEl.innerHTML = "";
      if (data.error) {
        chunkListEl.innerHTML = `<p class='text-red-500'>Error: ${data.error}</p>`;
        return;
      }
      if (!data || data.length === 0) {
        chunkListEl.innerHTML = "<p class='text-gray-400 text-sm'>No chunks found for this source.</p>";
        return;
      }
      chunkListPlaceholder.style.display = "none";
      data.forEach((ch) => {
        const card = document.createElement("div");
        card.className = "chunk-card";
        card.innerHTML = `
          <div class="chunk-header">
            <div class="chunk-title">Chunk ID: ${ch._id}</div>
            <div class="flex gap-2">
              <button data-id="${ch._id}" class="chunk-edit-btn text-xs bg-blue-500 hover:bg-blue-600 px-2 py-1 rounded">Edit</button>
              <button data-id="${ch._id}" class="chunk-delete-btn text-xs bg-red-600 hover:bg-red-700 px-2 py-1 rounded">Delete</button>
            </div>
          </div>
          <div class="chunk-content">${escapeHtml(ch.text)}</div>
        `;
        chunkListEl.appendChild(card);
      });
      // Attach events
      chunkListEl.querySelectorAll(".chunk-edit-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          onEditChunkClick(btn.getAttribute("data-id"));
        });
      });
      chunkListEl.querySelectorAll(".chunk-delete-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          onDeleteChunkClick(btn.getAttribute("data-id"));
        });
      });
    })
    .catch((err) => {
      console.error("Failed to load chunks:", err);
      chunkListEl.innerHTML = `<p class='text-red-500'>Error: ${err.message}</p>`;
    });
}

sourceBrowserCloseBtn.addEventListener("click", () => {
  closeSourceBrowser();
});

function closeSourceBrowser() {
  sourceBrowserOverlay.classList.add("opacity-0", "invisible");
  sourceBrowserContainer.classList.add("scale-95", "opacity-0");
}

// Edit chunk
function onEditChunkClick(chunkId) {
  // Find chunk text in UI
  const chunkCard = chunkListEl.querySelector(`.chunk-header button[data-id='${chunkId}']`)?.closest(".chunk-card");
  if (!chunkCard) return;
  const oldText = chunkCard.querySelector(".chunk-content").textContent;

  showModal({
    title: "Edit Chunk",
    text: `Chunk ID: ${chunkId}`,
    contentHTML: `
      <textarea id="edit-chunk-textarea" class="w-full h-32 bg-gray-700 text-gray-200 p-2 rounded">${escapeHtmlForTextarea(oldText)}</textarea>
    `,
    onSubmit: () => {
      const newText = document.getElementById("edit-chunk-textarea").value;
      fetch("/chunk/" + encodeURIComponent(chunkId), {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ content: newText }),
      })
        .then((r) => r.json())
        .then((resp) => {
          if (resp.error) {
            alert("Error updating chunk: " + resp.error);
            return;
          }
          // Update UI
          chunkCard.querySelector(".chunk-content").textContent = newText;
          hideModal();
        })
        .catch((err) => {
          alert("Error updating chunk: " + err.message);
        });
    },
  });
}

// Delete chunk
function onDeleteChunkClick(chunkId) {
  if (!confirm("Are you sure you want to delete this chunk?")) return;
  fetch(`/chunk/${encodeURIComponent(chunkId)}`, { method: "DELETE" })
    .then((r) => r.json())
    .then((resp) => {
      if (resp.error) {
        alert("Error deleting chunk: " + resp.error);
        return;
      }
      // remove chunk card
      const chunkCard = chunkListEl.querySelector(`.chunk-header button[data-id='${chunkId}']`)?.closest(".chunk-card");
      if (chunkCard) {
        chunkCard.remove();
      }
    })
    .catch((err) => {
      alert("Error deleting chunk: " + err.message);
    });
}

// --------
// Helpers
// --------
function escapeHtml(unsafe) {
  if (!unsafe) return "";
  return unsafe.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function escapeHtmlForTextarea(str) {
  return str.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

// ---------------------------------------------
// Chat Rendering (add messages to the chat box)
// ---------------------------------------------
function addBotMessage(content) {
  const div = document.createElement("div");
  div.className = "message bot-message bg-gray-700 p-3 rounded-lg animate-fade-in-up";
  div.innerHTML = marked.parse(content || "");
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addUserMessage(content) {
  const div = document.createElement("div");
  div.className = "message user-message bg-gray-600 p-3 rounded-lg animate-fade-in-up text-right";
  div.textContent = content;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addSystemMessage(content) {
  const div = document.createElement("div");
  div.className = "message system-message bg-yellow-900/50 text-yellow-300 border-l-4 border-yellow-500 p-3 rounded-r-lg animate-fade-in-up";
  div.innerHTML = `<strong>System:</strong> ${content}`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// -------------------------
// Session / State Functions
// -------------------------
function loadSessionsAndState() {
  fetch("/state")
    .then((r) => r.json())
    .then((data) => {
      allSessions = data.all_sessions || [];
      availableModels = data.available_embedding_models || [];
      currentSessionId = data.current_session || "default";

      // Populate session selector
      sessionSelector.innerHTML = "";
      allSessions.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        if (s === currentSessionId) {
          opt.selected = true;
        }
        sessionSelector.appendChild(opt);
      });

      // Populate embedding-model-selector
      embeddingModelSelector.innerHTML = "";
      availableModels.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        embeddingModelSelector.appendChild(opt);
      });
    })
    .catch((err) => {
      console.error("Failed to load state:", err);
    });
}

function switchSession(sessionId) {
  // We'll accomplish session switch by issuing a chat request "switch_session sessionId"
  // so the server calls the corresponding tool.
  // Then we reload state.
  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: `switch_session ${sessionId}`,
      session_id: currentSessionId,
    }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        console.error("Error switching session:", data.error);
      } else {
        // The server may have updated the current session to sessionId.
        // Also the "session_update" in the response might come back. Let's re-load /state.
        loadSessionsAndState();
        addSystemMessage(`Switched to session: ${sessionId}`);
      }
    })
    .catch((err) => console.error("Failed to switch session:", err));
}

function createSession(newSessionName) {
  // We'll do it by a chat command as well: "create_session newSessionName"
  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: `create_session ${newSessionName}`,
      session_id: currentSessionId,
    }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        console.error("Error creating session:", data.error);
        alert("Error creating session: " + data.error);
      } else {
        // reload state
        loadSessionsAndState();
        addSystemMessage(`Created new session: ${newSessionName}`);
      }
    })
    .catch((err) => console.error("Failed to create session:", err));
}

// ------
// Events
// ------
// On DOM load:
document.addEventListener("DOMContentLoaded", () => {
  loadSessionsAndState();
});

// session selector
sessionSelector.addEventListener("change", () => {
  const sel = sessionSelector.value;
  if (sel !== currentSessionId) {
    switchSession(sel);
  }
});

// new session button
newSessionBtn.addEventListener("click", () => {
  const name = prompt("Enter new session name:");
  if (name) {
    createSession(name.trim());
  }
});

// clear history
clearHistoryBtn.addEventListener("click", () => {
  if (!confirm("Clear chat history for this session?")) return;
  fetch("/history/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: currentSessionId }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        console.error("Error clearing history:", data.error);
      } else {
        addSystemMessage(data.message || "Chat history cleared");
        // also clear the chat box
        chatBox.innerHTML = "";
        // re-insert the welcome system message
        const welcomeDiv = document.createElement("div");
        welcomeDiv.className = "message system-message animate-fade-in-up bg-yellow-900/50 text-yellow-300 border-l-4 border-yellow-500 p-4 rounded-r-lg";
        welcomeDiv.innerHTML = "<b>Welcome!</b> Use the Control Panel on the right to manage sessions, add data, and fine-tune retrieval settings.";
        chatBox.appendChild(welcomeDiv);
      }
    })
    .catch((err) => console.error("Failed to clear history:", err));
});

// main chat form
chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  addUserMessage(text);

  // read settings
  const embeddingModel = embeddingModelSelector.value || "openai";
  const numSources = parseInt(numSourcesInput.value) || 3;
  const maxChunkLen = parseInt(maxCharsInput.value) || 2000;

  // Build request
  const payload = {
    query: text,
    session_id: currentSessionId,
    embedding_model: embeddingModel,
    rag_params: {
      num_sources: numSources,
      max_chunk_length: maxChunkLen,
    },
  };

  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        addBotMessage("Error: " + data.error);
        return;
      }
      // display messages
      const msgs = data.messages || [];
      msgs.forEach((m) => {
        if (m.type === "bot-message") {
          addBotMessage(m.content);
        } else if (m.type === "system-message") {
          addSystemMessage(m.content);
        }
      });
      // update session list if changed
      if (data.session_update) {
        allSessions = data.session_update.all_sessions || allSessions;
        currentSessionId = data.session_update.current_session || currentSessionId;
        loadSessionsAndState();
      }
    })
    .catch((err) => {
      addBotMessage("Error: " + err.message);
    })
    .finally(() => {
      userInput.value = "";
      userInput.focus();
    });
});

// tool buttons
toolButtonsContainer.addEventListener("click", (event) => {
  const btn = event.target.closest("button[data-action]");
  if (!btn) return;
  const action = btn.getAttribute("data-action");
  handleToolAction(action);
});

function handleToolAction(action) {
  if (action === "read_url") {
    handleReadUrl();
  } else if (action === "read_file") {
    handleReadFile();
  } else if (action === "browse_sources") {
    openSourceBrowser();
  } else if (action === "search_web") {
    handleWebSearch();
  } else if (action === "list_sources") {
    // just do a chat: "list_sources"
    addUserMessage("list_sources");
    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: "list_sources",
        session_id: currentSessionId,
      }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          addBotMessage("Error: " + data.error);
          return;
        }
        (data.messages || []).forEach((m) => {
          if (m.type === "bot-message") {
            addBotMessage(m.content);
          }
        });
      })
      .catch((err) => {
        addBotMessage("Error: " + err.message);
      });
  } else if (action === "remove_all") {
    // chat: "remove_all_sources"
    if (!confirm("Are you sure you want to remove all sources in this session?")) return;
    addUserMessage("remove_all_sources");
    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: "remove_all_sources",
        session_id: currentSessionId,
      }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          addBotMessage("Error: " + data.error);
          return;
        }
        (data.messages || []).forEach((m) => {
          if (m.type === "bot-message") {
            addBotMessage(m.content);
          }
        });
      })
      .catch((err) => {
        addBotMessage("Error: " + err.message);
      });
  }
}

function handleReadUrl() {
  // Show modal to ask for URL, or do it directly with prompt
  const url = prompt("Enter the publicly accessible URL to ingest:");
  if (!url) return;
  // We can optionally preview first, or ingest directly:
  ingestUrl(url);
}

function ingestUrl(url) {
  // We'll do /preview_url if we want to show user something, but let's do ingestion directly.
  const body = {
    session_id: currentSessionId,
    content: "", // not used for URL
    source: url,
    source_type: "url",
  };
  // Actually, the back-end read_url is a tool, but we also have a direct ingestion route.
  // We'll call the agent with "read_url <url>", or we can do the /chat approach:
  addUserMessage(`read_url ${url}`);
  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: `read_url ${url}`,
      session_id: currentSessionId,
    }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        addBotMessage("Error: " + data.error);
        return;
      }
      (data.messages || []).forEach((m) => {
        if (m.type === "bot-message") {
          addBotMessage(m.content);
        }
      });
    })
    .catch((err) => {
      addBotMessage("Error: " + err.message);
    });
}

function handleReadFile() {
  // Show a modal that allows user to pick local file, preview, and ingest
  showModal({
    title: "Add File",
    text: "Select a file to ingest into the knowledge base",
    contentHTML: `
      <input type="file" id="file-input" class="block mb-2 text-sm">
      <div id="file-preview" class="text-xs text-gray-300 whitespace-pre h-32 overflow-auto border border-gray-700 rounded p-2"></div>
    `,
    onSubmit: () => {
      // Actually ingest
      const fileEl = document.getElementById("file-input");
      if (!fileEl.files || fileEl.files.length === 0) {
        alert("No file selected");
        return;
      }
      const file = fileEl.files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        // send ingestion
        const content = e.target.result;
        fetch("/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content: content,
            source: file.name,
            source_type: "file",
            session_id: currentSessionId,
          }),
        })
          .then((r) => r.json())
          .then((resp) => {
            if (resp.error) {
              alert("Error ingesting file: " + resp.error);
              return;
            }
            if (resp.task_id) {
              hideModal();
              pollIngestionTask(resp.task_id);
            } else {
              alert("Unexpected response, no task_id");
            }
          })
          .catch((err) => {
            alert("Error ingesting file: " + err.message);
          });
      };
      reader.readAsText(file);
    },
  });

  const fileEl = document.getElementById("file-input");
  fileEl.addEventListener("change", (e) => {
    const previewEl = document.getElementById("file-preview");
    previewEl.textContent = "Loading preview...";
    const selectedFile = fileEl.files[0];
    if (!selectedFile) {
      previewEl.textContent = "No file selected";
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);
    fetch("/preview_file", {
      method: "POST",
      body: formData,
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          previewEl.textContent = "Error previewing file: " + data.error;
          return;
        }
        previewEl.textContent = data.content;
      })
      .catch((err) => {
        previewEl.textContent = "Failed to preview file: " + err.message;
      });
  });
}

// Poll ingestion tasks
function pollIngestionTask(taskId) {
  const intervalId = setInterval(() => {
    fetch("/ingest/status/" + taskId)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "not_found") {
          clearInterval(intervalId);
          addBotMessage("Error: Task not found on server");
        } else if (data.status === "failed") {
          clearInterval(intervalId);
          addBotMessage("Ingestion failed: " + data.message);
        } else if (data.status === "complete") {
          clearInterval(intervalId);
          addBotMessage("Ingestion complete: " + data.message);
        } else {
          // still processing...
          console.log("Ingestion task status:", data.step);
        }
      })
      .catch((err) => {
        clearInterval(intervalId);
        addBotMessage("Ingestion polling error: " + err.message);
      });
  }, 2000);
}

// search web
function handleWebSearch() {
  // Show a modal that asks for a query
  showModal({
    title: "Search the Web",
    text: "Enter your search query to do a DuckDuckGo-based web search:",
    contentHTML: `<input type="text" id="web-search-input" class="w-full bg-gray-700 p-2 rounded" placeholder="Search...">`,
    onSubmit: () => {
      const query = document.getElementById("web-search-input").value.trim();
      if (!query) {
        alert("No query provided");
        return;
      }
      fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, num_results: 5 }),
      })
        .then((r) => r.json())
        .then((results) => {
          if (results.error) {
            addBotMessage("Web search error: " + results.error);
          } else {
            let output = "Web Search Results:\n";
            results.forEach((r, i) => {
              output += `[${i + 1}] Title: ${r.title}\nURL: ${r.url}\nSnippet: ${r.body}\n\n`;
            });
            addBotMessage(output);
          }
          hideModal();
        })
        .catch((err) => {
          addBotMessage("Web search error: " + err.message);
          hideModal();
        });
    },
  });
}

// preview context
previewRagBtn.addEventListener("click", () => {
  const text = userInput.value.trim();
  if (!text) {
    alert("Type your query in the box first.");
    return;
  }
  const embeddingModel = embeddingModelSelector.value || "openai";
  const numSources = parseInt(numSourcesInput.value) || 3;
  fetch("/preview_search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: text,
      session_id: currentSessionId,
      embedding_model: embeddingModel,
      num_sources: numSources,
    }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        alert("Preview error: " + data.error);
        return;
      }
      if (!Array.isArray(data) || data.length === 0) {
        alert("No results. Possibly no relevant chunks.");
        return;
      }
      let previewContent = data
        .map((res, idx) => {
          return `(${idx + 1}) Score: ${res.score.toFixed(4)} | Source: ${res.source}\n${res.content}\n---\n`;
        })
        .join("");
      showModal({
        title: "RAG Context Preview",
        text: "Retrieved chunks for your current query:",
        contentHTML: `<pre class="text-xs whitespace-pre-wrap">${escapeHtml(previewContent)}</pre>`,
      });
    })
    .catch((err) => {
      alert("Preview request failed: " + err.message);
    });
});

// handle range inputs
minScoreInput.addEventListener("input", () => {
  minScoreValue.textContent = parseFloat(minScoreInput.value).toFixed(2);
});
maxCharsInput.addEventListener("input", () => {
  maxCharsValue.textContent = parseInt(maxCharsInput.value);
});