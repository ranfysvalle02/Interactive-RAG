// ---------------------------
// Global state and references
// ---------------------------
let currentSessionId = "default";
let allSessions = [];
let availableModels = [];
let chunkCache = new Map();

// Dom references
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const chatForm = document.getElementById("chat-form");
const sessionSelector = document.getElementById("session-selector");
const newSessionBtn = document.getElementById("new-session-btn");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const toolButtonsContainer = document.getElementById("tool-buttons");
const thinkingIndicator = document.getElementById("thinking-indicator");

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
const sourceBrowserTotalChunks = document.getElementById("source-browser-total-chunks");
const sourceBrowserSourceCount = document.getElementById("source-browser-source-count");
const sourceBrowserSelectedChunkCount = document.getElementById("source-browser-selected-chunk-count");

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
 sourceBrowserOverlay.classList.remove("opacity-0", "invisible");
 sourceBrowserContainer.classList.remove("scale-95", "opacity-0");
 sourceBrowserTotalChunks.textContent = "";
 sourceBrowserSourceCount.textContent = "Total: 0";
 sourceBrowserSelectedChunkCount.textContent = "Selected: 0";

 fetch(`/sources?session_id=${encodeURIComponent(currentSessionId)}`)
  .then((r) => r.json())
  .then((data) => {
   sourceListEl.innerHTML = "";
   chunkListEl.innerHTML = "";
   chunkListPlaceholder.style.display = "block";
  
   if (!data || data.length === 0) {
    sourceListEl.innerHTML = "<p class='text-gray-400 text-sm p-2'>No sources found.</p>";
    return;
   }

   sourceBrowserSourceCount.textContent = `Total: ${data.length}`;
   let totalChunks = 0;
   data.forEach(src => totalChunks += (src.chunk_count || 0));
   sourceBrowserTotalChunks.textContent = `(${totalChunks.toLocaleString()} Total Chunks)`;
  
   data.forEach((src) => {
    const btn = document.createElement("button");
    btn.className = "source-item";
   
    const chunkCount = src.chunk_count !== undefined ? `${src.chunk_count}` : '?';
    const sourceName = src.name + (src.type ? ` (${src.type})` : "");

    btn.innerHTML = `
     <span class="truncate pr-2" title="${escapeHtml(sourceName)}">${escapeHtml(sourceName)}</span>
     <span class="sb-count-badge flex-shrink-0">${chunkCount}</span>
    `;

    btn.onclick = () => {
     document.querySelectorAll('.source-item').forEach(b => b.classList.remove('active'));
     btn.classList.add('active');
     loadChunksForSource(src.name);
    };
    sourceListEl.appendChild(btn);
   });
  })
  .catch((err) => {
   console.error("Failed to list sources:", err);
   sourceListEl.innerHTML = `<p class='text-red-500 p-2'>Error: ${err.message}</p>`;
  });
}

function loadChunksForSource(sourceUrl) {
  sourceBrowserSelectedChunkCount.textContent = "Loading...";
  fetch(`/chunks?session_id=${encodeURIComponent(currentSessionId)}&source_url=` + encodeURIComponent(sourceUrl))
   .then((r) => r.json())
   .then((data) => {
    chunkListEl.innerHTML = "";
    chunkCache.clear();

    if (data.error) {
     chunkListEl.innerHTML = `<p class='text-red-500'>Error: ${data.error}</p>`;
     sourceBrowserSelectedChunkCount.textContent = "Error";
     return;
    }
    if (!data || data.length === 0) {
     chunkListEl.innerHTML = "<p class='text-gray-400 text-sm'>No chunks found for this source.</p>";
     sourceBrowserSelectedChunkCount.textContent = "Selected: 0";
     return;
    }

    sourceBrowserSelectedChunkCount.textContent = `Selected: ${data.length}`;

    chunkListPlaceholder.style.display = "none";
    data.forEach((ch) => {
     chunkCache.set(ch._id, ch);
     const card = document.createElement("div");
     card.className = "chunk-card";
     card.setAttribute("data-chunk-id", ch._id); // Add ID for easier selection
     card.innerHTML = `
       <div class="chunk-header">
         <div class="chunk-title">Chunk ID: ${ch._id}</div>
         <div class="chunk-actions flex gap-2">
           <button data-id="${ch._id}" class="chunk-edit-btn text-xs bg-blue-500 hover:bg-blue-600 px-2 py-1 rounded">Edit</button>
           <button data-id="${ch._id}" class="chunk-delete-btn text-xs bg-red-600 hover:bg-red-700 px-2 py-1 rounded">Delete</button>
         </div>
       </div>
       <div class="chunk-content prose prose-invert max-w-none prose-sm">${marked.parse(ch.text || "")}</div>
     `;
     chunkListEl.appendChild(card);
    });
   })
   .catch((err) => {
    console.error("Failed to load chunks:", err);
    chunkListEl.innerHTML = `<p class='text-red-500'>Error: ${err.message}</p>`;
    sourceBrowserSelectedChunkCount.textContent = "Error";
   });
}

// -----------------------------------------------------------
// CORRECTED: A single, robust event listener for all chunk buttons
// (Handles Edit, Delete, Save, and Cancel)
// -----------------------------------------------------------
chunkListEl.addEventListener('click', (event) => {
    const editButton = event.target.closest('.chunk-edit-btn');
    if (editButton) {
        const chunkId = editButton.getAttribute('data-id');
        startChunkEdit(chunkId);
        return;
    }

    const deleteButton = event.target.closest('.chunk-delete-btn');
    if (deleteButton) {
        const chunkId = deleteButton.getAttribute('data-id');
        onDeleteChunkClick(chunkId); // Uses existing delete logic
        return;
    }

    const saveButton = event.target.closest('.chunk-save-btn');
    if (saveButton) {
        const chunkId = saveButton.getAttribute('data-id');
        saveChunkEdit(chunkId);
        return;
    }

    const cancelButton = event.target.closest('.chunk-cancel-btn');
    if (cancelButton) {
        const chunkId = cancelButton.getAttribute('data-id');
        cancelChunkEdit(chunkId);
        return;
    }
});


sourceBrowserCloseBtn.addEventListener("click", () => {
 closeSourceBrowser();
});

function closeSourceBrowser() {
 sourceBrowserOverlay.classList.add("opacity-0", "invisible");
 sourceBrowserContainer.classList.add("scale-95", "opacity-0");
 sourceListEl.innerHTML = "";
 chunkListEl.innerHTML = "";
 chunkListPlaceholder.style.display = "block";
 sourceBrowserTotalChunks.textContent = "";
 sourceBrowserSourceCount.textContent = "Total: 0";
 sourceBrowserSelectedChunkCount.textContent = "Selected: 0";
}

// -------------------------------------------------
// --- NEW IN-PLACE CHUNK EDITING LOGIC ---
// -------------------------------------------------

function startChunkEdit(chunkId) {
    const chunkCard = chunkListEl.querySelector(`.chunk-card[data-chunk-id='${chunkId}']`);
    if (!chunkCard || chunkCard.classList.contains('is-editing')) return;

    const chunkData = chunkCache.get(chunkId);
    if (!chunkData) {
        alert("Error: Could not find chunk data to edit.");
        return;
    }

    chunkCard.classList.add('is-editing');
    const contentHost = chunkCard.querySelector('.chunk-content');
    const actionsHost = chunkCard.querySelector('.chunk-actions');

    // Store original HTML for cancellation
    chunkCard.dataset.originalContent = contentHost.innerHTML;
    chunkCard.dataset.originalActions = actionsHost.innerHTML;

    // Inject the textarea and new buttons
    contentHost.innerHTML = `
        <textarea class="chunk-edit-textarea w-full bg-gray-900 text-gray-200 p-2 rounded font-mono text-sm resize-y focus:outline-none focus:ring-2 focus:ring-mongodb-green-500">${escapeHtmlForTextarea(chunkData.text)}</textarea>
    `;
    actionsHost.innerHTML = `
        <button data-id="${chunkId}" class="chunk-cancel-btn text-xs bg-gray-600 hover:bg-gray-700 px-3 py-1 rounded">Cancel</button>
        <button data-id="${chunkId}" class="chunk-save-btn text-xs bg-green-600 hover:bg-green-700 px-3 py-1 rounded">Save</button>
    `;

    // Auto-resize and focus the textarea
    const textarea = contentHost.querySelector('textarea');
    const autoResize = () => {
        textarea.style.height = 'auto';
        textarea.style.height = (textarea.scrollHeight) + 'px';
    };
    textarea.addEventListener('input', autoResize);
    autoResize();
    textarea.focus();
}

function cancelChunkEdit(chunkId) {
    const chunkCard = chunkListEl.querySelector(`.chunk-card[data-chunk-id='${chunkId}']`);
    if (!chunkCard || !chunkCard.classList.contains('is-editing')) return;

    const contentHost = chunkCard.querySelector('.chunk-content');
    const actionsHost = chunkCard.querySelector('.chunk-actions');

    // Restore original content from dataset
    contentHost.innerHTML = chunkCard.dataset.originalContent;
    actionsHost.innerHTML = chunkCard.dataset.originalActions;

    chunkCard.classList.remove('is-editing');
    delete chunkCard.dataset.originalContent;
    delete chunkCard.dataset.originalActions;
}

function saveChunkEdit(chunkId) {
    const chunkCard = chunkListEl.querySelector(`.chunk-card[data-chunk-id='${chunkId}']`);
    if (!chunkCard) return;

    const textarea = chunkCard.querySelector('.chunk-edit-textarea');
    const newText = textarea.value;
    const saveBtn = chunkCard.querySelector('.chunk-save-btn');
    saveBtn.textContent = 'Saving...';
    saveBtn.disabled = true;

    fetch("/chunk/" + encodeURIComponent(chunkId), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: newText }),
    })
    .then(r => r.json())
    .then(resp => {
        if (resp.error) {
            alert("Error updating chunk: " + resp.error);
            saveBtn.textContent = 'Save';
            saveBtn.disabled = false; // Re-enable on failure
            return;
        }
        // Update local cache
        const chunkData = chunkCache.get(chunkId);
        chunkData.text = newText;
        chunkCache.set(chunkId, chunkData);

        // Restore view mode with the *new* content
        const contentHost = chunkCard.querySelector('.chunk-content');
        const actionsHost = chunkCard.querySelector('.chunk-actions');
        
        contentHost.innerHTML = marked.parse(newText);
        actionsHost.innerHTML = chunkCard.dataset.originalActions; // Restore original buttons

        chunkCard.classList.remove('is-editing');
        delete chunkCard.dataset.originalContent;
        delete chunkCard.dataset.originalActions;
    })
    .catch(err => {
        alert("Error updating chunk: " + err.message);
        saveBtn.textContent = 'Save';
        saveBtn.disabled = false;
    });
}

function onDeleteChunkClick(chunkId) {
 if (!confirm("Are you sure you want to delete this chunk?")) return;
 fetch(`/chunk/${encodeURIComponent(chunkId)}`, { method: "DELETE" })
  .then((r) => r.json())
  .then((resp) => {
   if (resp.error) {
    alert("Error deleting chunk: " + resp.error);
    return;
   }
   const chunkCard = chunkListEl.querySelector(`.chunk-card[data-chunk-id='${chunkId}']`);
   if (chunkCard) {
    chunkCard.remove();
    chunkCache.delete(chunkId);
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
// Store query info with messages for chunk inspection
let messageQueryMap = new Map();

function addBotMessage(message) {
 const content = message.content;
 const sources = message.sources || [];
 const query = message.query || null; // Store query if available

 const messageEl = document.createElement("div");
 messageEl.className = "message bot-message flex flex-col p-4 bg-gray-700 rounded-lg animate-fade-in-up";

 const contentDiv = document.createElement("div");
 contentDiv.className = "prose prose-invert max-w-none";

 if (content.trim().startsWith('<div')) {
  contentDiv.innerHTML = content;
 } else {
  contentDiv.innerHTML = marked.parse(content || "");
 }
 messageEl.appendChild(contentDiv);

 if (sources.length > 0) {
  // Store query info for this message
  let messageId = null;
  if (query) {
   messageId = `msg-${Date.now()}-${Math.random()}`;
   messageEl.setAttribute('data-message-id', messageId);
   messageQueryMap.set(messageId, query);
  }

  let sourceLinksHTML = sources.map(source => {
   const href = `/source_content?session_id=${encodeURIComponent(currentSessionId)}&source=${encodeURIComponent(source)}`;
   const target = `target="_blank" rel="noopener noreferrer"`;
  
   let displayName = source;
   try {
    if (source.startsWith('http')) displayName = new URL(source).hostname;
   } catch (e) { /* use original source name */ }

   return `
    <a href="${href}" ${target} title="View full source: ${escapeHtml(source)}">
     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="w-4 h-4">
      <path d="M8.75 3.75a.75.75 0 0 0-1.5 0v1.5h-1.5a.75.75 0 0 0 0 1.5h1.5v1.5a.75.75 0 0 0 1.5 0v-1.5h1.5a.75.75 0 0 0 0-1.5h-1.5v-1.5Z" />
      <path fill-rule="evenodd" d="M3 1.75C3 .784 3.784 0 4.75 0h6.5C12.216 0 13 .784 13 1.75v12.5A1.75 1.75 0 0 1 11.25 16h-6.5A1.75 1.75 0 0 1 3 14.25V1.75Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h6.5a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25h-6.5Z" clip-rule="evenodd" />
     </svg>
     <span>${escapeHtml(displayName)}</span>
    </a>
   `;
  }).join('');

  const sourcesContainer = document.createElement("div");
  sourcesContainer.className = "source-links mt-4 pt-4 border-t border-gray-600";
  
  const inspectButton = messageId ? `
    <button onclick="inspectRetrievedChunks('${messageId}')" 
            class="flex items-center gap-1 text-xs bg-mongodb-green-500/20 hover:bg-mongodb-green-500/30 text-mongodb-green-400 px-2 py-1 rounded transition-colors">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639l4.43-7.29a1.125 1.125 0 011.906 0l4.43 7.29c.356.586.356 1.35 0 1.936l-4.43 7.29a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
      </svg>
      Inspect Chunks
    </button>
  ` : '';
  
  sourcesContainer.innerHTML = `
    <div class="flex justify-between items-center mb-2">
      <h4 class="text-xs font-bold uppercase text-gray-400">Sources</h4>
      ${inspectButton}
    </div>
    <div class="flex flex-wrap gap-2">
      ${sourceLinksHTML}
    </div>
  `;
  messageEl.appendChild(sourcesContainer);
 }

 chatBox.appendChild(messageEl);
 chatBox.scrollTop = chatBox.scrollHeight;
}

function addUserMessage(content) {
 const messageEl = document.createElement("div");
 messageEl.className = "message user-message bg-gray-600 p-3 rounded-lg animate-fade-in-up text-right";
 messageEl.textContent = content;
 chatBox.appendChild(messageEl);
 chatBox.scrollTop = chatBox.scrollHeight;
}

function addSystemMessage(content) {
 const div = document.createElement("div");
 div.className = "message system-message bg-yellow-900/50 text-yellow-300 border-l-4 border-yellow-500 p-3 rounded-r-lg animate-fade-in-up";
 div.innerHTML = `<strong>System:</strong> ${content}`;
 chatBox.appendChild(div);
 chatBox.scrollTop = chatBox.scrollHeight;
}

function setThinking(isThinking) {
 const indicator = document.getElementById("thinking-indicator");
 if (isThinking) {
  indicator.classList.remove("invisible", "opacity-0");
  chatBox.scrollTop = chatBox.scrollHeight;
 } else {
  indicator.classList.add("invisible", "opacity-0");
 }
}

// -------------------------
// Session / State Functions
// -------------------------
let indexStatusCache = {};

function loadSessionsAndState() {
 fetch(`/state?session_id=${encodeURIComponent(currentSessionId)}`)
  .then((r) => r.json())
  .then((data) => {
   allSessions = data.all_sessions || [];
   availableModels = data.available_embedding_models || [];
   currentSessionId = data.current_session || "default";
   indexStatusCache = data.index_status || {};
  
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
  
   const selectedModel = embeddingModelSelector.value;
   embeddingModelSelector.innerHTML = "";
   availableModels.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    embeddingModelSelector.appendChild(opt);
   });
   if (selectedModel && availableModels.includes(selectedModel)) {
     embeddingModelSelector.value = selectedModel;
   }
   
   // Update index status indicator
   updateIndexStatusIndicator();
  })
  .catch((err) => {
   console.error("Failed to load state:", err);
  });
}

function updateIndexStatusIndicator() {
  const model = embeddingModelSelector.value;
  const status = indexStatusCache[model];
  
  // Remove existing indicator
  const existing = document.getElementById('index-status-indicator');
  if (existing) existing.remove();
  
  if (!status) {
    return;
  }
  
  const isReady = status.index_ready;
  const docCount = status.document_count || 0;
  const indexStatus = status.index_status || 'UNKNOWN';
  const indexQueryable = status.index_queryable || false;
  
  // Don't show indicator if no documents
  if (docCount === 0) {
    return;
  }
  
  // Create modern indicator with badge style
  const indicator = document.createElement('div');
  indicator.id = 'index-status-indicator';
  indicator.className = 'index-status-badge';
  
  let statusConfig = {};
  
  if (isReady && indexQueryable) {
    statusConfig = {
      variant: 'success',
      icon: '✓',
      text: `${docCount.toLocaleString()} docs indexed`,
      pulse: false
    };
  } else if (indexStatus === 'NOT_FOUND' && docCount > 0) {
    statusConfig = {
      variant: 'warning',
      icon: '⚠',
      text: `Creating index for ${docCount.toLocaleString()} docs...`,
      pulse: true
    };
    // Trigger index creation
    fetch(`/index_status?session_id=${encodeURIComponent(currentSessionId)}&embedding_model=${encodeURIComponent(model)}&auto_create=true`)
      .catch(err => console.error('Failed to trigger index creation:', err));
  } else if (indexStatus === 'CREATING') {
    statusConfig = {
      variant: 'info',
      icon: '⏳',
      text: `Creating index (${docCount.toLocaleString()} docs)...`,
      pulse: true
    };
  } else if (indexStatus === 'BUILDING' || indexStatus === 'PENDING') {
    statusConfig = {
      variant: 'info',
      icon: '⏳',
      text: `Index ${indexStatus === 'BUILDING' ? 'building' : 'pending'} (${docCount.toLocaleString()} docs)...`,
      pulse: true
    };
  } else if (indexStatus === 'STALE') {
    statusConfig = {
      variant: 'warning',
      icon: '🔄',
      text: `Index updating (${docCount.toLocaleString()} docs)...`,
      pulse: true
    };
  } else if (indexStatus === 'CREATION_FAILED' || indexStatus === 'FAILED') {
    statusConfig = {
      variant: 'error',
      icon: '❌',
      text: `Index failed (${docCount.toLocaleString()} docs)`,
      pulse: false
    };
  } else if (docCount > 0 && !indexQueryable) {
    statusConfig = {
      variant: 'warning',
      icon: '⏳',
      text: `Index ${indexStatus.toLowerCase()} (${docCount.toLocaleString()} docs)...`,
      pulse: true
    };
  } else {
    return;
  }
  
  indicator.innerHTML = `
    <div class="index-status-content ${statusConfig.pulse ? 'pulse-animation' : ''}">
      <span class="index-status-icon">${statusConfig.icon}</span>
      <span class="index-status-text">${statusConfig.text}</span>
      <button class="index-status-debug-btn" onclick="openDebugModal()" title="View debug info">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </button>
    </div>
  `;
  
  indicator.setAttribute('data-variant', statusConfig.variant);
  
  // Insert after embedding model selector
  const modelSelector = embeddingModelSelector;
  const parent = modelSelector.parentElement;
  if (parent) {
    parent.insertBefore(indicator, modelSelector.nextSibling);
  }
}

function checkIndexReady(embeddingModel, maxWait = 30000, pollInterval = 2000) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    
    const check = () => {
      // Auto-create index if missing
      fetch(`/index_status?session_id=${encodeURIComponent(currentSessionId)}&embedding_model=${encodeURIComponent(embeddingModel)}&auto_create=true`)
        .then(r => r.json())
        .then(data => {
          if (data.ready_for_search) {
            resolve(true);
            return;
          }
          
          // If index is being created, continue waiting
          if (data.index_status === 'CREATING' || data.index_status === 'BUILDING' || data.index_status === 'PENDING') {
            if (Date.now() - startTime > maxWait) {
              resolve(false);
              return;
            }
            setTimeout(check, pollInterval);
            return;
          }
          
          if (Date.now() - startTime > maxWait) {
            resolve(false);
            return;
          }
          
          setTimeout(check, pollInterval);
        })
        .catch(() => {
          // On error, assume not ready
          if (Date.now() - startTime > maxWait) {
            resolve(false);
          } else {
            setTimeout(check, pollInterval);
          }
        });
    };
    
    check();
  });
}

function switchSession(sessionId) {
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
    loadSessionsAndState();
    chatBox.innerHTML = '';
     const welcomeDiv = document.createElement("div");
     welcomeDiv.className = "message system-message animate-fade-in-up bg-yellow-900/50 text-yellow-300 border-l-4 border-yellow-500 p-4 rounded-r-lg";
     welcomeDiv.innerHTML = `<b>Switched to session: ${sessionId}</b>`;
     chatBox.appendChild(welcomeDiv);
   }
  })
  .catch((err) => console.error("Failed to switch session:", err));
}

function createSession(newSessionName) {
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
    addSystemMessage(`Created and switched to new session: ${newSessionName}`);
    loadSessionsAndState();
   }
  })
  .catch((err) => console.error("Failed to create session:", err));
}

// ------
// Events
// ------
let indexStatusRefreshInterval = null;

function startIndexStatusRefresh() {
  // Clear existing interval
  if (indexStatusRefreshInterval) {
    clearInterval(indexStatusRefreshInterval);
  }
  
  // Refresh index status every 3 seconds if there are documents but index isn't ready
  indexStatusRefreshInterval = setInterval(() => {
    const model = embeddingModelSelector.value;
    const status = indexStatusCache[model];
    
    // Refresh if:
    // 1. We have documents but index isn't ready
    // 2. Index is in a building/creating state (to show progress)
    // 3. Index status is unknown
    if (status && status.document_count > 0) {
      const needsRefresh = !status.index_ready || 
                          status.index_status === 'CREATING' || 
                          status.index_status === 'BUILDING' || 
                          status.index_status === 'PENDING' ||
                          status.index_status === 'STALE' ||
                          status.index_status === 'NOT_FOUND';
      
      if (needsRefresh) {
        loadSessionsAndState();
      }
    }
  }, 3000); // Check every 3 seconds for faster updates
}

document.addEventListener("DOMContentLoaded", () => {
 loadSessionsAndState();
 startIndexStatusRefresh();
 
 // Update index status when embedding model changes
 embeddingModelSelector.addEventListener("change", () => {
  // Refresh state to get latest status for new model
  loadSessionsAndState();
 });
});

sessionSelector.addEventListener("change", () => {
 const sel = sessionSelector.value;
 if (sel !== currentSessionId) {
  switchSession(sel);
  // Refresh status when session changes
  setTimeout(() => {
    loadSessionsAndState();
  }, 500);
 }
});

newSessionBtn.addEventListener("click", () => {
 const name = prompt("Enter new session name:");
 if (name) {
  createSession(name.trim());
 }
});

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
    chatBox.innerHTML = "";
    const welcomeDiv = document.createElement("div");
    welcomeDiv.className = "message system-message animate-fade-in-up bg-yellow-900/50 text-yellow-300 border-l-4 border-yellow-500 p-4 rounded-r-lg";
    welcomeDiv.innerHTML = "<b>Welcome!</b> Use the Control Panel on the right to manage sessions, add data, and fine-tune retrieval settings.";
    chatBox.appendChild(welcomeDiv);
   }
  })
  .catch((err) => console.error("Failed to clear history:", err));
});

chatForm.addEventListener("submit", async (event) => {
 event.preventDefault();
 const text = userInput.value.trim();
 if (!text) return;

 addUserMessage(text);
 setThinking(true);

 const embeddingModel = embeddingModelSelector.value;
 const numSources = parseInt(numSourcesInput.value) || 3;
 const maxChunkLen = parseInt(maxCharsInput.value) || 2000;

 // Check if index is ready, wait if needed
 const status = indexStatusCache[embeddingModel];
 if (status && !status.index_ready && status.document_count > 0) {
  addSystemMessage(`⏳ Waiting for index to be ready (${status.document_count} documents indexed)...`);
  const isReady = await checkIndexReady(embeddingModel, 30000, 2000);
  if (!isReady) {
   addSystemMessage(`⚠️ Index may still be building. Trying search anyway...`);
  } else {
   addSystemMessage(`✅ Index is ready!`);
   loadSessionsAndState(); // Refresh status
  }
 }

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
    addBotMessage({ content: `Error: ${data.error}` });
    return;
   }
   const msgs = data.messages || [];
   msgs.forEach((m) => {
    if (m.type === "bot-message") {
     addBotMessage(m);
    } else if (m.type === "system-message") {
     addSystemMessage(m.content);
    }
   });
   if (data.session_update) {
    loadSessionsAndState();
   }
  })
  .catch((err) => {
   addBotMessage({ content: `Error: ${err.message}` });
  })
  .finally(() => {
   setThinking(false);
   userInput.value = "";
   userInput.focus();
   userInput.style.height = 'auto';
  });
});

 userInput.addEventListener('input', () => {
   userInput.style.height = 'auto';
   userInput.style.height = (userInput.scrollHeight) + 'px';
 });

userInput.addEventListener("keydown", (event) => {
 if (event.key === "Enter" && !event.shiftKey) {
  event.preventDefault();
  chatForm.dispatchEvent(new Event('submit'));
 }
});

toolButtonsContainer.addEventListener("click", (event) => {
 const btn = event.target.closest("button[data-action]");
 if (!btn) return;
 const action = btn.getAttribute("data-action");
 handleToolAction(action);
});

function handleToolAction(action) {
 if (action === "read_url") {
  handleReadUrlAndChunking();
 } else if (action === "read_file") {
  handleReadFile();
 } else if (action === "browse_sources") {
  openSourceBrowser();
 } else if (action === "search_web") {
  handleWebSearch();
 } else if (action === "list_sources" || action === "remove_all") {
   const command = action === "list_sources" ? "list_sources" : "remove_all_sources";
   if (action === "remove_all" && !confirm("Are you sure you want to remove all sources in this session?")) {
     return;
   }

   addUserMessage(command);
   setThinking(true);
   fetch("/chat", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify({ query: command, session_id: currentSessionId }),
   })
   .then(r => r.json())
   .then(data => {
     if (data.error) {
       addBotMessage({ content: `Error: ${data.error}` });
     } else {
       (data.messages || []).forEach(m => {
         if (m.type === "bot-message" || m.type === "system-message") {
           addBotMessage(m);
         }
       });
     }
   })
   .catch(err => addBotMessage({ content: `Error: ${err.message}` }))
   .finally(() => setThinking(false));
 }
}

// ------------------------------------
// --- NEW INGESTION MODAL LOGIC ---
// ------------------------------------

async function renderChunkPreview(content, chunkSize, chunkOverlap, targetElementId, countElementId) {
  const targetEl = document.getElementById(targetElementId);
  const countEl = document.getElementById(countElementId);
  if (!targetEl || !countEl) return;

  targetEl.innerHTML = '<div class="flex justify-center items-center h-full"><div class="spinner-large"></div></div>';
  countEl.textContent = 'Total Chunks: ...';

  if (chunkOverlap >= chunkSize) {
    targetEl.innerHTML = '<p class="text-red-500 p-4 text-center">Error: Chunk overlap must be smaller than chunk size.</p>';
    countEl.textContent = 'Total Chunks: 0';
    return false;
  }

  try {
    const response = await fetch("/chunk_preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
    });
    const data = await response.json();

    if (data.error) {
      targetEl.innerHTML = `<p class="text-red-500 p-4 text-center">Error chunking: ${escapeHtml(data.error)}</p>`;
      countEl.textContent = 'Total Chunks: 0';
      return false;
    }

    if (!data.chunks || data.chunks.length === 0) {
      targetEl.innerHTML = '<p class="text-gray-400 p-4 text-center">Could not generate any chunks from the source content.</p>';
      countEl.textContent = 'Total Chunks: 0';
      return false;
    }

    const chunkHtml = data.chunks.map((c, i) => `
          <div class="chunk-card">
            <div class="chunk-header"><div class="chunk-title">Chunk ${i + 1}</div></div>
            <div class="chunk-content">${escapeHtml(c)}</div>
          </div>
    `).join('');
   
    targetEl.innerHTML = `<div class="chunk-list-container animate-fade-in-up">${chunkHtml}</div>`;
    countEl.textContent = `Total Chunks: ${data.chunks.length}`;
    return true;
  } catch (err) {
    targetEl.innerHTML = `<p class="text-red-500 p-4 text-center">Request error: ${escapeHtml(err.message)}</p>`;
    countEl.textContent = 'Total: 0';
    return false;
  }
}

function handleReadFile() {
  let sourceName = '';
  let currentFile = null;

  const modalHTML = `
    <div id="file-drop-zone" class="w-full p-6 border-2 border-dashed border-gray-600 rounded-lg text-center cursor-pointer hover:border-mongodb-green-500 transition-all duration-200">
      <input type="file" id="ingestion-file-input" class="hidden" />
      <div id="file-drop-zone-prompt" class="flex flex-col items-center justify-center text-gray-400">
         <svg xmlns="http://www.w3.org/2000/svg" class="w-12 h-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
           <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M12 15l-4-4m0 0l4-4m-4 4h12" />
         </svg>
        <p class="font-semibold text-gray-200">Drag & drop your file here</p>
        <p class="text-sm">or click to browse</p>
      </div>
      <div id="file-drop-zone-display" class="hidden flex-col items-center justify-center">
         <svg xmlns="http://www.w3.org/2000/svg" class="w-12 h-12 mb-2 text-mongodb-green-500" viewBox="0 0 20 20" fill="currentColor">
             <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clip-rule="evenodd" />
         </svg>
         <p id="file-name-display" class="font-semibold text-gray-200"></p>
         <p class="text-sm text-gray-400">Click again or drop another file to replace</p>
      </div>
    </div>
    <div class="flex gap-4 mt-4 h-[50vh]">
      <div class="w-1/2 flex flex-col bg-gray-900/50 rounded-lg">
        <div class="flex justify-between items-center border-b border-gray-700 p-3">
          <h4 class="font-bold text-mongodb-green-500">Source Content</h4>
          <button onclick="openSourceContentFromIngestion()" class="text-xs bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 px-2 py-1 rounded transition-colors flex items-center gap-1">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 002.25-2.25V6a2.25 2.25 0 00-2.25-2.25H6A2.25 2.25 0 003.75 6v2.25A2.25 2.25 0 006 10.5zm0 9.75h2.25A2.25 2.25 0 0010.5 18v-2.25a2.25 2.25 0 00-2.25-2.25H6a2.25 2.25 0 00-2.25 2.25V18A2.25 2.25 0 006 20.25zm9.75-9.75H18a2.25 2.25 0 002.25-2.25V6A2.25 2.25 0 0018 3.75h-2.25A2.25 2.25 0 0013.5 6v2.25A2.25 2.25 0 0015.75 10.5z" />
            </svg>
            Open in Modal
          </button>
        </div>
        <div class="flex-grow p-1">
         <textarea id="ingestion-source-content-textarea" class="w-full h-full bg-transparent text-gray-200 p-2 rounded-md resize-none focus:outline-none focus:ring-1 focus:ring-mongodb-green-500" placeholder="Select a file to begin..."></textarea>
        </div>
      </div>
      <div class="w-1/2 flex flex-col bg-gray-900/50 rounded-lg">
        <div class="flex justify-between items-center border-b border-gray-700 p-3">
          <h4 class="font-bold text-mongodb-green-500">Chunk Preview</h4>
          <span id="ingestion-chunk-count" class="text-sm font-mono bg-gray-700 text-mongodb-green-500 px-2 py-1 rounded">Total: 0</span>
        </div>
        <div id="ingestion-chunk-preview-host" class="flex-grow overflow-y-auto p-3">
          <p class="text-gray-400 text-center pt-10">Chunks will appear here.</p>
        </div>
      </div>
    </div>
    <div id="ingestion-controls" class="grid grid-cols-3 gap-4 text-sm p-4 border-t border-gray-700 mt-4 items-center">
      <div class="flex items-center gap-2">
        <label class="font-medium text-gray-300">Chunk Size:</label>
        <input type="number" id="ingestion-chunk-size" value="1000" min="100" step="100" class="w-24 bg-gray-700 border border-gray-600 rounded-md px-2 py-1 text-sm">
      </div>
      <div class="flex items-center gap-2">
        <label class="font-medium text-gray-300">Overlap:</label>
        <input type="number" id="ingestion-chunk-overlap" value="150" min="0" step="50" class="w-24 bg-gray-700 border border-gray-600 rounded-md px-2 py-1 text-sm">
      </div>
      <button id="ingestion-rechunk-btn" class="btn btn-secondary w-full">Update Chunk Preview</button>
    </div>`;

  showModal({
    title: "Add File to Knowledge Base",
    text: "Drop a file or click the area below, edit content if needed, adjust chunking, and submit to ingest.",
    contentHTML: modalHTML,
    onSubmit: () => {
      const content = document.getElementById('ingestion-source-content-textarea').value;
      if (!content || !sourceName) {
        alert('Please select and load a file first.');
        return;
      }
      const chunkSize = parseInt(document.getElementById('ingestion-chunk-size').value);
      const chunkOverlap = parseInt(document.getElementById('ingestion-chunk-overlap').value);

      if (chunkOverlap >= chunkSize) {
        alert("Chunk overlap must be less than chunk size.");
        return;
      }

      fetch("/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: content,
          source: sourceName,
          source_type: "file",
          session_id: currentSessionId,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
        }),
      }).then(r => r.json()).then(resp => {
        if (resp.error) {
          alert(`Error ingesting file: ${resp.error}`);
        } else if (resp.task_id) {
          hideModal();
          pollIngestionTask(resp.task_id);
        }
      }).catch(err => alert(`Error: ${err.message}`));
    }
  });

  const dropZone = document.getElementById('file-drop-zone');
  const fileInput = document.getElementById('ingestion-file-input');
  const dropZonePrompt = document.getElementById('file-drop-zone-prompt');
  const dropZoneDisplay = document.getElementById('file-drop-zone-display');
  const fileNameDisplay = document.getElementById('file-name-display');
  const contentTextarea = document.getElementById('ingestion-source-content-textarea');
  const rechunkBtn = document.getElementById('ingestion-rechunk-btn');

  const processFile = (file) => {
    if (!file) return;
    currentFile = file;
    
    fileNameDisplay.textContent = file.name;
    dropZonePrompt.classList.add('hidden');
    dropZoneDisplay.classList.remove('hidden');
    dropZoneDisplay.classList.add('flex');

    contentTextarea.value = 'Loading file content...';
    const formData = new FormData();
    formData.append('file', file);

    fetch('/preview_file', { method: 'POST', body: formData })
      .then(r => r.json()).then(data => {
        if (data.error) {
          contentTextarea.value = `Error: ${escapeHtml(data.error)}`;
          return;
        }
        sourceName = data.filename;
        contentTextarea.value = data.content;
       
        const chunkSize = parseInt(document.getElementById('ingestion-chunk-size').value);
        const chunkOverlap = parseInt(document.getElementById('ingestion-chunk-overlap').value);
        renderChunkPreview(data.content, chunkSize, chunkOverlap, 'ingestion-chunk-preview-host', 'ingestion-chunk-count');
      }).catch(err => {
        contentTextarea.value = `Fetch error: ${escapeHtml(err.message)}`;
      });
  }

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drop-zone-dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drop-zone-dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drop-zone-dragover');
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      processFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      processFile(fileInput.files[0]);
    }
  });

  rechunkBtn.addEventListener('click', () => {
    const content = contentTextarea.value;
    if (!content) { alert('Load a file first.'); return; }
    const chunkSize = parseInt(document.getElementById('ingestion-chunk-size').value);
    const chunkOverlap = parseInt(document.getElementById('ingestion-chunk-overlap').value);
    renderChunkPreview(content, chunkSize, chunkOverlap, 'ingestion-chunk-preview-host', 'ingestion-chunk-count');
    rechunkBtn.classList.remove('needs-update');
  });
 
  contentTextarea.addEventListener('input', () => rechunkBtn.classList.add('needs-update'));
  document.getElementById('ingestion-chunk-size').addEventListener('input', () => rechunkBtn.classList.add('needs-update'));
  document.getElementById('ingestion-chunk-overlap').addEventListener('input', () => rechunkBtn.classList.add('needs-update'));
}

function handleReadUrlAndChunking(initialUrl = '') {
  const modalHTML = `
    <div class="mb-4 flex gap-2">
      <input type="text" id="ingestion-url-input" value="${initialUrl}" placeholder="Enter URL..." class="flex-grow bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm focus:ring-2 focus:ring-mongodb-green-500 focus:outline-none">
      <button id="ingestion-load-url-btn" class="btn btn-primary">Load Content</button>
    </div>
    <div class="flex gap-4 mt-4 h-[55vh]">
      <div class="w-1/2 flex flex-col bg-gray-900/50 rounded-lg">
        <div class="flex justify-between items-center border-b border-gray-700 p-3">
          <h4 class="font-bold text-mongodb-green-500">Source Content</h4>
          <button onclick="openSourceContentFromIngestion()" class="text-xs bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 px-2 py-1 rounded transition-colors flex items-center gap-1">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 002.25-2.25V6a2.25 2.25 0 00-2.25-2.25H6A2.25 2.25 0 003.75 6v2.25A2.25 2.25 0 006 10.5zm0 9.75h2.25A2.25 2.25 0 0010.5 18v-2.25a2.25 2.25 0 00-2.25-2.25H6a2.25 2.25 0 00-2.25 2.25V18A2.25 2.25 0 006 20.25zm9.75-9.75H18a2.25 2.25 0 002.25-2.25V6A2.25 2.25 0 0018 3.75h-2.25A2.25 2.25 0 0013.5 6v2.25A2.25 2.25 0 0015.75 10.5z" />
            </svg>
            Open in Modal
          </button>
        </div>
        <div class="flex-grow p-1">
         <textarea id="ingestion-source-content-textarea" class="w-full h-full bg-transparent text-gray-200 p-2 rounded-md resize-none focus:outline-none focus:ring-1 focus:ring-mongodb-green-500" placeholder="Enter a URL and click 'Load Content'..."></textarea>
        </div>
      </div>
      <div class="w-1/2 flex flex-col bg-gray-900/50 rounded-lg">
        <div class="flex justify-between items-center border-b border-gray-700 p-3">
          <h4 class="font-bold text-mongodb-green-500">Chunk Preview</h4>
          <span id="ingestion-chunk-count" class="text-sm font-mono bg-gray-700 text-mongodb-green-500 px-2 py-1 rounded">Total: 0</span>
        </div>
        <div id="ingestion-chunk-preview-host" class="flex-grow overflow-y-auto p-3">
          <p class="text-gray-400 text-center pt-10">Chunks will appear here.</p>
        </div>
      </div>
    </div>
    <div id="ingestion-controls" class="grid grid-cols-3 gap-4 text-sm p-4 border-t border-gray-700 mt-4 items-center">
      <div class="flex items-center gap-2">
        <label class="font-medium text-gray-300">Chunk Size:</label>
        <input type="number" id="ingestion-chunk-size" value="1000" min="100" step="100" class="w-24 bg-gray-700 border border-gray-600 rounded-md px-2 py-1 text-sm">
      </div>
      <div class="flex items-center gap-2">
        <label class="font-medium text-gray-300">Overlap:</label>
        <input type="number" id="ingestion-chunk-overlap" value="150" min="0" step="50" class="w-24 bg-gray-700 border border-gray-600 rounded-md px-2 py-1 text-sm">
      </div>
      <button id="ingestion-rechunk-btn" class="btn btn-secondary w-full">Update Chunk Preview</button>
    </div>`;
 
  showModal({
    title: "Add URL to Knowledge Base",
    text: "Fetch content, edit if needed, adjust chunking, and submit to ingest.",
    contentHTML: modalHTML,
    onSubmit: () => {
      const url = document.getElementById('ingestion-url-input').value.trim();
      const content = document.getElementById('ingestion-source-content-textarea').value;
      if (!url || !content) {
        alert('Please load the URL content first.');
        return;
      }
      const chunkSize = parseInt(document.getElementById('ingestion-chunk-size').value);
      const chunkOverlap = parseInt(document.getElementById('ingestion-chunk-overlap').value);

      if (chunkOverlap >= chunkSize) {
        alert("Chunk overlap must be less than chunk size.");
        return;
      }
     
      fetch("/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: content,
          source: url,
          source_type: "url",
          session_id: currentSessionId,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
        }),
      }).then(r => r.json()).then(resp => {
        if (resp.error) {
          alert(`Error ingesting URL: ${resp.error}`);
        } else if (resp.task_id) {
          hideModal();
          pollIngestionTask(resp.task_id);
        }
      }).catch(err => alert(`Error: ${err.message}`));
    }
  });

  const urlInput = document.getElementById('ingestion-url-input');
  const loadBtn = document.getElementById('ingestion-load-url-btn');
  const contentTextarea = document.getElementById('ingestion-source-content-textarea');
  const rechunkBtn = document.getElementById('ingestion-rechunk-btn');

  const loadUrlContent = () => {
    const url = urlInput.value.trim();
    if (!url) return;
    contentTextarea.value = 'Loading URL content...';
   
    fetch(`/preview_url?url=${encodeURIComponent(url)}`)
      .then(r => r.json()).then(data => {
        if (data.error) {
          contentTextarea.value = `Error: ${escapeHtml(data.error)}`;
          return;
        }
        contentTextarea.value = data.markdown;
       
        const chunkSize = parseInt(document.getElementById('ingestion-chunk-size').value);
        const chunkOverlap = parseInt(document.getElementById('ingestion-chunk-overlap').value);
        renderChunkPreview(data.markdown, chunkSize, chunkOverlap, 'ingestion-chunk-preview-host', 'ingestion-chunk-count');
      }).catch(err => {
        contentTextarea.value = `Fetch error: ${escapeHtml(err.message)}`;
      });
  };

  loadBtn.addEventListener('click', loadUrlContent);
  rechunkBtn.addEventListener('click', () => {
    const content = contentTextarea.value;
    if (!content) { alert('Load URL content first.'); return; }
    const chunkSize = parseInt(document.getElementById('ingestion-chunk-size').value);
    const chunkOverlap = parseInt(document.getElementById('ingestion-chunk-overlap').value);
    renderChunkPreview(content, chunkSize, chunkOverlap, 'ingestion-chunk-preview-host', 'ingestion-chunk-count');
    rechunkBtn.classList.remove('needs-update');
  });
 
  contentTextarea.addEventListener('input', () => rechunkBtn.classList.add('needs-update'));
  document.getElementById('ingestion-chunk-size').addEventListener('input', () => rechunkBtn.classList.add('needs-update'));
  document.getElementById('ingestion-chunk-overlap').addEventListener('input', () => rechunkBtn.classList.add('needs-update'));

  if (initialUrl) {
    loadUrlContent();
  }
}

function handleWebSearch() {
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
   hideModal();
   addUserMessage(`web_search ${query}`);
   setThinking(true);
  
   fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, num_results: 5 }),
   })
    .then((r) => r.json())
    .then((data) => {
     if (data.error) {
      addBotMessage({ content: `Web search error: ${data.error}` });
     } else if (data.results && data.results.length > 0) {
      const resultsHtml = data.results.map((r) => {
       const isValidUrl = r.href && (r.href.startsWith('http://') || r.href.startsWith('https://'));
       const url = isValidUrl ? r.href : '#';
       let host = 'N/A';
       if (isValidUrl) {
        try {
         host = new URL(url).hostname;
        } catch (e) { console.error('Failed to parse URL', e); }
       }
      
       return `
          <div class="web-result-card animate-fade-in-up">
           <div class="flex justify-between items-start mb-2">
            <h4 class="text-white font-bold text-lg leading-tight">
             <a href="${url}" target="_blank" class="hover:underline">${escapeHtml(r.title)}</a>
            </h4>
            <button data-url="${url}" class="read-url-btn text-xs px-3 py-1 rounded-full transition-colors font-medium flex-shrink-0 flex items-center gap-1">
             <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-4 h-4">
              <path stroke-linecap="round" stroke-linejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
             </svg>
             Read & Ingest
            </button>
           </div>
           <p class="text-gray-300 text-sm mb-2">${escapeHtml(r.body)}</p>
           <a href="${url}" target="_blank" class="url-link hover:underline">${host}</a>
          </div>
         `;
      }).join('');

      addBotMessage({ content: `<div><p>Web Search Results:</p><div class="mt-4">${resultsHtml}</div></div>` });
     
      document.querySelectorAll('.read-url-btn').forEach(button => {
       button.addEventListener('click', (e) => {
        const url = e.target.closest('button').getAttribute('data-url');
        if (url && url !== '#') {
         handleReadUrlAndChunking(url);
        }
       });
      });

     } else {
      addBotMessage({ content: "No web search results found." });
     }
    })
    .catch((err) => {
     addBotMessage({ content: `Web search error: ${err.message}` });
    })
    .finally(() => {
     setThinking(false);
    });
  },
 });
}

previewRagBtn.addEventListener("click", () => {
 const text = userInput.value.trim();
 if (!text) {
  alert("Type your query in the box first.");
  return;
 }
 const embeddingModel = embeddingModelSelector.value;
 const numSources = parseInt(numSourcesInput.value) || 3;
 const minScore = parseFloat(minScoreInput.value) || 0;
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
    alert(`Preview error: ${data.error}`);
    return;
   }
   const filteredData = data.filter(res => res.score >= minScore);
   if (!Array.isArray(filteredData) || filteredData.length === 0) {
    alert("No results found for the given query and minimum score.");
    return;
   }
   let previewContent = filteredData
    .map((res, idx) => {
     return `(${idx + 1}) Score: ${res.score.toFixed(4)} | Source: ${res.source}\n${res.content}\n---\n`;
    })
    .join("");
   showModal({
    title: "RAG Context Preview",
    text: "Retrieved chunks for your current query:",
    contentHTML: `<pre class="text-xs whitespace-pre-wrap h-96 overflow-y-auto bg-gray-900 rounded-md p-4">${escapeHtml(previewContent)}</pre>`,
   });
  })
  .catch((err) => {
   alert(`Preview request failed: ${err.message}`);
  });
});

minScoreInput.addEventListener("input", () => {
 minScoreValue.textContent = parseFloat(minScoreInput.value).toFixed(2);
});
maxCharsInput.addEventListener("input", () => {
 maxCharsValue.textContent = parseInt(maxCharsInput.value);
});

// ---------------------------
// Ingestion Progress Overlay
// ---------------------------
let ingestionOverlay = null;
let ingestionStatusInterval = null;

function createIngestionOverlay() {
  if (ingestionOverlay) return ingestionOverlay;
  
  const overlay = document.createElement('div');
  overlay.id = 'ingestion-overlay';
  overlay.className = 'fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50';
  overlay.innerHTML = `
    <div class="bg-gray-800 rounded-lg shadow-xl w-full max-w-md p-6 border border-gray-700">
      <div class="flex items-center gap-3 mb-4">
        <div class="spinner-large"></div>
        <h3 class="text-xl font-bold text-white">Processing Ingestion</h3>
      </div>
      <div id="ingestion-status-text" class="text-gray-300 mb-4 min-h-[3rem]">
        Starting ingestion...
      </div>
      <div class="w-full bg-gray-700 rounded-full h-2 mb-2">
        <div id="ingestion-progress-bar" class="bg-mongodb-green-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
      </div>
      <p class="text-xs text-gray-400 text-center">Please wait while your content is being processed...</p>
    </div>
  `;
  document.body.appendChild(overlay);
  ingestionOverlay = overlay;
  return overlay;
}

function showIngestionOverlay() {
  const overlay = createIngestionOverlay();
  overlay.classList.remove('hidden');
  overlay.style.display = 'flex';
  
  // Disable all interactive elements
  document.querySelectorAll('button, input, select, textarea').forEach(el => {
    if (!el.closest('#ingestion-overlay')) {
      el.disabled = true;
      el.style.pointerEvents = 'none';
      el.style.opacity = '0.5';
    }
  });
}

function hideIngestionOverlay() {
  if (ingestionOverlay) {
    ingestionOverlay.classList.add('hidden');
    ingestionOverlay.style.display = 'none';
  }
  
  // Re-enable all interactive elements
  document.querySelectorAll('button, input, select, textarea').forEach(el => {
    el.disabled = false;
    el.style.pointerEvents = '';
    el.style.opacity = '';
  });
  
  if (ingestionStatusInterval) {
    clearInterval(ingestionStatusInterval);
    ingestionStatusInterval = null;
  }
}

function updateIngestionStatus(status, step, progress = 0) {
  const statusText = document.getElementById('ingestion-status-text');
  const progressBar = document.getElementById('ingestion-progress-bar');
  
  if (statusText) {
    const stepMessages = {
      'pending': 'Initializing...',
      'processing': step || 'Processing...',
      'complete': 'Complete!',
      'failed': 'Failed'
    };
    
    let message = stepMessages[status] || status;
    if (status === 'processing' && step) {
      message = step;
    } else if (status === 'complete') {
      message = '✅ Ingestion completed successfully!';
    } else if (status === 'failed') {
      message = `❌ Ingestion failed: ${step || 'Unknown error'}`;
    }
    
    statusText.textContent = message;
    
    // Add index status info if available
    if (status === 'complete') {
      setTimeout(() => {
        loadSessionsAndState();
        const model = embeddingModelSelector.value;
        const status = indexStatusCache[model];
        if (status && !status.index_ready) {
          statusText.innerHTML = `
            <div>${message}</div>
            <div class="text-xs text-yellow-400 mt-2">
              ⏳ Index is building... (${status.document_count} docs) - This may take 10-30 seconds
            </div>
          `;
        }
      }, 500);
    }
  }
  
  if (progressBar) {
    let progressPercent = progress;
    if (status === 'pending') progressPercent = 10;
    else if (status === 'processing') progressPercent = Math.max(20, Math.min(90, progress));
    else if (status === 'complete') progressPercent = 100;
    else if (status === 'failed') progressPercent = 0;
    
    progressBar.style.width = `${progressPercent}%`;
  }
}

// ---------------------------
// Debug Modal Functions
// ---------------------------
// Make debug modal functions globally accessible
window.openDebugModal = function() {
  const model = embeddingModelSelector.value;
  
  // Show loading state
  const modalHTML = `
    <div class="debug-modal-overlay" id="debug-modal-overlay">
      <div class="debug-modal-container">
        <div class="debug-modal-header">
          <div class="debug-modal-title">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
              <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Debug & Insights
          </div>
          <button class="debug-modal-close" onclick="closeDebugModal()">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-5 h-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div class="debug-modal-tabs">
          <button class="debug-tab active" data-tab="overview" onclick="switchDebugTab('overview')">Overview</button>
          <button class="debug-tab" data-tab="chunks" onclick="switchDebugTab('chunks')">Chunks</button>
          <button class="debug-tab" data-tab="requests" onclick="switchDebugTab('requests')">LLM Requests</button>
          <button class="debug-tab" data-tab="retrievals" onclick="switchDebugTab('retrievals')">Retrievals</button>
        </div>
        <div class="debug-modal-content">
          <div class="debug-tab-content active" id="debug-tab-overview">
            <div class="flex justify-center items-center h-64">
              <div class="spinner-large"></div>
            </div>
          </div>
          <div class="debug-tab-content" id="debug-tab-chunks"></div>
          <div class="debug-tab-content" id="debug-tab-requests"></div>
          <div class="debug-tab-content" id="debug-tab-retrievals"></div>
        </div>
      </div>
    </div>
  `;
  
  document.body.insertAdjacentHTML('beforeend', modalHTML);
  
  // Load debug data
  fetch(`/debug?session_id=${encodeURIComponent(currentSessionId)}&embedding_model=${encodeURIComponent(model)}`)
    .then(r => r.json())
    .then(data => {
      renderDebugOverview(data);
      renderDebugChunks(data);
      renderDebugRequests(data);
      renderDebugRetrievals(data);
    })
    .catch(err => {
      document.getElementById('debug-tab-overview').innerHTML = `
        <div class="debug-empty-state">
          <div class="debug-empty-state-icon">⚠️</div>
          <p>Failed to load debug data: ${escapeHtml(err.message)}</p>
        </div>
      `;
    });
  
  // Close on overlay click
  document.getElementById('debug-modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'debug-modal-overlay') {
      closeDebugModal();
    }
  });
  
  // Close on Escape key
  const escapeHandler = (e) => {
    if (e.key === 'Escape') {
      closeDebugModal();
      document.removeEventListener('keydown', escapeHandler);
    }
  };
  document.addEventListener('keydown', escapeHandler);
}

window.closeDebugModal = function() {
  const overlay = document.getElementById('debug-modal-overlay');
  if (overlay) {
    overlay.style.animation = 'fadeOut 0.2s ease-out';
    setTimeout(() => overlay.remove(), 200);
  }
}

window.switchDebugTab = function(tabName) {
  // Update tabs
  document.querySelectorAll('.debug-tab').forEach(tab => {
    tab.classList.remove('active');
    if (tab.dataset.tab === tabName) {
      tab.classList.add('active');
    }
  });
  
  // Update content
  document.querySelectorAll('.debug-tab-content').forEach(content => {
    content.classList.remove('active');
    if (content.id === `debug-tab-${tabName}`) {
      content.classList.add('active');
    }
  });
}

function renderDebugOverview(data) {
  const indexStatus = data.index_status || {};
  const html = `
    <div class="debug-section">
      <h3 class="debug-section-title">Index Status</h3>
      <div class="debug-info-grid">
        <div class="debug-info-item">
          <div class="debug-info-label">Index Name</div>
          <div class="debug-info-value">${escapeHtml(indexStatus.name || 'N/A')}</div>
        </div>
        <div class="debug-info-item">
          <div class="debug-info-label">Status</div>
          <div class="debug-info-value">${escapeHtml(indexStatus.status || 'UNKNOWN')}</div>
        </div>
        <div class="debug-info-item">
          <div class="debug-info-label">Queryable</div>
          <div class="debug-info-value">${indexStatus.queryable ? '✓ Yes' : '✗ No'}</div>
        </div>
        <div class="debug-info-item">
          <div class="debug-info-label">Documents</div>
          <div class="debug-info-value">${(indexStatus.document_count || 0).toLocaleString()}</div>
        </div>
      </div>
    </div>
    
    <div class="debug-section">
      <h3 class="debug-section-title">Session Info</h3>
      <div class="debug-info-grid">
        <div class="debug-info-item">
          <div class="debug-info-label">Session ID</div>
          <div class="debug-info-value">${escapeHtml(data.session_id || 'N/A')}</div>
        </div>
        <div class="debug-info-item">
          <div class="debug-info-label">Embedding Model</div>
          <div class="debug-info-value">${escapeHtml(data.embedding_model || 'N/A')}</div>
        </div>
        <div class="debug-info-item">
          <div class="debug-info-label">Chat History</div>
          <div class="debug-info-value">${data.chat_history_length || 0} messages</div>
        </div>
        <div class="debug-info-item">
          <div class="debug-info-label">Requests Stored</div>
          <div class="debug-info-value">${data.total_requests_stored || 0}</div>
        </div>
      </div>
    </div>
  `;
  document.getElementById('debug-tab-overview').innerHTML = html;
}

function renderDebugChunks(data) {
  const chunks = data.sample_chunks || [];
  if (chunks.length === 0) {
    document.getElementById('debug-tab-chunks').innerHTML = `
      <div class="debug-empty-state">
        <div class="debug-empty-state-icon">📄</div>
        <p>No chunks found in this session</p>
      </div>
    `;
    return;
  }
  
  const html = `
    <div class="debug-section">
      <h3 class="debug-section-title">Sample Chunks (${chunks.length})</h3>
      <div class="debug-chunk-list">
        ${chunks.map(chunk => `
          <div class="debug-chunk-item">
            <div class="debug-chunk-header">
              <span class="debug-chunk-id">ID: ${escapeHtml(chunk._id)}</span>
              <span class="debug-chunk-source" onclick="window.open('/source_content?session_id=${encodeURIComponent(data.session_id)}&source=${encodeURIComponent(chunk.source)}', '_blank')">
                ${escapeHtml(chunk.source)}
              </span>
            </div>
            <div class="debug-chunk-text">${escapeHtml(chunk.text)}</div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
  document.getElementById('debug-tab-chunks').innerHTML = html;
}

function renderDebugRequests(data) {
  const requests = data.recent_requests || [];
  if (requests.length === 0) {
    document.getElementById('debug-tab-requests').innerHTML = `
      <div class="debug-empty-state">
        <div class="debug-empty-state-icon">💬</div>
        <p>No LLM requests recorded yet</p>
        <p class="text-sm mt-2">Requests will appear here after you ask questions</p>
      </div>
    `;
    return;
  }
  
  const html = `
    <div class="debug-section">
      <h3 class="debug-section-title">Recent LLM Requests (${requests.length})</h3>
      ${requests.reverse().map(req => {
        const date = new Date(req.timestamp);
        return `
          <div class="debug-request-item">
            <div class="debug-request-header">
              <div class="debug-request-time">${date.toLocaleString()}</div>
              <div class="text-xs text-gray-400">Model: ${escapeHtml(req.embedding_model || 'N/A')} | k: ${req.num_sources || 'N/A'}</div>
            </div>
            <div class="debug-request-query">${escapeHtml(req.query)}</div>
            <div class="debug-request-details">
              <span>Response: ${(req.response_length || 0).toLocaleString()} chars</span>
              <span>Sources: ${(req.sources_used || []).length}</span>
              ${req.sources_used && req.sources_used.length > 0 ? `
                <div class="mt-2">
                  ${req.sources_used.map(src => `
                    <span class="text-xs bg-gray-700 px-2 py-1 rounded mr-1">${escapeHtml(src)}</span>
                  `).join('')}
                </div>
              ` : ''}
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
  document.getElementById('debug-tab-requests').innerHTML = html;
}

function renderDebugRetrievals(data) {
  const retrievals = data.recent_retrieved_chunks || [];
  if (retrievals.length === 0) {
    document.getElementById('debug-tab-retrievals').innerHTML = `
      <div class="debug-empty-state">
        <div class="debug-empty-state-icon">🔍</div>
        <p>No retrievals recorded yet</p>
        <p class="text-sm mt-2">Retrievals will appear here after searches</p>
      </div>
    `;
    return;
  }
  
  const html = `
    <div class="debug-section">
      <h3 class="debug-section-title">Recent Retrievals (${retrievals.length})</h3>
      ${retrievals.reverse().map(ret => {
        const date = new Date(ret.timestamp);
        return `
          <div class="debug-request-item">
            <div class="debug-request-header">
              <div class="debug-request-time">${date.toLocaleString()}</div>
              <div class="text-xs text-gray-400">Model: ${escapeHtml(ret.embedding_model || 'N/A')} | Chunks: ${(ret.chunks || []).length}</div>
            </div>
            <div class="debug-request-query">Query: ${escapeHtml(ret.query)}</div>
            <div class="mt-3">
              ${(ret.chunks || []).map((chunk, idx) => `
                <div class="debug-chunk-item mt-2">
                  <div class="debug-chunk-header">
                    <span class="text-xs text-gray-400">Chunk ${idx + 1}</span>
                    <span class="text-xs text-mongodb-green-500">Score: ${chunk.score?.toFixed(4) || 'N/A'}</span>
                    <span class="debug-chunk-source">${escapeHtml(chunk.source)}</span>
                  </div>
                  <div class="debug-chunk-text">${escapeHtml(chunk.text)}</div>
                </div>
              `).join('')}
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
  document.getElementById('debug-tab-retrievals').innerHTML = html;
}

function pollIngestionTask(taskId) {
  showIngestionOverlay();
  updateIngestionStatus('pending', 'Starting ingestion...', 10);
  
  let pollCount = 0;
  const maxPolls = 300; // 10 minutes max (300 * 2 seconds)
  
  const checkStatus = () => {
    pollCount++;
    if (pollCount > maxPolls) {
      hideIngestionOverlay();
      addSystemMessage(`Ingestion timeout: Task ${taskId} took too long. Please check server logs.`);
      return;
    }
    
    fetch(`/ingest/status/${taskId}`)
     .then(r => r.json())
     .then(data => {
      const status = data.status || 'pending';
      const step = data.step || data.message || '';
      
      // Calculate progress based on step
      let progress = 20;
      if (step.includes('Chunking')) progress = 30;
      else if (step.includes('Generating embeddings')) progress = 50;
      else if (step.includes('Verifying vector search') || step.includes('Checking indexes')) progress = 60;
      else if (step.includes('Preparing documents')) progress = 70;
      else if (step.includes('Saving')) progress = 85;
      else if (step.includes('Verifying')) progress = 95;
      
      updateIngestionStatus(status, step, progress);
      
      if (status === 'complete') {
        hideIngestionOverlay();
        addSystemMessage(`✅ Ingestion successful! ${data.message || ''}`);
        
        // Force immediate status refresh
        loadSessionsAndState();
        
        // Check index status and show appropriate message, then keep checking
        const checkIndexStatus = (attempt = 0) => {
          setTimeout(() => {
            loadSessionsAndState(); // Refresh to get latest status
            setTimeout(() => {
              const model = embeddingModelSelector.value;
              const idxStatus = indexStatusCache[model];
              
              if (idxStatus) {
                if (idxStatus.index_ready) {
                  addSystemMessage(`✅ Index is ready! (${idxStatus.document_count} documents indexed)`);
                } else if (idxStatus.index_status === 'CREATING' || idxStatus.index_status === 'BUILDING' || idxStatus.index_status === 'PENDING') {
                  if (attempt === 0) {
                    addSystemMessage(`⏳ Index is ${idxStatus.index_status.toLowerCase()} (${idxStatus.document_count} documents). This may take 10-30 seconds. Status will update automatically.`);
                  }
                  // Keep checking every 5 seconds until ready (max 12 attempts = 60 seconds)
                  if (attempt < 12) {
                    checkIndexStatus(attempt + 1);
                  }
                } else if (idxStatus.index_status === 'NOT_FOUND' && idxStatus.document_count > 0) {
                  if (attempt === 0) {
                    addSystemMessage(`⏳ Index creation initiated for ${idxStatus.document_count} documents. This may take 10-30 seconds.`);
                  }
                  // Keep checking every 5 seconds until ready (max 12 attempts = 60 seconds)
                  if (attempt < 12) {
                    checkIndexStatus(attempt + 1);
                  }
                } else {
                  addSystemMessage('💡 Note: Vector search indexes may take 10-30 seconds to index new documents. If search doesn\'t work immediately, wait a moment and try again.');
                }
              }
            }, 500); // Small delay to ensure cache is updated
          }, attempt === 0 ? 1000 : 5000); // First check after 1s, then every 5s
        };
        
        checkIndexStatus();
      } else if (status === 'failed') {
        hideIngestionOverlay();
        addSystemMessage(`❌ Ingestion failed: ${data.message || 'Unknown error'}`);
      } else {
        // Continue polling
        ingestionStatusInterval = setTimeout(checkStatus, 2000);
      }
     })
     .catch(err => {
      console.error('Failed to get ingestion status:', err);
      // Retry on error
      ingestionStatusInterval = setTimeout(checkStatus, 2000);
     });
  };
  
  // Start polling after a short delay
  ingestionStatusInterval = setTimeout(checkStatus, 1000);
}

// ---------------------------
// Source Content Modal
// ---------------------------
window.openSourceContentModal = function(content, sourceName = '') {
  const modalHTML = `
    <div class="source-content-modal-overlay" id="source-content-modal-overlay">
      <div class="source-content-modal-container">
        <div class="source-content-modal-header">
          <div class="source-content-modal-title">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
            Source Content${sourceName ? `: ${escapeHtml(sourceName)}` : ''}
          </div>
          <button class="source-content-modal-close" onclick="closeSourceContentModal()">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-5 h-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div class="source-content-modal-content">
          <textarea id="source-content-textarea" class="source-content-textarea" readonly>${escapeHtmlForTextarea(content || '')}</textarea>
        </div>
        <div class="source-content-modal-footer">
          <button class="btn btn-secondary" onclick="closeSourceContentModal()">Close</button>
          <button class="btn btn-primary" onclick="copySourceContent()">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
              <path stroke-linecap="round" stroke-linejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" />
            </svg>
            Copy
          </button>
        </div>
      </div>
    </div>
  `;
  
  document.body.insertAdjacentHTML('beforeend', modalHTML);
  
  // Close on overlay click
  document.getElementById('source-content-modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'source-content-modal-overlay') {
      closeSourceContentModal();
    }
  });
  
  // Close on Escape key
  const escapeHandler = (e) => {
    if (e.key === 'Escape') {
      closeSourceContentModal();
      document.removeEventListener('keydown', escapeHandler);
    }
  };
  document.addEventListener('keydown', escapeHandler);
  
  // Focus textarea
  setTimeout(() => {
    const textarea = document.getElementById('source-content-textarea');
    if (textarea) {
      textarea.focus();
      textarea.select();
    }
  }, 100);
}

window.closeSourceContentModal = function() {
  const overlay = document.getElementById('source-content-modal-overlay');
  if (overlay) {
    overlay.style.animation = 'fadeOut 0.2s ease-out';
    setTimeout(() => overlay.remove(), 200);
  }
}

window.copySourceContent = function() {
  const textarea = document.getElementById('source-content-textarea');
  if (textarea) {
    textarea.select();
    document.execCommand('copy');
    // Show brief feedback
    const btn = event.target.closest('button');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4"><path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg> Copied!';
    setTimeout(() => {
      btn.innerHTML = originalText;
    }, 2000);
  }
}

// ---------------------------
// Chunk Inspection Modal
// ---------------------------
window.inspectRetrievedChunks = function(messageId) {
  const query = messageQueryMap.get(messageId);
  if (!query) {
    alert('Query information not available for this message.');
    return;
  }
  
  const model = embeddingModelSelector.value;
  
  // Show loading state
  const modalHTML = `
    <div class="chunk-inspection-modal-overlay" id="chunk-inspection-modal-overlay">
      <div class="chunk-inspection-modal-container">
        <div class="chunk-inspection-modal-header">
          <div class="chunk-inspection-modal-title">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639l4.43-7.29a1.125 1.125 0 011.906 0l4.43 7.29c.356.586.356 1.35 0 1.936l-4.43 7.29a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
            </svg>
            Retrieved Chunks
          </div>
          <button class="chunk-inspection-modal-close" onclick="closeChunkInspectionModal()">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-5 h-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div class="chunk-inspection-modal-content">
          <div class="chunk-inspection-query">
            <div class="text-sm text-gray-400 mb-1">Query:</div>
            <div class="text-base font-medium text-white">${escapeHtml(query)}</div>
          </div>
          <div class="flex justify-center items-center h-64">
            <div class="spinner-large"></div>
          </div>
        </div>
      </div>
    </div>
  `;
  
  document.body.insertAdjacentHTML('beforeend', modalHTML);
  
  // Fetch retrieved chunks from debug endpoint
  fetch(`/debug?session_id=${encodeURIComponent(currentSessionId)}&embedding_model=${encodeURIComponent(model)}`)
    .then(r => r.json())
    .then(data => {
      const retrievals = data.recent_retrieved_chunks || [];
      
      // Find the most recent retrieval matching this query
      const matchingRetrieval = retrievals
        .filter(ret => ret.query === query && ret.session_id === currentSessionId)
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
      
      if (!matchingRetrieval || !matchingRetrieval.chunks || matchingRetrieval.chunks.length === 0) {
        document.querySelector('#chunk-inspection-modal-overlay .chunk-inspection-modal-content').innerHTML = `
          <div class="chunk-inspection-query">
            <div class="text-sm text-gray-400 mb-1">Query:</div>
            <div class="text-base font-medium text-white">${escapeHtml(query)}</div>
          </div>
          <div class="debug-empty-state">
            <div class="debug-empty-state-icon">🔍</div>
            <p>No chunks found for this query</p>
            <p class="text-sm mt-2">The retrieval may have occurred too long ago or no chunks were retrieved.</p>
          </div>
        `;
        return;
      }
      
      const chunks = matchingRetrieval.chunks;
      const html = `
        <div class="chunk-inspection-query">
          <div class="text-sm text-gray-400 mb-1">Query:</div>
          <div class="text-base font-medium text-white">${escapeHtml(query)}</div>
          <div class="text-xs text-gray-500 mt-1">Model: ${escapeHtml(model)} | ${chunks.length} chunks retrieved</div>
        </div>
        <div class="chunk-inspection-list">
          ${chunks.map((chunk, idx) => `
            <div class="chunk-inspection-item">
              <div class="chunk-inspection-item-header">
                <div class="flex items-center gap-2">
                  <span class="chunk-inspection-item-number">${idx + 1}</span>
                  <span class="chunk-inspection-item-score">Score: ${(chunk.score || 0).toFixed(4)}</span>
                </div>
                <a href="/source_content?session_id=${encodeURIComponent(currentSessionId)}&source=${encodeURIComponent(chunk.source)}" 
                   target="_blank" 
                   class="chunk-inspection-item-source">
                  ${escapeHtml(chunk.source)}
                </a>
              </div>
              <div class="chunk-inspection-item-content prose prose-invert max-w-none prose-sm">
                ${marked.parse(chunk.text || '')}
              </div>
            </div>
          `).join('')}
        </div>
      `;
      
      document.querySelector('#chunk-inspection-modal-overlay .chunk-inspection-modal-content').innerHTML = html;
    })
    .catch(err => {
      document.querySelector('#chunk-inspection-modal-overlay .chunk-inspection-modal-content').innerHTML = `
        <div class="chunk-inspection-query">
          <div class="text-sm text-gray-400 mb-1">Query:</div>
          <div class="text-base font-medium text-white">${escapeHtml(query)}</div>
        </div>
        <div class="debug-empty-state">
          <div class="debug-empty-state-icon">⚠️</div>
          <p>Failed to load chunks: ${escapeHtml(err.message)}</p>
        </div>
      `;
    });
  
  // Close on overlay click
  document.getElementById('chunk-inspection-modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'chunk-inspection-modal-overlay') {
      closeChunkInspectionModal();
    }
  });
  
  // Close on Escape key
  const escapeHandler = (e) => {
    if (e.key === 'Escape') {
      closeChunkInspectionModal();
      document.removeEventListener('keydown', escapeHandler);
    }
  };
  document.addEventListener('keydown', escapeHandler);
}

window.closeChunkInspectionModal = function() {
  const overlay = document.getElementById('chunk-inspection-modal-overlay');
  if (overlay) {
    overlay.style.animation = 'fadeOut 0.2s ease-out';
    setTimeout(() => overlay.remove(), 200);
  }
}

// Helper function to open source content from ingestion modal
window.openSourceContentFromIngestion = function() {
  const textarea = document.getElementById('ingestion-source-content-textarea');
  if (!textarea || !textarea.value) {
    alert('No content to display. Please load content first.');
    return;
  }
  openSourceContentModal(textarea.value, 'Source Content');
}