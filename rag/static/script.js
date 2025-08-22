
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
      sourceBrowserSelectedChunkCount.textContent = "Error";
     });
   }
  
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
  
   function onEditChunkClick(chunkId) {
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
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ content: newText }),
      })
       .then((r) => r.json())
       .then((resp) => {
        if (resp.error) {
         alert("Error updating chunk: " + resp.error);
         return;
        }
        chunkCard.querySelector(".chunk-content").textContent = newText;
        hideModal();
       })
       .catch((err) => {
        alert("Error updating chunk: " + err.message);
       });
     },
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
   function addBotMessage(message) {
    const content = message.content;
    const sources = message.sources || [];

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
     let sourceLinksHTML = sources.map(source => {
      // MODIFIED: All links now point to the new /source_content endpoint
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
     sourcesContainer.innerHTML = `
       <h4 class="text-xs font-bold uppercase text-gray-400 mb-2">Sources</h4>
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
   function loadSessionsAndState() {
    fetch("/state")
     .then((r) => r.json())
     .then((data) => {
      allSessions = data.all_sessions || [];
      availableModels = data.available_embedding_models || [];
      currentSessionId = data.current_session || "default";
     
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
     })
     .catch((err) => {
      console.error("Failed to load state:", err);
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
   document.addEventListener("DOMContentLoaded", () => {
    loadSessionsAndState();
   });
  
   sessionSelector.addEventListener("change", () => {
    const sel = sessionSelector.value;
    if (sel !== currentSessionId) {
     switchSession(sel);
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
  
   chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;
   
    addUserMessage(text);
    setThinking(true);
   
    const embeddingModel = embeddingModelSelector.value;
    const numSources = parseInt(numSourcesInput.value) || 3;
    const maxChunkLen = parseInt(maxCharsInput.value) || 2000;
   
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

     const modalHTML = `
      <input
        type="file"
        id="ingestion-file-input"
        class="w-full text-sm text-gray-300 border border-gray-600 rounded-lg cursor-pointer bg-gray-700 focus:outline-none focus:ring-2 focus:ring-mongodb-green-500 file:bg-gray-800 file:text-mongodb-green-500 file:font-semibold file:border-0 file:py-2.5 file:px-4 file:mr-4 hover:file:bg-mongodb-green-500 hover:file:text-gray-900 transition-colors duration-200"
      />
       <div class="flex gap-4 mt-4 h-[55vh]">
         <div class="w-1/2 flex flex-col bg-gray-900/50 rounded-lg">
           <h4 class="font-bold text-mongodb-green-500 border-b border-gray-700 p-3">Source Content (Editable)</h4>
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
       text: "Select a file, edit content if needed, adjust chunking, and submit to ingest.",
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

     const fileInput = document.getElementById('ingestion-file-input');
     const contentTextarea = document.getElementById('ingestion-source-content-textarea');
     const rechunkBtn = document.getElementById('ingestion-rechunk-btn');

     fileInput.addEventListener('change', (e) => {
       const file = e.target.files[0];
       if (!file) return;

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
           <h4 class="font-bold text-mongodb-green-500 border-b border-gray-700 p-3">Source Content (Editable)</h4>
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
  
   function pollIngestionTask(taskId) {
    const checkStatus = () => {
     fetch(`/ingest/status/${taskId}`)
      .then(r => r.json())
      .then(data => {
       if (data.status === 'complete') {
        addSystemMessage(`Ingestion successful! ${data.message}`);
        loadSessionsAndState();
       } else if (data.status === 'failed') {
        addSystemMessage(`Ingestion failed: ${data.message}`);
       } else {
        setTimeout(checkStatus, 2000);
       }
      })
      .catch(err => {
       addSystemMessage(`Failed to get ingestion status: ${err.message}`);
      });
    };
    addSystemMessage(`Ingestion started with Task ID: ${taskId}. This may take a moment.`);
    setTimeout(checkStatus, 2000);
   }