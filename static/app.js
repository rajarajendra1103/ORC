// ==========================================================================
// AI Agent - OCR Document Intelligence JavaScript Client
// ==========================================================================

let activeDocuments = [];
let selectedDocId = null;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initFileUpload();
    loadDatasetStats();
});

// Navigation Tab Switcher
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanels = document.querySelectorAll('.tab-panel');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const tabId = item.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// File Upload Handler
function initFileUpload() {
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('file-dropzone');

    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#818CF8';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'var(--primary)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary)';
        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });
}

async function handleFiles(files) {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    const spinner = document.getElementById('processing-spinner');
    spinner.style.display = 'block';

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        spinner.style.display = 'none';

        if (data.status === 'success') {
            activeDocuments = data.documents;
            populateDocSelector();
            document.getElementById('document-results-section').style.display = 'block';
        } else {
            alert('Upload failed: ' + (data.detail || 'Unknown error'));
        }
    } catch (err) {
        spinner.style.display = 'none';
        alert('Error connecting to backend server: ' + err.message);
    }
}

function populateDocSelector() {
    const selector = document.getElementById('doc-selector');
    selector.innerHTML = '';

    activeDocuments.forEach(doc => {
        const opt = document.createElement('option');
        opt.value = doc.doc_id;
        opt.textContent = `${doc.filename} (${doc.total_pages} page(s))`;
        selector.appendChild(opt);
    });

    if (activeDocuments.length > 0) {
        selectedDocId = activeDocuments[0].doc_id;
        renderActiveDocView();
    }
}

function renderActiveDocView() {
    const selector = document.getElementById('doc-selector');
    selectedDocId = selector.value;
    const doc = activeDocuments.find(d => d.doc_id === selectedDocId);

    if (!doc || !doc.pages || doc.pages.length === 0) return;

    const page1 = doc.pages[0];
    
    // Rendered image
    document.getElementById('rendered-doc-img').src = page1.image_base64;

    // TrOCR output
    document.getElementById('trocr-text-output').textContent = page1.text || 'No text extracted.';

    // LayoutLMv3 output (Formatted Normal Text)
    document.getElementById('layoutlm-output').innerHTML = formatLayoutLMOutput(page1.layout_summary);

    // Donut output (Formatted Normal Text)
    document.getElementById('donut-output').innerHTML = formatDonutOutput(doc.structured_knowledge);
}

function formatLayoutLMOutput(summary) {
    if (!summary) return '<p style="color: var(--text-muted);">No layout data available.</p>';
    let html = `
        <div style="font-size: 0.95rem; color: var(--text-primary);">
            <p style="margin-bottom: 6px;"><strong>Total Blocks Identified:</strong> ${summary.total_blocks || 0}</p>
            <p style="margin-bottom: 6px;"><strong>Headers & Titles:</strong> ${summary.header_count || 0}</p>
            <p style="margin-bottom: 6px;"><strong>Paragraph Blocks:</strong> ${summary.paragraph_count || 0}</p>
            <p style="margin-bottom: 6px;"><strong>Table Regions:</strong> ${summary.table_region_count || 0}</p>
    `;
    if (summary.detected_headers && summary.detected_headers.length > 0) {
        html += `<p style="margin-top: 8px;"><strong>Sample Headers:</strong> ${summary.detected_headers.join(', ')}</p>`;
    }
    html += `</div>`;
    return html;
}

function formatDonutOutput(structured) {
    if (!structured) return '<p style="color: var(--text-muted);">No structured data available.</p>';
    let html = `
        <div style="font-size: 0.95rem; color: var(--text-primary);">
            <p style="margin-bottom: 6px;"><strong>Document Category:</strong> <span style="color: var(--text-accent); font-weight: 600;">${structured.document_type || 'General Document'}</span></p>
            <p style="margin-bottom: 6px;"><strong>Summary:</strong> ${structured.structured_summary || 'N/A'}</p>
    `;

    if (structured.metadata) {
        const meta = structured.metadata;
        if (meta.extracted_dates && meta.extracted_dates.length > 0) {
            html += `<p style="margin-bottom: 6px;"><strong>Extracted Dates:</strong> ${meta.extracted_dates.join(', ')}</p>`;
        }
        if (meta.extracted_amounts && meta.extracted_amounts.length > 0) {
            html += `<p style="margin-bottom: 6px;"><strong>Extracted Amounts:</strong> ${meta.extracted_amounts.join(', ')}</p>`;
        }
        if (meta.contacts && (meta.contacts.emails.length > 0 || meta.contacts.urls.length > 0)) {
            const contacts = [...meta.contacts.emails, ...meta.contacts.urls];
            html += `<p style="margin-bottom: 6px;"><strong>Contacts / Links:</strong> ${contacts.join(', ')}</p>`;
        }
    }

    if (structured.key_value_attributes && Object.keys(structured.key_value_attributes).length > 0) {
        html += `<p style="margin-top: 8px; margin-bottom: 4px;"><strong>Extracted Key Fields:</strong></p><ul style="padding-left: 20px; font-size: 0.9rem;">`;
        for (const [k, v] of Object.entries(structured.key_value_attributes)) {
            html += `<li><strong>${k}:</strong> ${v}</li>`;
        }
        html += `</ul>`;
    }
    html += `</div>`;
    return html;
}

// AI Document Chat
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query) return;

    const chatBox = document.getElementById('chat-box');

    // Append User Message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-bubble user';
    userMsg.textContent = query;
    chatBox.appendChild(userMsg);

    input.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, doc_ids: activeDocuments.map(d => d.doc_id) })
        });

        const data = await response.json();

        // Format clean answer (convert **text** to bold strong tag and remove raw asterisks)
        let formattedAnswer = (data.answer || '')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');

        // Append Agent Message
        const agentMsg = document.createElement('div');
        agentMsg.className = 'chat-bubble agent';
        agentMsg.innerHTML = `<strong>🤖 Answer:</strong><br>${formattedAnswer}<br><br><small style="color: var(--text-muted);">Sources: ${data.sources.join(', ')} | Confidence: ${(data.confidence*100).toFixed(0)}%</small>`;
        chatBox.appendChild(agentMsg);

        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (err) {
        alert('Chat error: ' + err.message);
    }
}

// Automatic Summarizer
async function fetchSummary() {
    if (!selectedDocId) {
        alert('Please upload a document first.');
        return;
    }

    try {
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doc_id: selectedDocId })
        });

        const data = await response.json();
        document.getElementById('summary-results').style.display = 'block';
        document.getElementById('executive-summary-text').textContent = data.executive_summary;

        const pointsList = document.getElementById('key-points-list');
        pointsList.innerHTML = '';
        data.key_points.forEach(pt => {
            const li = document.createElement('li');
            li.style.marginBottom = '8px';
            li.textContent = pt;
            pointsList.appendChild(li);
        });

        const topicsContainer = document.getElementById('topic-summaries-container');
        topicsContainer.innerHTML = '';
        for (const [topic, text] of Object.entries(data.topic_summaries)) {
            const div = document.createElement('div');
            div.className = 'card';
            div.style.padding = '14px';
            div.style.marginBottom = '12px';
            div.innerHTML = `<h4 style="color: var(--text-accent);">${topic}</h4><p style="color: var(--text-primary); font-size: 0.95rem;">${text}</p>`;
            topicsContainer.appendChild(div);
        }
    } catch (err) {
        alert('Summarizer error: ' + err.message);
    }
}

// MCQ Quiz Generator
async function generateMCQs() {
    if (!selectedDocId) {
        alert('Please upload a document first.');
        return;
    }

    const num = document.getElementById('mcq-num').value;
    const diff = document.getElementById('mcq-diff').value;

    try {
        const response = await fetch('/api/mcq', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doc_id: selectedDocId, num_questions: num, difficulty: diff })
        });

        const data = await response.json();
        const container = document.getElementById('mcq-container');
        container.innerHTML = '';

        data.mcqs.forEach((item, idx) => {
            const card = document.createElement('div');
            card.className = 'mcq-card';
            
            let optionsHtml = '';
            for (const [key, val] of Object.entries(item.options)) {
                optionsHtml += `<div class="mcq-option"><strong>(${key})</strong> ${val}</div>`;
            }

            card.innerHTML = `
                <h4 style="color: var(--text-primary); margin-bottom: 10px;">Q${item.question_num}. ${item.question} <span style="font-size: 0.8rem; color: var(--text-accent); float: right;">${item.difficulty}</span></h4>
                ${optionsHtml}
                <div style="margin-top: 12px;">
                    <button class="btn" style="background: var(--bg-card); color: var(--text-accent); padding: 6px 12px; font-size: 0.85rem;" onclick="this.nextElementSibling.style.display='block'">Show Answer</button>
                    <div style="display:none; margin-top: 10px; padding: 10px; background: var(--bg-card); border-radius: 6px; border: 1px solid var(--border-color);">
                        <strong style="color: var(--success);">Answer: (${item.correct_answer}) ${item.correct_text}</strong>
                        <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">${item.explanation}</p>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    } catch (err) {
        alert('MCQ error: ' + err.message);
    }
}

// Keyword Highlighter
async function fetchHighlighting() {
    if (!selectedDocId) {
        alert('Please upload a document first.');
        return;
    }

    const input = document.getElementById('keywords-input').value;
    const keywords = input.split(',').map(k => k.trim()).filter(k => k);

    try {
        const response = await fetch('/api/highlight', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doc_id: selectedDocId, page_num: 1, keywords: keywords })
        });

        const data = await response.json();
        document.getElementById('highlighted-img-view').src = data.highlighted_image;
    } catch (err) {
        alert('Highlight error: ' + err.message);
    }
}

// Dataset Stats & Search
async function loadDatasetStats() {
    try {
        const res = await fetch('/api/dataset/stats');
        const data = await res.json();
        // pre-loaded numbers
    } catch (e) {}
}

async function searchDataset() {
    const input = document.getElementById('dataset-search-input').value.trim();
    
    try {
        const response = await fetch(`/api/dataset/search?query=${encodeURIComponent(input)}`);
        const data = await response.json();
        
        const container = document.getElementById('dataset-results-container');
        if (!data.results || data.results.length === 0) {
            container.innerHTML = '<p style="color: var(--text-muted);">No matching document records found.</p>';
            return;
        }

        let tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Image ID</th>
                        <th>Document Text Snippet</th>
                        <th>Word Count</th>
                        <th>Line Count</th>
                    </tr>
                </thead>
                <tbody>
        `;

        data.results.forEach(row => {
            tableHtml += `
                <tr>
                    <td><code>${row.image_id}</code></td>
                    <td>${row.full_document_text.substring(0, 100)}...</td>
                    <td>${row.word_count}</td>
                    <td>${row.line_count}</td>
                </tr>
            `;
        });

        tableHtml += '</tbody></table>';
        container.innerHTML = tableHtml;
    } catch (err) {
        alert('Search error: ' + err.message);
    }
}
