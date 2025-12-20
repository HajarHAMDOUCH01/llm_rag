const API_URL = 'http://localhost:5000/api';

const elements = {
    chatMessages: document.getElementById('chatMessages'),
    questionInput: document.getElementById('questionInput'),
    sendButton: document.getElementById('sendButton'),
    status: document.getElementById('status')
};

// Auto-resize textarea
elements.questionInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Send message on Enter (Shift+Enter for new line)
elements.questionInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Send button click
elements.sendButton.addEventListener('click', sendMessage);

// Check server health on load
checkHealth();

async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            updateStatus('Ready', 'ready');
        } else {
            updateStatus('Server error', 'error');
        }
    } catch (error) {
        updateStatus('Server offline', 'error');
        console.error('Health check failed:', error);
    }
}

function updateStatus(message, state = 'ready') {
    const statusText = elements.status.querySelector('span:last-child');
    statusText.textContent = message;
    
    elements.status.className = 'status';
    if (state === 'error') {
        elements.status.classList.add('error');
    } else if (state === 'processing') {
        elements.status.classList.add('processing');
    }
}

async function sendMessage() {
    const question = elements.questionInput.value.trim();
    
    if (!question) return;
    
    // Remove welcome message if exists
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Add question to chat
    addMessage(question, 'question');
    
    // Clear input
    elements.questionInput.value = '';
    elements.questionInput.style.height = 'auto';
    
    // Disable send button
    elements.sendButton.disabled = true;
    updateStatus('Processing...', 'processing');
    
    // Show loading indicator
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });
        
        if (!response.ok) {
            throw new Error('Server error');
        }
        
        const data = await response.json();
        
        // Remove loading indicator
        removeLoadingMessage(loadingId);
        
        // Add answer to chat
        addMessage(data.answer, 'answer', data.sources);
        
        updateStatus('Ready', 'ready');
        
    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage('Sorry, there was an error processing your question. Please make sure the server is running.', 'answer');
        updateStatus('Error', 'error');
        console.error('Error:', error);
    } finally {
        elements.sendButton.disabled = false;
        elements.questionInput.focus();
    }
}

function addMessage(text, type, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    
    if (type === 'question') {
        messageDiv.classList.add('message-question');
        messageDiv.innerHTML = `
            <div class="message-bubble question-bubble">
                ${escapeHtml(text)}
            </div>
        `;
    } else {
        messageDiv.classList.add('message-answer');
        const sourcesHtml = sources && sources.length > 0 ? createSourcesHtml(sources) : '';
        messageDiv.innerHTML = `
            <div class="answer-container">
                <div class="message-bubble answer-bubble">
                    ${escapeHtml(text)}
                </div>
                ${sourcesHtml}
            </div>
        `;
    }
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function createSourcesHtml(sources) {
    if (!sources || sources.length === 0) return '';
    
    const sourcesItems = sources.map((source, index) => `
        <div class="source-item">
            <div class="source-file">${index + 1}. ${escapeHtml(source.file)} (Page ${source.page})</div>
            <div class="source-text">${escapeHtml(source.text)}</div>
        </div>
    `).join('');
    
    return `
        <div class="sources">
            <div class="sources-title">Sources</div>
            ${sourcesItems}
        </div>
    `;
}

function addLoadingMessage() {
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message message-answer';
    loadingDiv.innerHTML = `
        <div class="answer-container">
            <div class="loading">
                <span>Thinking</span>
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        </div>
    `;
    
    elements.chatMessages.appendChild(loadingDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    return loadingId;
}

function removeLoadingMessage(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}