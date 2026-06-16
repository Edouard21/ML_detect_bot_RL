/* ============================================================
   RL Bot Detector — Frontend Logic
   ============================================================ */

(() => {
    'use strict';

    // --- Configuration ---
    const MAX_FILE_SIZE = Math.floor(4.5 * 1024 * 1024); // Vercel Hobby payload limit

    // --- DOM Elements ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadSection = document.getElementById('uploadSection');
    const loadingSection = document.getElementById('loadingSection');
    const errorSection = document.getElementById('errorSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorMessage = document.getElementById('errorMessage');
    const retryBtn = document.getElementById('retryBtn');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    const resultsGrid = document.getElementById('resultsGrid');
    const replayName = document.getElementById('replayName');
    const loaderSubtext = document.getElementById('loaderSubtext');

    // --- State ---
    let isProcessing = false;

    // --- Section Management ---
    function showSection(section) {
        [uploadSection, loadingSection, errorSection, resultsSection].forEach(s => {
            s.hidden = true;
        });
        section.hidden = false;
    }

    // --- Drag & Drop ---
    dropZone.addEventListener('click', () => {
        if (!isProcessing) fileInput.click();
    });

    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            if (!isProcessing) fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // --- File Handling ---
    function handleFile(file) {
        // Validate extension
        if (!file.name.toLowerCase().endsWith('.replay')) {
            showError('Invalid file type. Please upload a .replay file from your Rocket League replays folder.');
            return;
        }

        // Validate size
        if (file.size > MAX_FILE_SIZE) {
            showError(`File too large (${formatSize(file.size)}). Maximum size is ${formatSize(MAX_FILE_SIZE)}.`);
            return;
        }

        uploadAndAnalyze(file);
    }

    // --- API Call ---
    async function uploadAndAnalyze(file) {
        isProcessing = true;
        showSection(loadingSection);

        // Animate loader subtexts
        const messages = [
            'Securely uploading to proxy...',
            'Extracting player data...',
            'Parsing network frames...',
            'Running AI model...',
            'Aggregating predictions...',
        ];
        let msgIdx = 0;
        const msgInterval = setInterval(() => {
            msgIdx = (msgIdx + 1) % messages.length;
            loaderSubtext.textContent = messages[msgIdx];
        }, 2500);

        try {
            const formData = new FormData();
            formData.append('file', file);

            // Calls the secure Vercel proxy instead of the Hugging Face API directly
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            clearInterval(msgInterval);

            if (!response.ok) {
                let detail = 'Unknown error occurred.';
                try {
                    const errData = await response.json();
                    detail = errData.detail || detail;
                } catch (_) {
                    // ignore JSON parse errors
                }
                showError(detail);
                return;
            }

            const data = await response.json();
            displayResults(data);
        } catch (err) {
            clearInterval(msgInterval);
            if (err.name === 'TypeError' && err.message.includes('fetch')) {
                showError('Could not connect to the server. Please make sure the backend is running.');
            } else {
                showError(`Network error: ${err.message}`);
            }
        } finally {
            isProcessing = false;
            fileInput.value = '';
        }
    }

    // --- Display Results ---
    function displayResults(data) {
        showSection(resultsSection);

        // Set replay name
        replayName.textContent = data.replay_name || 'Unknown replay';

        // Compute summary
        const players = data.players || [];
        const realCount = players.filter(p => p.prediction === 'real').length;
        const botCount = players.filter(p => p.prediction === 'bots').length;

        animateCounter('summaryPlayers', players.length);
        animateCounter('summaryReal', realCount);
        animateCounter('summaryBots', botCount);

        // Render player cards
        resultsGrid.innerHTML = '';
        players.forEach((player, idx) => {
            const card = createPlayerCard(player, idx);
            resultsGrid.appendChild(card);
        });

        // Animate confidence bars after a brief delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                document.querySelectorAll('.confidence-bar-fill').forEach(bar => {
                    bar.style.width = bar.dataset.width;
                });
            }, 100);
        });
    }

    function createPlayerCard(player, index) {
        const isBot = player.prediction === 'bots';
        const typeClass = isBot ? 'is-bot' : 'is-real';
        const emoji = isBot ? '🤖' : '👤';
        const label = isBot ? 'Bot' : 'Real';
        const confidence = player.confidence;

        const card = document.createElement('div');
        card.className = `player-card ${typeClass}`;
        card.style.animationDelay = `${index * 0.08}s`;

        card.innerHTML = `
            <div class="player-avatar">${emoji}</div>
            <div class="player-info">
                <div class="player-name" title="${escapeHtml(player.name)}">${escapeHtml(player.name)}</div>
                <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" data-width="${confidence}%"></div>
                </div>
            </div>
            <div class="player-result">
                <div class="result-badge">${label}</div>
                <div class="result-confidence">${confidence.toFixed(1)}%</div>
            </div>
        `;

        return card;
    }

    // --- Error Display ---
    function showError(message) {
        errorMessage.textContent = message;
        showSection(errorSection);
        isProcessing = false;
    }

    // --- Button Handlers ---
    retryBtn.addEventListener('click', () => {
        showSection(uploadSection);
    });

    newAnalysisBtn.addEventListener('click', () => {
        showSection(uploadSection);
    });

    // --- Utilities ---
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function animateCounter(elementId, target) {
        const el = document.querySelector(`#${elementId} .summary-value`);
        if (!el) return;

        const duration = 600;
        const start = performance.now();
        const from = 0;

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(from + (target - from) * eased);
            el.textContent = current;
            if (progress < 1) {
                requestAnimationFrame(tick);
            }
        }

        requestAnimationFrame(tick);
    }

})();
