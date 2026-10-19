

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    loadingOverlay: $('#loading-overlay'),
    loadingStatus: $('#loading-status'),
    connectionStatus: $('#connection-status'),
    fpsText: $('#fps-text'),
    fpsBadge: $('#fps-badge'),
    langSelect: $('#lang-select'),
    cameraPlaceholder: $('#camera-placeholder'),
    handStatus: $('#hand-status'),
    handIcon: $('#hand-icon'),
    pipelineStages: $$('.pipeline-stage'),
    predictionRows: $$('.prediction-row'),
    collectorPhase: $('#collector-phase'),
    wordChips: $('#word-chips'),
    wordProgressFill: $('#word-progress-fill'),
    wordTimer: $('#word-timer'),
    llmStatus: $('#llm-status'),
    llmOutput: $('#llm-output'),
    ttsStatus: $('#tts-status'),
    audioWave: $('#audio-wave'),
    btnStart: $('#btn-start'),
    btnStop: $('#btn-stop'),
    btnReset: $('#btn-reset'),
    btnForceLlm: $('#btn-force-llm'),
    videoInput: $('#video-input'),
    canvasOutput: $('#canvas-output'),

    
    btnModeInference: $('#btn-mode-inference'),
    btnModeTraining: $('#btn-mode-training'),
    btnModeV2S: $('#btn-mode-v2s'),
    inferenceSection: $('#inference-section'),
    trainingSection: $('#training-section'),
    v2sSection: $('#v2s-section'),
    
    
    btnMicStart: $('#btn-mic-start'),
    v2sTranscript: $('#v2s-transcript'),
    v2sGlosses: $('#v2s-glosses'),
    newLabelInput: $('#new-label-input'),
    btnAddLabel: $('#btn-add-label'),
    labelsTbody: $('#labels-tbody'),
    collectLabelName: $('#collect-label-name'),
    collectFramesBadge: $('#collect-frames'),
    collectProgressFill: $('#collect-progress-fill'),
    btnRecord: $('#btn-record'),
    btnFewShot: $('#btn-fewshot'),
    btnTrain: $('#btn-train'),
    trainConsole: $('#train-console'),
};

let ws = null;
let handProcessor = null;
let isRunning = false;
let serverConfig = null;
let pipelineStage = 0;
let pipelineInterval = null;
let currentAudio = null;

let appMode = 'inference'; 
let isRecording = false;
let isFewShotRecording = false;
let recordedFrames = [];
const MAX_RECORD_FRAMES = 60;
let selectedLabelSlug = null;

function getWsUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws/sign-language`;
}

function connectWebSocket() {
    DOM.loadingStatus.textContent = 'Kết nối WebSocket...';
    DOM.connectionStatus.textContent = 'Đang kết nối...';

    ws = new WebSocket(getWsUrl());

    ws.onopen = () => {
        console.log('[WS] Connected');
        DOM.connectionStatus.textContent = 'Đã kết nối ✓';
        DOM.loadingStatus.textContent = 'Đang tải AI Engine...';
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        DOM.connectionStatus.textContent = 'Mất kết nối — đang thử lại...';
        DOM.fpsBadge.style.display = 'none';
        
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (e) => {
        console.error('[WS] Error', e);
    };

    ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
            handleAudioBlob(event.data);
            return;
        }
        try {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        } catch (err) {
            console.error('[WS] Parse error:', err);
        }
    };
}

function handleServerMessage(msg) {
    switch (msg.type) {
        case 'init':
            serverConfig = msg;
            console.log('[Init]', msg);
            DOM.loadingOverlay.classList.add('hidden');
            DOM.wordTimer.textContent = `mỗi từ ~${msg.word_seconds}s`;
            if (!msg.engine_ready) {
                DOM.llmStatus.textContent = 'LLM: ' + msg.llm_mode;
            }
            break;

        case 'prediction':
            updatePredictions(msg.top5, msg.best_p, msg.best_word);
            break;

        case 'word_locked':
            updateWords(msg.words);
            flashWordLock(msg.display);
            break;

        case 'collector_status':
            DOM.collectorPhase.textContent = msg.status;
            if (msg.words && msg.words.length > 0) {
                updateWords(msg.words);
            }
            
            updateHandStatus(msg.hands_visible);
            break;

        case 'sequence_ready':
            DOM.collectorPhase.textContent = `Đủ ${msg.words.length} từ — đang ghép câu…`;
            break;

        case 'llm_start':
            DOM.llmStatus.textContent = 'LLM: đang ghép câu…';
            DOM.llmOutput.innerHTML = '<p class="llm-placeholder">Đang xử lý...</p>';
            break;

        case 'llm_done':
            DOM.llmStatus.textContent = 'LLM: xong ✓';
            DOM.llmOutput.innerHTML = `<p class="llm-sentence">${escapeHtml(msg.text)}</p>`;
            break;

        case 'tts_start':
            DOM.ttsStatus.textContent = 'Audio: đang tổng hợp giọng nói…';
            break;

        case 'tts_audio':
            DOM.ttsStatus.textContent = 'Audio: đang phát…';
            
            break;

        case 'cycle_reset':
            DOM.ttsStatus.textContent = 'Audio: xong ✓';
            DOM.audioWave.classList.remove('playing');
            DOM.collectorPhase.textContent = 'Ký từ 1/5 — mỗi từ giữ ~3 giây';
            clearWords();
            setTimeout(() => {
                DOM.llmOutput.innerHTML = '<p class="llm-placeholder">Câu sẽ xuất hiện ở đây...</p>';
                DOM.llmStatus.textContent = 'LLM: idle';
                DOM.ttsStatus.textContent = 'Audio: idle';
            }, 2000);
            break;

        case 'error':
            console.warn('[Server Error]', msg.message);
            break;
    }
}

function handleAudioBlob(blob) {
    const audioBlob = new Blob([blob], { type: 'audio/mpeg' });
    const url = URL.createObjectURL(audioBlob);
    if (currentAudio) {
        currentAudio.pause();
        URL.revokeObjectURL(currentAudio.src);
    }
    currentAudio = new Audio(url);
    DOM.audioWave.classList.add('playing');
    DOM.ttsStatus.textContent = 'Audio: đang phát…';

    currentAudio.play().catch(err => {
        console.error('[Audio] Playback failed:', err);
        DOM.ttsStatus.textContent = 'Audio: lỗi phát — tap để cho phép';
    });

    currentAudio.onended = () => {
        DOM.audioWave.classList.remove('playing');
        DOM.ttsStatus.textContent = 'Audio: xong ✓';
        URL.revokeObjectURL(url);
    };
}

function updatePredictions(top5, bestP, bestWord) {
    const rows = DOM.predictionRows;
    for (let i = 0; i < 5; i++) {
        const row = rows[i];
        const nameEl = row.querySelector('.pred-name');
        const fillEl = row.querySelector('.pred-bar-fill');
        const pctEl = row.querySelector('.pred-pct');

        if (i < top5.length) {
            const [name, prob] = top5[i];
            nameEl.textContent = name;
            fillEl.style.width = `${Math.round(prob * 100)}%`;
            pctEl.textContent = `${Math.round(prob * 100)}%`;
        } else {
            nameEl.textContent = '—';
            fillEl.style.width = '0%';
            pctEl.textContent = '0%';
        }
    }
}

function updateWords(words) {
    const maxWords = serverConfig ? serverConfig.max_words : 5;
    DOM.wordChips.innerHTML = '';
    for (let i = 0; i < maxWords; i++) {
        const chip = document.createElement('span');
        if (i < words.length) {
            chip.className = 'word-chip';
            chip.textContent = words[i];
        } else {
            chip.className = 'word-chip empty-slot';
            chip.textContent = `Từ ${i + 1}`;
        }
        DOM.wordChips.appendChild(chip);
    }

    
    const progress = (words.length / maxWords) * 100;
    DOM.wordProgressFill.style.width = `${progress}%`;
}

function clearWords() {
    const maxWords = serverConfig ? serverConfig.max_words : 5;
    DOM.wordChips.innerHTML = '';
    for (let i = 0; i < maxWords; i++) {
        const chip = document.createElement('span');
        chip.className = 'word-chip empty-slot';
        chip.textContent = `Từ ${i + 1}`;
        DOM.wordChips.appendChild(chip);
    }
    DOM.wordProgressFill.style.width = '0%';
}

function flashWordLock(display) {
    DOM.collectorPhase.style.color = '#22c55e';
    setTimeout(() => {
        DOM.collectorPhase.style.color = '';
    }, 500);
}

function updateHandStatus(visible) {
    DOM.handStatus.classList.toggle('visible', true);
    DOM.handStatus.classList.toggle('detected', visible);
    DOM.handIcon.textContent = visible ? '✋' : '👋';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function startPipelineAnimation() {
    if (pipelineInterval) return;
    pipelineInterval = setInterval(() => {
        DOM.pipelineStages.forEach((el, i) => {
            el.classList.toggle('active', i === pipelineStage);
        });
        pipelineStage = (pipelineStage + 1) % DOM.pipelineStages.length;
    }, 120);
}

function stopPipelineAnimation() {
    if (pipelineInterval) {
        clearInterval(pipelineInterval);
        pipelineInterval = null;
    }
    DOM.pipelineStages.forEach(el => el.classList.remove('active'));
}

async function startCamera() {
    if (isRunning) return;

    try {
        handProcessor = new HandProcessor({
            width: 640,
            height: 480,
            smoothAlpha: 0.35,
            mode: serverConfig ? serverConfig.mp_mode : 'hands',
            onResult: onHandResult,
        });

        await handProcessor.initialize(DOM.videoInput, DOM.canvasOutput);
        handProcessor.start();

        isRunning = true;
        DOM.cameraPlaceholder.classList.add('hidden');
        DOM.btnStart.disabled = true;
        DOM.btnStop.disabled = false;
        DOM.btnReset.disabled = false;
        DOM.btnForceLlm.disabled = false;
        DOM.fpsBadge.style.display = '';

        startPipelineAnimation();
        clearWords();

        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'reset' }));
        }

    } catch (err) {
        console.error('[Camera] Failed to start:', err);
        alert('Không thể mở camera. Hãy cho phép quyền truy cập camera.\n\n' + err.message);
    }
}

function stopCamera() {
    if (!isRunning) return;

    if (handProcessor) {
        handProcessor.stop();
        handProcessor = null;
    }

    isRunning = false;
    DOM.cameraPlaceholder.classList.remove('hidden');
    DOM.btnStart.disabled = false;
    DOM.btnStop.disabled = true;
    DOM.btnReset.disabled = true;
    DOM.btnForceLlm.disabled = true;

    stopPipelineAnimation();

    
    const ctx = DOM.canvasOutput.getContext('2d');
    ctx.clearRect(0, 0, DOM.canvasOutput.width, DOM.canvasOutput.height);
}

function onHandResult(result) {
    
    DOM.fpsText.textContent = `${result.fps} FPS`;

    if (appMode === 'inference') {
        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'frame',
                vector: result.vector,
                hands_visible: result.handsVisible,
                lang: DOM.langSelect.value,
            }));
        }
    } else if (appMode === 'training') {
        
        if ((isRecording || isFewShotRecording) && result.handsVisible && result.vector) {
            recordedFrames.push(result.vector);
            const progress = (recordedFrames.length / MAX_RECORD_FRAMES) * 100;
            DOM.collectProgressFill.style.width = `${progress}%`;
            DOM.collectFramesBadge.textContent = `${recordedFrames.length} / ${MAX_RECORD_FRAMES} frames`;

            if (recordedFrames.length >= MAX_RECORD_FRAMES) {
                if (isFewShotRecording) {
                    stopFewShotAndSave();
                } else {
                    stopRecordingAndSave();
                }
            }
        }
    }
}

DOM.btnStart.addEventListener('click', startCamera);
DOM.btnStop.addEventListener('click', stopCamera);

DOM.btnReset.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'reset' }));
    }
    clearWords();
    DOM.llmOutput.innerHTML = '<p class="llm-placeholder">Câu sẽ xuất hiện ở đây...</p>';
    DOM.llmStatus.textContent = 'LLM: idle';
    DOM.ttsStatus.textContent = 'Audio: idle';
    DOM.collectorPhase.textContent = 'Đã reset — ký từ 1/5 trong 3 giây';
});

DOM.btnForceLlm.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'force_llm',
            lang: DOM.langSelect.value,
        }));
    }
});

DOM.btnModeInference.addEventListener('click', () => {
    appMode = 'inference';
    DOM.btnModeInference.classList.add('btn-active');
    DOM.btnModeTraining.classList.remove('btn-active');
    DOM.btnModeV2S.classList.remove('btn-active');
    DOM.inferenceSection.classList.remove('hidden');
    DOM.trainingSection.classList.add('hidden');
    DOM.v2sSection.classList.add('hidden');
});

DOM.btnModeTraining.addEventListener('click', () => {
    appMode = 'training';
    DOM.btnModeTraining.classList.add('btn-active');
    DOM.btnModeInference.classList.remove('btn-active');
    DOM.btnModeV2S.classList.remove('btn-active');
    DOM.trainingSection.classList.remove('hidden');
    DOM.inferenceSection.classList.add('hidden');
    DOM.v2sSection.classList.add('hidden');
    loadVocab();
});

DOM.btnModeV2S.addEventListener('click', () => {
    appMode = 'v2s';
    DOM.btnModeV2S.classList.add('btn-active');
    DOM.btnModeInference.classList.remove('btn-active');
    DOM.btnModeTraining.classList.remove('btn-active');
    DOM.v2sSection.classList.remove('hidden');
    DOM.inferenceSection.classList.add('hidden');
    DOM.trainingSection.classList.add('hidden');
    
    
    if (isRunning) stopCamera();
    
    
    if (window.avatarRenderer) {
        setTimeout(() => window.avatarRenderer.onWindowResize(), 50);
    }
});

let recognition = null;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRec();
    recognition.continuous = false;
    recognition.interimResults = true;
    
    recognition.onstart = () => {
        DOM.btnMicStart.innerHTML = '<span class="btn-icon">🔴</span> Đang nghe...';
        DOM.btnMicStart.classList.replace('btn-primary', 'btn-danger');
        DOM.v2sTranscript.textContent = '...';
        DOM.v2sGlosses.textContent = 'Đang chờ dữ liệu...';
    };
    
    recognition.onresult = (event) => {
        let text = '';
        for (let i = 0; i < event.results.length; i++) {
            text += event.results[i][0].transcript;
        }
        DOM.v2sTranscript.textContent = text;
        DOM.v2sTranscript.classList.remove('llm-placeholder');
        
        if (event.results[0].isFinal) {
            
            const words = text.toLowerCase().replace(/[.,!?]/g, '').split(' ');
            
            const stopWords = ['là', 'thì', 'mà', 'rằng', 'của', 'đang', 'đã', 'sẽ', 'những', 'các'];
            const glosses = words.filter(w => !stopWords.includes(w) && w.trim().length > 0);
            
            DOM.v2sGlosses.innerHTML = `Glosses: <strong style="color:var(--accent-green)">[${glosses.join('] -> [')}]</strong>`;
            
            
            if (window.avatarRenderer) {
                window.avatarRenderer.playSequence(glosses);
            }
        }
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        DOM.btnMicStart.innerHTML = '<span class="btn-icon">🎤</span> Lỗi: ' + event.error + ' (Chạm thử lại)';
        DOM.btnMicStart.classList.replace('btn-danger', 'btn-primary');
    };
    
    recognition.onend = () => {
        DOM.btnMicStart.innerHTML = '<span class="btn-icon">🎤</span> Chạm để nói lại';
        DOM.btnMicStart.classList.replace('btn-danger', 'btn-primary');
    };
} else {
    DOM.btnMicStart.disabled = true;
    DOM.btnMicStart.textContent = 'Trình duyệt không hỗ trợ Web Speech API';
}

DOM.btnMicStart.addEventListener('click', () => {
    if (!recognition) return;
    try {
        recognition.lang = DOM.langSelect.value === 'en' ? 'en-US' : 'vi-VN';
        recognition.start();
    } catch (e) {
        
        recognition.stop();
    }
});

async function loadVocab() {
    try {
        const res = await fetch('/api/vocab?t=' + Date.now());
        const data = await res.json();
        renderLabelsTable(data);
    } catch (e) {
        console.error('Failed to load vocab', e);
    }
}

function renderLabelsTable(data) {
    DOM.labelsTbody.innerHTML = '';
    const classes = data.classes || [];
    classes.forEach(slug => {
        const name = data.labels_vi[slug] || slug;
        const count = data.counts[slug] || 0;
        
        const tr = document.createElement('tr');
        if (slug === selectedLabelSlug) {
            tr.classList.add('selected');
        }
        
        tr.innerHTML = `
            <td><strong>${name}</strong> <small style="color:var(--text-muted)">(${slug})</small></td>
            <td><span class="badge">${count}</span></td>
            <td>
                <button class="btn-delete" data-slug="${slug}">Xóa</button>
            </td>
        `;
        
        tr.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') return;
            selectedLabelSlug = slug;
            DOM.collectLabelName.textContent = name;
            DOM.btnRecord.disabled = !isRunning;
            DOM.btnFewShot.disabled = !isRunning;
            loadVocab(); 
        });
        
        const btnDelete = tr.querySelector('.btn-delete');
        btnDelete.addEventListener('click', async () => {
            if (confirm(`Bạn có chắc muốn xóa nhãn "${name}" và toàn bộ dữ liệu của nó?`)) {
                await fetch(`/api/vocab/${slug}`, { method: 'DELETE' });
                if (selectedLabelSlug === slug) {
                    selectedLabelSlug = null;
                    DOM.collectLabelName.textContent = 'Chưa chọn nhãn';
                    DOM.btnRecord.disabled = true;
                    DOM.btnFewShot.disabled = true;
                }
                loadVocab();
            }
        });
        
        DOM.labelsTbody.appendChild(tr);
    });
}

DOM.btnAddLabel.addEventListener('click', async () => {
    const name = DOM.newLabelInput.value.trim();
    if (!name) return;
    
    try {
        await fetch('/api/vocab', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name })
        });
        DOM.newLabelInput.value = '';
        loadVocab();
    } catch (e) {
        alert('Lỗi thêm nhãn');
    }
});

DOM.btnRecord.addEventListener('click', () => {
    if (!selectedLabelSlug || !isRunning) return;
    
    if (!isRecording) {
        isRecording = true;
        recordedFrames = [];
        DOM.btnRecord.innerHTML = '<span class="btn-icon">⏹</span> Đang ghi...';
        DOM.btnRecord.classList.remove('btn-danger');
        DOM.btnRecord.classList.add('btn-primary');
        DOM.collectProgressFill.style.width = '0%';
    } else {
        
        isRecording = false;
        DOM.btnRecord.innerHTML = '<span class="btn-icon">⏺</span> Bắt đầu Record';
        DOM.btnRecord.classList.remove('btn-primary');
        DOM.btnRecord.classList.add('btn-danger');
        DOM.collectProgressFill.style.width = '0%';
        DOM.collectFramesBadge.textContent = `0 / ${MAX_RECORD_FRAMES} frames`;
    }
});

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && appMode === 'training' && selectedLabelSlug && isRunning) {
        if (document.activeElement.tagName === 'INPUT') return;
        e.preventDefault();
        DOM.btnRecord.click();
    }
});

async function stopRecordingAndSave() {
    isRecording = false;
    DOM.btnRecord.innerHTML = '<span class="btn-icon">⌛</span> Đang lưu...';
    DOM.btnRecord.disabled = true;
    
    try {
        const res = await fetch('/api/collect', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                slug: selectedLabelSlug,
                landmarks: recordedFrames
            })
        });
        const data = await res.json();
        console.log('Saved', data);
        loadVocab(); 
    } catch (e) {
        console.error('Save failed', e);
        alert('Lỗi khi lưu dữ liệu');
    } finally {
        DOM.btnRecord.innerHTML = '<span class="btn-icon">⏺</span> Bắt đầu Record';
        DOM.btnRecord.classList.remove('btn-primary');
        DOM.btnRecord.classList.add('btn-danger');
        DOM.btnRecord.disabled = false;
        DOM.collectProgressFill.style.width = '0%';
        DOM.collectFramesBadge.textContent = `0 / ${MAX_RECORD_FRAMES} frames`;
    }
}

DOM.btnFewShot.addEventListener('click', () => {
    if (!selectedLabelSlug || !isRunning) return;
    
    if (!isFewShotRecording) {
        isFewShotRecording = true;
        recordedFrames = [];
        DOM.btnFewShot.innerHTML = '<span class="btn-icon">🧠</span> Đang học...';
        DOM.collectProgressFill.style.width = '0%';
        DOM.btnRecord.disabled = true;
    } else {
        isFewShotRecording = false;
        DOM.btnFewShot.innerHTML = '<span class="btn-icon">🧠</span> Học nhanh (Few-Shot)';
        DOM.collectProgressFill.style.width = '0%';
        DOM.collectFramesBadge.textContent = `0 / ${MAX_RECORD_FRAMES} frames`;
        DOM.btnRecord.disabled = false;
    }
});

async function stopFewShotAndSave() {
    isFewShotRecording = false;
    DOM.btnFewShot.innerHTML = '<span class="btn-icon">⌛</span> Đang xử lý...';
    DOM.btnFewShot.disabled = true;
    
    try {
        const res = await fetch('/api/fewshot', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                slug: selectedLabelSlug,
                landmarks: recordedFrames
            })
        });
        const data = await res.json();
        console.log('Few-shot Saved', data);
        alert('Đã học xong từ mới! Bạn có thể sang tab Nhận diện để thử ngay lập tức.');
    } catch (e) {
        console.error('Few-shot failed', e);
        alert('Lỗi khi học từ mới');
    } finally {
        DOM.btnFewShot.innerHTML = '<span class="btn-icon">🧠</span> Học nhanh (Few-Shot)';
        DOM.btnFewShot.disabled = false;
        DOM.btnRecord.disabled = false;
        DOM.collectProgressFill.style.width = '0%';
        DOM.collectFramesBadge.textContent = `0 / ${MAX_RECORD_FRAMES} frames`;
    }
}

DOM.btnTrain.addEventListener('click', () => {
    if (DOM.btnTrain.disabled) return;
    
    DOM.btnTrain.disabled = true;
    DOM.trainConsole.innerHTML = '> Đang kết nối tới trình huấn luyện...\n';
    
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsTrain = new WebSocket(`${proto}//${location.host}/ws/train`);
    
    wsTrain.onopen = () => {
        DOM.trainConsole.innerHTML += '> Bắt đầu huấn luyện...\n';
    };
    
    wsTrain.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'log') {
            DOM.trainConsole.innerHTML += msg.message + '\n';
            DOM.trainConsole.scrollTop = DOM.trainConsole.scrollHeight;
        } else if (msg.type === 'error') {
            DOM.trainConsole.innerHTML += `<span style="color:var(--accent-rose)">${msg.message}</span>\n`;
            DOM.trainConsole.scrollTop = DOM.trainConsole.scrollHeight;
        } else if (msg.type === 'reload_success') {
            DOM.trainConsole.innerHTML += '<span style="color:var(--accent-cyan)">🔄 PWA đã nạp lại mô hình mới nhất! Bạn có thể sang thẻ Nhận diện để thử ngay.</span>\n';
            DOM.trainConsole.scrollTop = DOM.trainConsole.scrollHeight;
        }
    };
    
    wsTrain.onclose = () => {
        DOM.trainConsole.innerHTML += '> [Kết nối WebSocket đã đóng]\n';
        DOM.btnTrain.disabled = false;
    };
});

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then(reg => console.log('[SW] Registered:', reg.scope))
        .catch(err => console.warn('[SW] Registration failed:', err));
}

window.addEventListener('load', () => {
    if (typeof AvatarRenderer !== 'undefined') {
        window.avatarRenderer = new AvatarRenderer('avatar-container');
    }
});

connectWebSocket();
clearWords();
