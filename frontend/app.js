document.addEventListener('DOMContentLoaded', () => {
    // DOM Element References
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');

    const sampleFakeBtn = document.getElementById('sampleFakeBtn');
    const sampleRealBtn = document.getElementById('sampleRealBtn');
    const sampleArtBtn = document.getElementById('sampleArtBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');

    const progressBox = document.getElementById('progressBox');
    const progressStep = document.getElementById('progressStep');
    const progressBarFill = document.getElementById('progressBarFill');

    const emptyState = document.getElementById('emptyState');
    const resultsContent = document.getElementById('resultsContent');

    const resultCard = document.getElementById('resultCard');
    const predictionBadge = document.getElementById('predictionBadge');
    const confidencePercentage = document.getElementById('confidencePercentage');
    const gaugeFill = document.getElementById('gaugeFill');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');

    const barFakeFill = document.getElementById('barFakeFill');
    const barRealFill = document.getElementById('barRealFill');
    const probFakeText = document.getElementById('probFakeText');
    const probRealText = document.getElementById('probRealText');

    const evidenceList = document.getElementById('evidenceList');
    const evidencePredType = document.getElementById('evidencePredType');

    const visualizerGrid = document.getElementById('visualizerGrid');
    const visCrop = document.getElementById('visCrop');
    const visHeatmap = document.getElementById('visHeatmap');
    const visOverlay = document.getElementById('visOverlay');
    const visFFT = document.getElementById('visFFT');
    const visTabs = document.querySelectorAll('.vis-tab');

    const metricFaceConf = document.getElementById('metricFaceConf');
    const metricModelConf = document.getElementById('metricModelConf');
    const metricFreqScore = document.getElementById('metricFreqScore');
    const metricRegion = document.getElementById('metricRegion');

    const historyTableBody = document.getElementById('historyTableBody');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
    const historySearchInput = document.getElementById('historySearchInput');
    const exportHistoryBtn = document.getElementById('exportHistoryBtn');

    const copyReportBtn = document.getElementById('copyReportBtn');
    const exportReportBtn = document.getElementById('exportReportBtn');
    const resetScanBtn = document.getElementById('resetScanBtn');

    const statusText = document.getElementById('statusText');
    const deviceBadge = document.getElementById('deviceBadge');

    // Lightbox modal elements
    const imageModal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const modalTitle = document.getElementById('modalTitle');
    const modalDownloadBtn = document.getElementById('modalDownloadBtn');
    const closeModalBtn = document.getElementById('closeModalBtn');

    // Accent buttons
    const accentBtns = document.querySelectorAll('.accent-btn');

    let currentFile = null;
    let currentAnalysisData = null;
    let cachedHistory = [];

    // ── 1. Initial Setup & Health Check ──────────────────────────────────────
    checkSystemHealth();
    loadHistory();
    initThemePicker();
    initLightbox();
    initVisTabs();

    async function checkSystemHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                statusText.textContent = "Neural Engines Active";
                deviceBadge.textContent = (data.device || "CPU").toUpperCase();
            }
        } catch (e) {
            statusText.textContent = "Backend Offline";
            deviceBadge.textContent = "ERR";
        }
    }

    // ── 2. Theme Accent Picker ──────────────────────────────────────────────
    function initThemePicker() {
        accentBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                accentBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const accent = btn.getAttribute('data-accent');
                document.body.setAttribute('data-accent', accent);
            });
        });
    }

    // ── 3. Drag & Drop File Handlers ─────────────────────────────────────────
    browseBtn.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', (e) => {
        if (e.target === dropzone || e.target.closest('#dropzoneContent')) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP, BMP).');
            return;
        }
        if (file.size > 15 * 1024 * 1024) {
            alert('File size exceeds 15MB limit.');
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropzoneContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearFileSelection() {
        currentFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.classList.add('hidden');
        dropzoneContent.classList.remove('hidden');
        analyzeBtn.disabled = true;
    }

    // ── 4. Sample Images Handlers ────────────────────────────────────────────
    sampleFakeBtn.addEventListener('click', async () => {
        await loadSampleImage('/outputs/heatmaps/sample_deepfake.jpg', 'sample_deepfake.jpg');
    });

    sampleRealBtn.addEventListener('click', async () => {
        await loadSampleImage('/outputs/heatmaps/sample_real.jpg', 'sample_real.jpg');
    });

    if (sampleArtBtn) {
        sampleArtBtn.addEventListener('click', async () => {
            await loadSampleImage('/outputs/heatmaps/sample_ai_art.jpg', 'sample_tiger_ai_art.jpg');
        });
    }

    async function loadSampleImage(url, filename) {
        try {
            const res = await fetch(url);
            if (!res.ok) {
                createBlobFromSample(filename);
                return;
            }
            const blob = await res.blob();
            const file = new File([blob], filename, { type: 'image/jpeg' });
            handleSelectedFile(file);
        } catch (e) {
            createBlobFromSample(filename);
        }
    }

    function createBlobFromSample(filename) {
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 256;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = filename.includes('fake') ? '#2e1017' : '#0f291e';
        ctx.fillRect(0, 0, 256, 256);
        ctx.fillStyle = '#ffccaa';
        ctx.beginPath(); ctx.arc(128, 128, 70, 0, Math.PI * 2); ctx.fill();
        canvas.toBlob((blob) => {
            const file = new File([blob], filename, { type: 'image/jpeg' });
            handleSelectedFile(file);
        }, 'image/jpeg');
    }

    // ── 5. Analyze Image Execution ───────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        analyzeBtn.disabled = true;
        progressBox.classList.remove('hidden');
        updateProgress(20, "Extracting features & spatial domains...");

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            setTimeout(() => updateProgress(45, "Running Neural Ensemble (ViT + Swin)..."), 300);
            setTimeout(() => updateProgress(75, "Computing Grad-CAM & 2D-FFT noise spectrum..."), 650);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            updateProgress(100, "Synthesizing forensic diagnostic report...");

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Analysis failed');
            }

            const data = await response.json();
            currentAnalysisData = data;

            renderAnalysisResults(data);
            loadHistory();

        } catch (err) {
            alert(`Analysis Error: ${err.message}`);
        } finally {
            setTimeout(() => {
                progressBox.classList.add('hidden');
                analyzeBtn.disabled = false;
            }, 450);
        }
    });

    function updateProgress(pct, message) {
        progressBarFill.style.width = `${pct}%`;
        progressStep.textContent = message;
    }

    // ── 6. Render Results ────────────────────────────────────────────────────
    function renderAnalysisResults(data) {
        emptyState.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        const primary = data.primary_result;
        const isFake = primary.prediction === 'DEEPFAKE' || primary.prediction === 'AI-GENERATED';
        const isUnknown = primary.prediction === 'UNKNOWN';

        // Result Card Styling
        resultCard.className = `result-card ${isFake ? 'is-fake' : isUnknown ? 'is-unknown' : 'is-real'}`;
        predictionBadge.className = `prediction-tag ${isFake ? 'fake' : isUnknown ? 'unknown' : 'real'}`;
        
        let displayBadge = primary.prediction;
        if (primary.prediction === 'DEEPFAKE') displayBadge = 'SYNTHETIC DEEPFAKE';
        else if (primary.prediction === 'AI-GENERATED') displayBadge = 'AI-GENERATED ART';
        else if (primary.prediction === 'REAL') displayBadge = 'AUTHENTIC REAL';
        else if (isUnknown) displayBadge = 'NO SUBJECT DETECTED';
        
        predictionBadge.textContent = displayBadge;

        const conf = primary.confidence;
        confidencePercentage.textContent = isUnknown ? 'N/A' : `${conf}%`;
        gaugeFill.setAttribute('stroke-dasharray', `${isUnknown ? 0 : conf}, 100`);

        resultTitle.textContent = isUnknown ? 'SUBJECT RECOGNITION FAILED' : `VERDICT: ${displayBadge}`;
        
        const typeInfo = primary.image_type ? `[${primary.image_type}] ` : '';
        const modelInfo = primary.model_used ? ` • Active Engine: ${primary.model_used.split('/').pop()}` : '';
        resultSubtitle.textContent = isUnknown
            ? 'Please upload an image with a visible target for verification'
            : `${typeInfo}Certainty: ${conf}%${modelInfo}`;

        barFakeFill.style.width = isUnknown ? '0%' : `${primary.fake_probability}%`;
        barRealFill.style.width = isUnknown ? '0%' : `${primary.real_probability}%`;
        probFakeText.textContent = isUnknown ? 'N/A' : `${primary.fake_probability}%`;
        probRealText.textContent = isUnknown ? 'N/A' : `${primary.real_probability}%`;

        // Diagnostic Findings
        evidencePredType.textContent = isFake ? "Threat Findings" : isUnknown ? "Notice" : "Authenticity Indicators";
        evidenceList.innerHTML = '';

        if (primary.explanation_reasons && primary.explanation_reasons.length > 0) {
            primary.explanation_reasons.forEach(reason => {
                const li = document.createElement('li');
                li.className = `evidence-item ${isFake ? 'critical' : isUnknown ? 'warning' : 'medium'}`;

                const sevBadge = document.createElement('span');
                sevBadge.className = `badge-sev ${isFake ? 'critical' : isUnknown ? 'warning' : 'info'}`;
                sevBadge.textContent = isFake ? "Flagged" : isUnknown ? "Notice" : "Verified";

                const textSpan = document.createElement('span');
                textSpan.textContent = reason;

                li.appendChild(sevBadge);
                li.appendChild(textSpan);
                evidenceList.appendChild(li);
            });
        }

        // Grad-CAM Visualizations
        const urls = primary.heatmap_urls;
        if (urls && urls.crop) {
            visCrop.src = urls.crop;
            visHeatmap.src = urls.heatmap;
            visOverlay.src = urls.overlay;
            visFFT.src = urls.fft || urls.crop;
        }

        // Telemetry Diagnostics Tiles
        metricFaceConf.textContent = isUnknown ? '0.0%' : `${primary.face_detection_confidence}%`;
        metricModelConf.textContent = isUnknown ? 'N/A' : `${primary.confidence}%`;
        metricFreqScore.textContent = isUnknown ? 'N/A' : `${primary.frequency_anomaly_score}%`;
        metricRegion.textContent = primary.dominant_region || "Contour / Global";
    }

    // ── 7. Visualizer Interactive Tabs ────────────────────────────────────────
    function initVisTabs() {
        visTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                visTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const view = tab.getAttribute('data-view');
                const items = visualizerGrid.querySelectorAll('.vis-item');

                items.forEach(item => {
                    if (view === 'all') {
                        item.classList.remove('hidden');
                    } else if (item.getAttribute('data-type') === view) {
                        item.classList.remove('hidden');
                    } else {
                        item.classList.add('hidden');
                    }
                });
            });
        });
    }

    // ── 8. Image Lightbox Modal ──────────────────────────────────────────────
    function initLightbox() {
        const visItems = document.querySelectorAll('.vis-item');
        visItems.forEach(item => {
            item.addEventListener('click', () => {
                const img = item.querySelector('img');
                if (!img || !img.src) return;
                const titleText = item.querySelector('.vis-title')?.textContent || 'Visual Evidence';
                modalTitle.textContent = titleText;
                modalImg.src = img.src;
                modalDownloadBtn.href = img.src;
                imageModal.classList.remove('hidden');
            });
        });

        closeModalBtn.addEventListener('click', () => imageModal.classList.add('hidden'));
        imageModal.addEventListener('click', (e) => {
            if (e.target === imageModal) imageModal.classList.add('hidden');
        });
    }

    // ── 9. Action Buttons (Copy, Export JSON, Reset) ──────────────────────────
    if (copyReportBtn) {
        copyReportBtn.addEventListener('click', () => {
            if (!currentAnalysisData) return;
            const p = currentAnalysisData.primary_result;
            const summary = [
                `--- SENTINEL AI FORENSIC REPORT ---`,
                `File: ${currentAnalysisData.filename}`,
                `Verdict: ${p.prediction}`,
                `Confidence: ${p.confidence}%`,
                `Type: ${p.image_type || 'General'}`,
                `Engine: ${p.model_used || 'Ensemble'}`,
                `Spectral Score: ${p.frequency_anomaly_score}%`,
                `Findings:`,
                ...(p.explanation_reasons || []).map((r, i) => `  ${i + 1}. ${r}`)
            ].join('\n');

            navigator.clipboard.writeText(summary).then(() => {
                const orig = copyReportBtn.innerHTML;
                copyReportBtn.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
                setTimeout(() => copyReportBtn.innerHTML = orig, 1800);
            });
        });
    }

    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', () => {
            if (!currentAnalysisData) return;
            const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentAnalysisData, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", jsonStr);
            downloadAnchor.setAttribute("download", `forensic_report_${Date.now()}.json`);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        });
    }

    if (resetScanBtn) {
        resetScanBtn.addEventListener('click', () => {
            clearFileSelection();
            resultsContent.classList.add('hidden');
            emptyState.classList.remove('hidden');
            currentAnalysisData = null;
        });
    }

    // ── 10. History Logs with Search & Export ─────────────────────────────────
    refreshHistoryBtn.addEventListener('click', loadHistory);

    if (historySearchInput) {
        historySearchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            filterHistoryTable(query);
        });
    }

    if (exportHistoryBtn) {
        exportHistoryBtn.addEventListener('click', () => {
            if (!cachedHistory || cachedHistory.length === 0) {
                alert('No history logs to export.');
                return;
            }
            const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(cachedHistory, null, 2));
            const dl = document.createElement('a');
            dl.setAttribute("href", jsonStr);
            dl.setAttribute("download", `sentinel_audit_logs_${Date.now()}.json`);
            document.body.appendChild(dl);
            dl.click();
            dl.remove();
        });
    }

    async function loadHistory() {
        try {
            const res = await fetch('/history');
            if (!res.ok) return;
            cachedHistory = await res.json();
            renderHistoryRows(cachedHistory);
        } catch (e) {
            console.error("History fetch error:", e);
        }
    }

    function filterHistoryTable(query) {
        if (!cachedHistory) return;
        const filtered = cachedHistory.filter(item => {
            const name = (item.filename || '').toLowerCase();
            const pred = (item.prediction || '').toLowerCase();
            const region = (item.dominant_region || '').toLowerCase();
            return name.includes(query) || pred.includes(query) || region.includes(query);
        });
        renderHistoryRows(filtered);
    }

    function renderHistoryRows(items) {
        if (!items || items.length === 0) {
            historyTableBody.innerHTML = `<tr><td colspan="8" class="text-center">No past prediction history recorded yet.</td></tr>`;
            return;
        }

        historyTableBody.innerHTML = items.map(item => {
            const isFake = item.prediction === 'DEEPFAKE' || item.prediction === 'AI-GENERATED';
            const isUnknown = item.prediction === 'UNKNOWN';
            const tagClass = isFake ? 'fake' : isUnknown ? 'unknown' : 'real';
            
            let label = item.prediction;
            if (item.prediction === 'DEEPFAKE') label = 'DEEPFAKE';
            else if (item.prediction === 'AI-GENERATED') label = 'AI ART';
            else if (item.prediction === 'REAL') label = 'AUTHENTIC';

            return `
                <tr>
                    <td><span style="font-family:var(--font-mono); font-size:11px;">${item.created_at}</span></td>
                    <td><strong style="color:var(--text-primary);">${item.filename}</strong></td>
                    <td><span class="prediction-tag ${tagClass}" style="padding:2px 8px; font-size:10px;">${label}</span></td>
                    <td><strong style="font-family:var(--font-mono); color:${isFake ? 'var(--accent-fake)' : 'var(--accent-real)'};">${item.confidence}%</strong></td>
                    <td>${item.face_count}</td>
                    <td>${item.dominant_region || 'Contour'}</td>
                    <td><span style="font-family:var(--font-mono);">${item.frequency_score ? item.frequency_score + '%' : 'N/A'}</span></td>
                    <td>
                        ${item.heatmap_path ? `<a href="${item.heatmap_path}" target="_blank" class="btn-ghost-sm" style="font-size:10px; padding:3px 8px;"><i class="fa-solid fa-eye"></i> View</a>` : '—'}
                    </td>
                </tr>
            `;
        }).join('');
    }
});
