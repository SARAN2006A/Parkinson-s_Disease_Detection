document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const videoInput = document.getElementById('videoInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    const uploadCard = document.getElementById('uploadCard');
    const resultCard = document.getElementById('resultCard');
    const mainContent = document.querySelector('.app-layout');
    const taskType = document.getElementById('taskType');

    // Result elements
    const conditionBadge = document.getElementById('conditionBadge');
    const confidenceScore = document.getElementById('confidenceScore');
    const predictedScore = document.getElementById('predictedScore');
    const severityLabel = document.getElementById('severityLabel');
    const findingsList = document.getElementById('findingsList');
    const resetBtn = document.getElementById('resetBtn');

    let selectedFile = null;
    // UI Error Handling
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.remove('hidden');
    }

    function hideError() {
        errorBanner.classList.add('hidden');
    }

    let gaugeChartInstance = null;

    // Drag and Drop Handlers
    dropZone.addEventListener('click', () => videoInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    videoInput.addEventListener('change', () => {
        if (videoInput.files.length) {
            handleFile(videoInput.files[0]);
        }
    });

    function handleFile(file) {
        hideError(); // clear old errors
        if (!file.type.startsWith('video/')) {
            showError('Please upload a valid video file (MP4, AVI, MOV).');
            return;
        }
        selectedFile = file;

        // Create an object URL for the uploaded video to preview it
        const videoURL = URL.createObjectURL(file);

        // Update UI to show the video matching the new design block
        dropZone.innerHTML = `
            <div class="upload-content" style="width: 100%; height: 100%;">
                <video src="${videoURL}" controls autoplay muted loop style="max-width: 100%; max-height: 250px; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.2);"></video>
                <div style="display: flex; flex-direction: column; gap: 0.2rem; align-items: center;">
                    <span style="font-weight: 600; color: var(--success); display: flex; align-items: center; gap: 0.5rem;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                        Ready to Analyze
                    </span>
                    <small style="color: var(--text-secondary);">${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)</small>
                    <small style="color: var(--text-tertiary); margin-top: 0.5rem; text-decoration: underline; cursor: pointer;">Click here to choose a different video</small>
                </div>
            </div>
        `;
        analyzeBtn.disabled = false;
    }

    // Analysis Handler
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        hideError(); // clear any previous errors
        // UI Loading state
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('video', selectedFile);
        formData.append('task_name', taskType.value);

        try {
            // Pointing to the relative Flask API root
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed on the server.');
            }

            displayResults(data);

        } catch (error) {
            showError(`Error: ${error.message || 'Failed to connect to the server.'}`);
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
        }
    });

    function getSeverityColor(level) {
        const colors = {
            0: "#10b981",  // Emerald
            1: "#f59e0b",  // Amber
            2: "#f97316",  // Orange
            3: "#ef4444",  // Red
            4: "#991b1b"   // Dark Red
        };
        return colors[level] || "#6c757d";
    }

    function displayResults(data) {
        // Stop loading UI
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;

        // Reveal Results Card
        // The CSS grid changes column numbers based on .hidden state of resultCard.
        // We will just unhide the resultCard.
        resultCard.classList.remove('hidden');

        // Populate Data
        const color = getSeverityColor(data.severity_level);

        // Condition Badge
        conditionBadge.textContent = data.condition_text;
        if (data.is_parkinson) {
            conditionBadge.style.backgroundColor = 'var(--danger-bg)';
            conditionBadge.style.color = 'var(--danger)';
            conditionBadge.style.borderColor = 'var(--danger-border)';
        } else {
            conditionBadge.style.backgroundColor = 'var(--success-bg)';
            conditionBadge.style.color = 'var(--success)';
            conditionBadge.style.borderColor = 'var(--success-border)';
        }

        confidenceScore.textContent = `${data.confidence.toFixed(1)}%`;

        predictedScore.textContent = data.score.toFixed(1);
        severityLabel.textContent = `${data.severity_label} Severity`;
        severityLabel.style.color = color;

        // Populate Findings
        findingsList.innerHTML = '';
        if (data.findings && data.findings.length > 0) {
            data.findings.forEach(findingHTML => {
                const div = document.createElement('div');
                div.className = 'finding-item';
                div.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--info)" stroke-width="2" style="flex-shrink:0; margin-top:2px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <span>${findingHTML}</span>`;
                findingsList.appendChild(div);
            });
        } else {
            findingsList.innerHTML = `
            <div class="finding-item" style="border-color: var(--success); background: var(--success-bg);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2" style="flex-shrink:0; margin-top:2px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span>No significant motor anomalies detected.</span>
            </div>`;
        }

        // Draw Gauge Chart
        drawChart(data.score, color);
    }

    function drawChart(score, color) {
        const ctx = document.getElementById('gaugeChart').getContext('2d');

        if (gaugeChartInstance) {
            gaugeChartInstance.destroy();
        }

        // Calculate rotation for semi-circle
        // Score range is roughly 0-132, let's cap the gauge visual at 80
        const visualMax = 80;
        const normalizedScore = Math.min(score, visualMax);

        gaugeChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [normalizedScore, visualMax - normalizedScore],
                    backgroundColor: [color, 'rgba(255, 255, 255, 0.05)'],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '85%',
                plugins: {
                    tooltip: { enabled: false },
                    legend: { enabled: false }
                }
            }
        });
    }

    // Reset Handler
    resetBtn.addEventListener('click', () => {
        resultCard.classList.add('hidden');
        selectedFile = null;
        videoInput.value = '';
        analyzeBtn.disabled = true;

        dropZone.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon-wrapper">
                    <svg class="upload-icon" xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                </div>
                <h3>Upload Patient Video</h3>
                <p>Drag and drop or click to browse</p>
                <span class="file-hint">Supported formats: MP4, AVI, MOV</span>
            </div>
        `;

        if (gaugeChartInstance) {
            gaugeChartInstance.destroy();
        }
    });
});
