document.addEventListener('DOMContentLoaded', () => {

    // UI Elements
    const fileInput = document.getElementById('video-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const form = document.getElementById('pipeline-form');

    const loadingIndicator = document.getElementById('loading-indicator');
    const statusMessage = document.getElementById('upload-status');
    const dashboardSection = document.getElementById('dashboard-section');

    const btnSimilarity = document.getElementById('btn-similarity');
    const similarityResults = document.getElementById('similarity-results');
    const similarityGrid = document.getElementById('similarity-grid');

    const btnPersonas = document.getElementById('btn-personas');
    const personaResults = document.getElementById('persona-results');
    const personaGrid = document.getElementById('persona-grid');
    const subredditHero = document.getElementById('subreddit-hero');
    const subredditHeroName = document.getElementById('subreddit-hero-name');

    // Check for existing fingerprint on page load
    async function checkExistingState() {
        try {
            const res = await fetch('/api/check_status');
            if (res.ok) {
                const data = await res.json();
                if (data.has_fingerprint) {
                    // Reveal the dashboard immediately without re-uploading
                    dashboardSection.classList.remove('hidden');
                    statusMessage.textContent = 'Existing Fingerprint Found! You can analyze directly.';
                    statusMessage.classList.add('status-success');
                }
            }
        } catch (e) {
            console.log("Could not poll server state.");
        }
    }
    checkExistingState();

    // Make file upload look nice
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
            fileNameDisplay.style.color = 'var(--accent-hover)';
        }
    });

    // Handle initial pipeline submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (fileInput.files.length === 0) return;

        // Build concepts configuration
        const conceptsConfig = {
            object_categories: document.getElementById('object-categories').value.split(',').map(s => s.trim()),
            scene_categories: document.getElementById('scene-categories').value.split(',').map(s => s.trim()),
            custom_actions: document.getElementById('custom-actions').value.split(',').map(s => s.trim()),
            custom_themes: document.getElementById('custom-themes').value.split(',').map(s => s.trim())
        };

        const formData = new FormData();
        formData.append('video', fileInput.files[0]);
        formData.append('concepts', JSON.stringify(conceptsConfig));

        // UI Reset
        statusMessage.textContent = '';
        statusMessage.className = 'status-message';
        loadingIndicator.classList.remove('hidden');
        document.getElementById('btn-fingerprint').disabled = true;
        dashboardSection.classList.add('hidden');
        similarityResults.classList.add('hidden');
        personaResults.classList.add('hidden');

        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        progressBar.style.width = '0%';
        progressText.textContent = 'Initializing...';

        let progressInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/progress');
                if (res.ok) {
                    const data = await res.json();

                    // Don't auto-clear if it was stuck at 100 from previous video upload,
                    // wait for the server to actually reset it to 0 before we start tracking again.
                    if (data.percent === 100 && progressBar.style.width === '0%') {
                        return; // waiting for backend wipe
                    }

                    progressBar.style.width = data.percent + '%';
                    progressText.textContent = `${data.status} (${data.percent}%)`;

                    // If it genuinely hits 100 during our tracking, clear it.
                    if (data.percent >= 100 && parseInt(progressBar.style.width) > 0) {
                        clearInterval(progressInterval);
                    }
                }
            } catch (err) {
                // Ignore fetch errors during polling
            }
        }, 1500);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                statusMessage.textContent = 'Fingerprint successfully generated!';
                statusMessage.classList.add('status-success');
                dashboardSection.classList.remove('hidden');
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.classList.add('status-error');
        } finally {
            if (progressInterval) clearInterval(progressInterval);
            loadingIndicator.classList.add('hidden');
            document.getElementById('btn-fingerprint').disabled = false;
        }
    });

    // Handle Similarity Analysis
    btnSimilarity.addEventListener('click', async () => {
        const originalText = btnSimilarity.innerHTML;
        btnSimilarity.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        btnSimilarity.disabled = true;

        try {
            const response = await fetch('/api/similar');
            const data = await response.json();

            if (response.ok) {
                similarityGrid.innerHTML = ''; // Clear old

                data.similar_movies.forEach(movie => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `
                        <h4>${movie.title} (${movie.year || 'Unknown'})</h4>
                        <div class="score"><i class="fa-solid fa-fire"></i> Match: ${((movie.similarity_score || 0) * 100).toFixed(1)}%</div>
                        <div class="meta"><strong>Vis Match:</strong> ${((movie._visual_score || 0) * 100).toFixed(1)}%</div>
                        <div class="meta"><strong>Desc Match:</strong> ${((movie._keyword_score || 0) * 100).toFixed(1)}%</div>
                        <div class="meta" style="margin-top: 0.5rem"><em>${movie.genres ? movie.genres.join(', ') : 'Unknown Genre'}</em></div>
                    `;
                    similarityGrid.appendChild(card);
                });

                similarityResults.classList.remove('hidden');
            } else {
                alert(data.error || "Failed to load similar movies");
            }

        } catch (error) {
            console.error("API Error", error);
            alert("API connection failed.");
        } finally {
            btnSimilarity.innerHTML = originalText;
            btnSimilarity.disabled = false;
        }
    });

    // Handle Persona Generation
    btnPersonas.addEventListener('click', async () => {
        const originalText = btnPersonas.innerHTML;
        btnPersonas.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Extracting...';
        btnPersonas.disabled = true;

        try {
            const response = await fetch('/api/personas');
            const data = await response.json();

            if (response.ok) {
                personaGrid.innerHTML = ''; // Clear old

                // Parse and display TOP subreddit in hero (from #1 persona)
                const topPersona = data.personas[0];
                const topSubreddits = topPersona?.demographics?.subreddits
                    ? String(topPersona.demographics.subreddits).split(',').map(s => s.trim()).filter(Boolean)
                    : [];
                const finalSubreddit = topSubreddits[0] || 'r/movies';

                if (finalSubreddit) {
                    subredditHeroName.textContent = finalSubreddit.startsWith('r/') ? finalSubreddit : 'r/' + finalSubreddit;
                    subredditHero.classList.remove('hidden');
                }

                data.personas.forEach((p, idx) => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    // The first 3 are primary matches, the rest get dimmer
                    if (idx > 2) card.style.opacity = "0.6";

                    const subredditStr = p.demographics?.subreddits || '';
                    const subredditList = subredditStr.split(',').map(s => s.trim()).filter(Boolean);
                    const subredditPillsHtml = subredditList.map((sr, i) => {
                        const displayName = sr.startsWith('r/') ? sr : 'r/' + sr;
                        const isPrimary = idx === 0 && i === 0;
                        return `<span class="subreddit-pill ${isPrimary ? 'primary' : ''}"><i class="fa-brands fa-reddit"></i>${displayName}</span>`;
                    }).join('');

                    card.innerHTML = `
                        <h4>${p.persona}</h4>
                        <div class="score"><i class="fa-solid fa-bullseye"></i> Affinity: ${(p.affinity_score * 100).toFixed(1)}%</div>
                        <div class="meta"><i class="fa-solid fa-users"></i> Size: ${p.size} Sampled Responses</div>
                        
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--panel-border)">
                            <strong style="display:block; margin-bottom: 0.5rem; color: var(--text-secondary);">Target Subreddits</strong>
                            <div class="subreddit-pills">${subredditPillsHtml || '<span class="subreddit-pill">N/A</span>'}</div>
                        </div>

                        <div class="tag-container">
                            ${p.top_keywords.map(kw => `<span class="tag">${kw}</span>`).join('')}
                        </div>
                    `;
                    personaGrid.appendChild(card);
                });

                personaResults.classList.remove('hidden');
            } else {
                alert(data.error || "Failed to load personas");
            }

        } catch (error) {
            console.error("API Error", error);
            alert("API connection failed.");
        } finally {
            btnPersonas.innerHTML = originalText;
            btnPersonas.disabled = false;
        }
    });
});
