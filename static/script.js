// Toggle between single and batch mode
document.getElementById('singleMode').addEventListener('click', () => {
    document.getElementById('singleMode').classList.add('active');
    document.getElementById('batchMode').classList.remove('active');
    document.getElementById('singleInput').classList.add('active');
    document.getElementById('batchInput').classList.remove('active');
    clearResults();
});

document.getElementById('batchMode').addEventListener('click', () => {
    document.getElementById('batchMode').classList.add('active');
    document.getElementById('singleMode').classList.remove('active');
    document.getElementById('batchInput').classList.add('active');
    document.getElementById('singleInput').classList.remove('active');
    clearResults();
});

// Initialize reCAPTCHA if available
let recaptchaSiteKey = null;
if (typeof grecaptcha !== 'undefined') {
    // Get site key from script tag or global variable
    const scripts = document.getElementsByTagName('script');
    for (let script of scripts) {
        if (script.src && script.src.includes('recaptcha')) {
            const match = script.src.match(/render=([^&]+)/);
            if (match) {
                recaptchaSiteKey = match[1];
                break;
            }
        }
    }
}

// Get reCAPTCHA token
async function getRecaptchaToken() {
    if (!recaptchaSiteKey || typeof grecaptcha === 'undefined') {
        return null;
    }
    try {
        const token = await grecaptcha.execute(recaptchaSiteKey, {action: 'assess'});
        return token;
    } catch (error) {
        console.warn('reCAPTCHA error:', error);
        return null;
    }
}

// Generate JS token (proves JavaScript is enabled)
function generateJSToken() {
    return btoa(Date.now().toString() + Math.random().toString()).substring(0, 16);
}

// Single requirement form
document.getElementById('singleForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const requirement = document.getElementById('requirement').value.trim();
    
    if (!requirement) {
        showError('Please enter a requirement');
        return;
    }
    
    await assessRequirement(requirement);
});

// CSV file input handler
document.getElementById('csvFile').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const text = await file.text();
    const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    
    // Parse CSV - assume first column contains requirements
    const requirements = lines.slice(1).map(line => {
        const columns = line.split(',').map(col => col.trim().replace(/^"|"$/g, ''));
        return columns[0]; // First column
    }).filter(req => req.length > 0);
    
    if (requirements.length > 0) {
        document.getElementById('requirements').value = requirements.join('\n');
    }
});

// Batch requirements form
document.getElementById('batchForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const requirementsText = document.getElementById('requirements').value.trim();
    
    if (!requirementsText) {
        showError('Please enter at least one requirement or upload a CSV file');
        return;
    }
    
    const requirements = requirementsText.split('\n')
        .map(r => r.trim())
        .filter(r => r.length > 0);
    
    if (requirements.length === 0) {
        showError('Please enter at least one valid requirement');
        return;
    }
    
    await assessBatch(requirements);
});

async function assessRequirement(requirement) {
    showLoading();
    hideError();
    hideResults();
    
    try {
        // Get reCAPTCHA token and JS token
        const recaptchaToken = await getRecaptchaToken();
        const jsToken = generateJSToken();
        
        const response = await fetch('/assess', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-JS-Token': jsToken,
            },
            body: JSON.stringify({ 
                requirement,
                recaptcha_token: recaptchaToken,
                _js_token: jsToken
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Assessment failed');
        }
        
        hideLoading();
        displaySingleResult(data);
        
    } catch (error) {
        hideLoading();
        showError('Error: ' + error.message);
    }
}

async function assessBatch(requirements) {
    showLoading();
    hideError();
    hideResults();
    
    try {
        // Get reCAPTCHA token and JS token
        const recaptchaToken = await getRecaptchaToken();
        const jsToken = generateJSToken();
        
        const response = await fetch('/assess_batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-JS-Token': jsToken,
            },
            body: JSON.stringify({ 
                requirements,
                recaptcha_token: recaptchaToken,
                _js_token: jsToken
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Batch assessment failed');
        }
        
        hideLoading();
        displayBatchResults(data);
        
    } catch (error) {
        hideLoading();
        showError('Error: ' + error.message);
    }
}

function displaySingleResult(data) {
    const resultsContent = document.getElementById('resultsContent');
    
    const riskLevelClass = data.overall_risk_level.toLowerCase();
    const gateClass = data.sprint_ready ? 'approved' : 'blocked';
    const gateText = data.sprint_ready ? '✓ APPROVED' : '✗ BLOCKED';
    
    let html = `
        <div class="result-card">
            <h3>Requirement Assessment</h3>
            <p style="margin-bottom: 15px; color: #666; font-style: italic;">"${escapeHtml(data.requirement)}"</p>
            
            <div class="result-summary">
                <div class="summary-item">
                    <label>Overall Risk Score</label>
                    <div class="value ${riskLevelClass}">${data.overall_score.toFixed(3)}</div>
                    <div class="score-level ${riskLevelClass}">${data.overall_risk_level}</div>
                </div>
                <div class="summary-item">
                    <label>Confidence</label>
                    <div class="value">${data.confidence.toFixed(3)}</div>
                </div>
                <div class="summary-item">
                    <label>Sprint Readiness</label>
                    <div class="gate-status ${gateClass}">${gateText}</div>
                </div>
            </div>
            
            ${data.gate_reason && !data.sprint_ready ? `
                <div class="gate-status blocked" style="margin-top: 15px;">
                    <strong>Blocking Reason:</strong> ${escapeHtml(data.gate_reason)}
                </div>
            ` : ''}
            
            <h4 style="margin-top: 25px; margin-bottom: 15px; color: #555;">Risk Scores by Category</h4>
            <div class="scores-grid">
    `;
    
    // Display scores for each risk category
    const categoryNames = {
        'ambiguity': 'Ambiguity Risk',
        'complexity': 'Complexity Risk',
        'access_security': 'Security Control',
        'io_accuracy': 'Data & I/O Integrity',
        'compliance': 'Compliance & Regulatory',
        'user_error': 'User Interaction Risk',
        'performance': 'Performance & Capacity',
    };
    
    for (const [key, score] of Object.entries(data.scores)) {
        const level = data.risk_levels[key].toLowerCase();
        const categoryName = categoryNames[key] || key;
        
        html += `
            <div class="score-card">
                <h4>${categoryName}</h4>
                <div class="score-value ${level}">${score.toFixed(3)}</div>
                <div class="score-level ${level}">${data.risk_levels[key]}</div>
                <div class="progress-bar">
                    <div class="progress-fill ${level}" style="width: ${score * 100}%"></div>
                </div>
            </div>
        `;
    }
    
    html += `
            </div>
        </div>
    `;
    
    // Add COBIT alignment section
    if (data.cobit_alignment && Object.keys(data.cobit_alignment).length > 0) {
        html += `
            <div class="cobit-section" style="margin-top: 25px;">
                <h4 style="margin-bottom: 15px; color: #555;">COBIT 2019 Alignment</h4>
                <div class="cobit-grid">
        `;
        
        const cobitStatuses = {
            'OK': { class: 'cobit-ok', label: 'OK' },
            'REVIEW': { class: 'cobit-review', label: 'REVIEW' },
            'AT_RISK': { class: 'cobit-risk', label: 'AT RISK' }
        };
        
        for (const [objective, status] of Object.entries(data.cobit_alignment)) {
            const statusInfo = cobitStatuses[status] || cobitStatuses['OK'];
            html += `
                <div class="cobit-item ${statusInfo.class}">
                    <div class="cobit-objective">${escapeHtml(objective)}</div>
                    <div class="cobit-status">${statusInfo.label}</div>
                </div>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
    }
    
    html += `</div>`;
    
    // Add export buttons
    html += `
        <div class="export-buttons">
            <button class="export-btn excel" data-action="excel">
                📊 Export to Excel
            </button>
            <button class="export-btn json" data-action="json">
                💾 Download JSON
            </button>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    
    // Store data for export and attach event listeners
    const exportButtons = resultsContent.querySelectorAll('.export-buttons button');
    exportButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.action === 'excel') {
                exportToExcel(data);
            } else if (btn.dataset.action === 'json') {
                exportToJSON(data);
            }
        });
    });
    
    showResults();
}

function displayBatchResults(data) {
    const resultsContent = document.getElementById('resultsContent');
    
    // Calculate summary statistics
    const total = data.total_requirements;
    const approved = data.results.filter(r => r.sprint_ready).length;
    const blocked = total - approved;
    const avgScore = data.results.reduce((sum, r) => sum + r.overall_score, 0) / total;
    const avgConfidence = data.results.reduce((sum, r) => sum + r.confidence, 0) / total;
    
    let html = `
        <div class="batch-summary">
            <h3>Batch Assessment Summary</h3>
            <div class="batch-stats">
                <div class="stat-item">
                    <div class="number">${total}</div>
                    <div class="label">Total Requirements</div>
                </div>
                <div class="stat-item">
                    <div class="number" style="color: #28a745;">${approved}</div>
                    <div class="label">Approved</div>
                </div>
                <div class="stat-item">
                    <div class="number" style="color: #dc3545;">${blocked}</div>
                    <div class="label">Blocked</div>
                </div>
                <div class="stat-item">
                    <div class="number">${avgScore.toFixed(3)}</div>
                    <div class="label">Avg Risk Score</div>
                </div>
                <div class="stat-item">
                    <div class="number">${avgConfidence.toFixed(3)}</div>
                    <div class="label">Avg Confidence</div>
                </div>
            </div>
        </div>
    `;
    
    // Add heatmap if available
    if (data.heatmap) {
        html += `
            <div class="heatmap-section" style="margin: 30px 0; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);">
                <h3 style="margin-bottom: 15px; color: #667eea;">Risk Heatmap</h3>
                <img src="data:image/png;base64,${data.heatmap}" alt="Risk Heatmap" style="max-width: 100%; height: auto; border-radius: 6px;">
            </div>
        `;
    }
    
    html += `<div class="batch-results">`;
    
    // Display each requirement result
    data.results.forEach((result, index) => {
        const riskLevelClass = result.overall_risk_level.toLowerCase();
        const gateClass = result.sprint_ready ? 'approved' : 'blocked';
        const gateText = result.sprint_ready ? '✓ APPROVED' : '✗ BLOCKED';
        
        html += `
            <div class="result-card">
                <h3>${result.requirement_id}: ${escapeHtml(result.requirement)}</h3>
                
                <div class="result-summary">
                    <div class="summary-item">
                        <label>Overall Risk Score</label>
                        <div class="value ${riskLevelClass}">${result.overall_score.toFixed(3)}</div>
                        <div class="score-level ${riskLevelClass}">${result.overall_risk_level}</div>
                    </div>
                    <div class="summary-item">
                        <label>Confidence</label>
                        <div class="value">${result.confidence.toFixed(3)}</div>
                    </div>
                    <div class="summary-item">
                        <label>Sprint Readiness</label>
                        <div class="gate-status ${gateClass}">${gateText}</div>
                    </div>
                </div>
                
                ${result.gate_reason && !result.sprint_ready ? `
                    <div class="gate-status blocked" style="margin-top: 15px;">
                        <strong>Blocking Reason:</strong> ${escapeHtml(result.gate_reason)}
                    </div>
                ` : ''}
                
                <details style="margin-top: 15px;">
                    <summary style="cursor: pointer; font-weight: 600; color: #667eea;">View Detailed Scores</summary>
                    <div class="scores-grid" style="margin-top: 15px;">
        `;
        
        const categoryNames = {
            'ambiguity': 'Ambiguity Risk',
            'complexity': 'Complexity Risk',
            'access_security': 'Security Control',
            'io_accuracy': 'Data & I/O Integrity',
            'compliance': 'Compliance & Regulatory',
            'user_error': 'User Interaction Risk',
            'performance': 'Performance & Capacity',
        };
        
        for (const [key, score] of Object.entries(result.scores)) {
            const level = result.risk_levels[key].toLowerCase();
            const categoryName = categoryNames[key] || key;
            
            html += `
                <div class="score-card">
                    <h4>${categoryName}</h4>
                    <div class="score-value ${level}">${score.toFixed(3)}</div>
                    <div class="score-level ${level}">${result.risk_levels[key]}</div>
                    <div class="progress-bar">
                        <div class="progress-fill ${level}" style="width: ${score * 100}%"></div>
                    </div>
                </div>
            `;
        }
        
        html += `
                    </div>
                </details>
        `;
        
        // Add COBIT alignment for each requirement
        if (result.cobit_alignment && Object.keys(result.cobit_alignment).length > 0) {
            html += `
                <details style="margin-top: 15px;">
                    <summary style="cursor: pointer; font-weight: 600; color: #667eea;">View COBIT 2019 Alignment</summary>
                    <div class="cobit-grid" style="margin-top: 15px;">
            `;
            
            const cobitStatuses = {
                'OK': { class: 'cobit-ok', label: 'OK' },
                'REVIEW': { class: 'cobit-review', label: 'REVIEW' },
                'AT_RISK': { class: 'cobit-risk', label: 'AT RISK' }
            };
            
            for (const [objective, status] of Object.entries(result.cobit_alignment)) {
                const statusInfo = cobitStatuses[status] || cobitStatuses['OK'];
                html += `
                    <div class="cobit-item ${statusInfo.class}">
                        <div class="cobit-objective">${escapeHtml(objective)}</div>
                        <div class="cobit-status">${statusInfo.label}</div>
                    </div>
                `;
            }
            
            html += `
                    </div>
                </details>
            `;
        }
        
        html += `</div>`;
    });
    
    html += `</div>`;
    
    // Add export buttons
    html += `
        <div class="export-buttons">
            <button class="export-btn excel" data-action="excel">
                📊 Export to Excel
            </button>
            <button class="export-btn json" data-action="json">
                💾 Download JSON
            </button>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    
    // Store data for export and attach event listeners
    const exportButtons = resultsContent.querySelectorAll('.export-buttons button');
    exportButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.action === 'excel') {
                exportToExcel(data);
            } else if (btn.dataset.action === 'json') {
                exportToJSON(data);
            }
        });
    });
    
    showResults();
}

function exportToExcel(data) {
    fetch('/export_excel', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Export failed');
        }
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `kibo_assessment_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => {
        showError('Error exporting to Excel: ' + error.message);
    });
}

function exportToJSON(data) {
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `kibo_assessment_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    document.getElementById('error').classList.add('hidden');
}

function showResults() {
    document.getElementById('results').classList.remove('hidden');
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
}

function clearResults() {
    hideResults();
    hideError();
    document.getElementById('resultsContent').innerHTML = '';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

