// API Base URL - will be set from localStorage or user input
const API_BASE_URL = window.API_BASE_URL || localStorage.getItem('kibo_api_url') || '';

// Helper function to get full API URL
function getApiUrl(endpoint) {
    if (!API_BASE_URL && !window.API_BASE_URL) {
        const savedUrl = localStorage.getItem('kibo_api_url');
        if (savedUrl) {
            window.API_BASE_URL = savedUrl;
            return `${savedUrl}${endpoint}`;
        }
        throw new Error('API URL not configured. Please configure it in the header.');
    }
    const baseUrl = window.API_BASE_URL || API_BASE_URL;
    return `${baseUrl}${endpoint}`;
}

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
        const response = await fetch(getApiUrl('/assess'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ requirement })
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
        const response = await fetch(getApiUrl('/assess_batch'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ requirements })
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
    
    const riskLevelClass = data.overall_risk_level?.toLowerCase() || 'medium';
    const gateClass = data.sprint_ready ? 'approved' : 'blocked';
    const gateText = data.sprint_ready ? '✓ APPROVED' : '✗ BLOCKED';
    
    let html = `
        <div class="result-card">
            <h3>Requirement Assessment</h3>
            <p style="margin-bottom: 15px; color: #666; font-style: italic;">"${escapeHtml(data.requirement)}"</p>
            
            <div class="result-summary">
                <div class="summary-item">
                    <label>Overall Risk Score</label>
                    <div class="value ${riskLevelClass}">${(data.overall_risk || data.overall_score || 0).toFixed(3)}</div>
                    <div class="score-level ${riskLevelClass}">${data.overall_risk_level || 'MEDIUM'}</div>
                </div>
                <div class="summary-item">
                    <label>Confidence</label>
                    <div class="value">${(data.confidence || 0).toFixed(3)}</div>
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
        'delivery_predictability': 'Delivery Predictability',
    };
    
    const scores = data.scores || {};
    const riskLevels = data.risk_levels || {};
    
    for (const [key, score] of Object.entries(scores)) {
        const level = (riskLevels[key] || 'MEDIUM').toLowerCase();
        const categoryName = categoryNames[key] || key;
        
        html += `
            <div class="score-card">
                <h4>${categoryName}</h4>
                <div class="score-value ${level}">${score.toFixed(3)}</div>
                <div class="score-level ${level}">${riskLevels[key] || 'MEDIUM'}</div>
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
    
    // Add export buttons (JSON only for GitHub Pages - Excel requires backend)
    html += `
        <div class="export-buttons">
            <button class="export-btn json" onclick="exportToJSON(${JSON.stringify(data).replace(/"/g, '&quot;')})">
                💾 Download JSON
            </button>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    showResults();
}

function displayBatchResults(data) {
    const resultsContent = document.getElementById('resultsContent');
    const results = data.results || [];
    
    // Calculate summary statistics
    const total = results.length;
    const approved = results.filter(r => r.sprint_ready).length;
    const blocked = total - approved;
    const avgScore = results.reduce((sum, r) => sum + (r.overall_risk || r.overall_score || 0), 0) / total;
    const avgConfidence = results.reduce((sum, r) => sum + (r.confidence || 0), 0) / total;
    
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
    results.forEach((result, index) => {
        const riskLevelClass = (result.overall_risk_level || 'MEDIUM').toLowerCase();
        const gateClass = result.sprint_ready ? 'approved' : 'blocked';
        const gateText = result.sprint_ready ? '✓ APPROVED' : '✗ BLOCKED';
        
        html += `
            <div class="result-card">
                <h3>R${index + 1}: ${escapeHtml(result.requirement)}</h3>
                
                <div class="result-summary">
                    <div class="summary-item">
                        <label>Overall Risk Score</label>
                        <div class="value ${riskLevelClass}">${(result.overall_risk || result.overall_score || 0).toFixed(3)}</div>
                        <div class="score-level ${riskLevelClass}">${result.overall_risk_level || 'MEDIUM'}</div>
                    </div>
                    <div class="summary-item">
                        <label>Confidence</label>
                        <div class="value">${(result.confidence || 0).toFixed(3)}</div>
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
            'delivery_predictability': 'Delivery Predictability',
        };
        
        const scores = result.scores || {};
        const riskLevels = result.risk_levels || {};
        
        for (const [key, score] of Object.entries(scores)) {
            const level = (riskLevels[key] || 'MEDIUM').toLowerCase();
            const categoryName = categoryNames[key] || key;
            
            html += `
                <div class="score-card">
                    <h4>${categoryName}</h4>
                    <div class="score-value ${level}">${score.toFixed(3)}</div>
                    <div class="score-level ${level}">${riskLevels[key] || 'MEDIUM'}</div>
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
            <button class="export-btn json" onclick="exportBatchToJSON(${JSON.stringify(data).replace(/"/g, '&quot;')})">
                💾 Download JSON
            </button>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    showResults();
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

function exportBatchToJSON(data) {
    exportToJSON(data);
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


