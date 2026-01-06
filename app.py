"""
KIBO Requirements Auditor Web UI
Flask web application for entering and assessing requirements online.
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
from krr1 import KIBORequirementsAuditor, RISK_PROTOTYPES, RISK_CATEGORIES, get_risk_level, results_to_dataframe
from security import (
    security_middleware, validate_requirement, validate_batch_requirements,
    sanitize_input, setup_security_headers, cleanup_rate_limits, log_security_event
)
from bot_detection import bot_detection_middleware, get_recaptcha_site_key
import json
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import atexit

app = Flask(__name__)

# Configure Flask security settings
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max request size
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Add security headers to all responses
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    return setup_security_headers(response)

# Cleanup rate limits on shutdown
atexit.register(cleanup_rate_limits)

# Import bot detection cleanup
try:
    from bot_detection import cleanup_bot_detection
    atexit.register(cleanup_bot_detection)
except ImportError:
    pass  # Bot detection module not available

def generate_heatmap_base64(results_data):
    """Generate a heatmap image from batch results and return as base64 string."""
    try:
        # Create DataFrame from results
        rows = []
        for i, result in enumerate(results_data):
            row = {'requirement': f'R{i+1}'}
            for k, v in result['scores'].items():
                row[k] = v
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Get score columns (the 8 risk categories)
        score_cols = [col for col in RISK_CATEGORIES if col in df.columns]
        if not score_cols:
            return None
        
        scores = df[score_cols].values
        fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.5)))
        im = ax.imshow(scores, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Risk Score")
        
        # Set labels
        col_labels = [col.replace("access_security", "SECURITY").replace("_", " ").title() for col in score_cols]
        ax.set_xticks(range(len(score_cols)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["requirement"].tolist())
        
        # Add text annotations
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                ax.text(j, i, f"{scores[i, j]:.2f}", ha="center", va="center", fontsize=8)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

# Initialize the auditor (load models once at startup)
print("Initializing KIBO Requirements Auditor...")
auditor = KIBORequirementsAuditor(RISK_PROTOTYPES)
print("Auditor ready!")

@app.route('/')
def index():
    """Home page with requirement input form."""
    # Pass reCAPTCHA site key to template if configured
    recaptcha_key = get_recaptcha_site_key()
    response = render_template('index.html', recaptcha_site_key=recaptcha_key)
    return response

@app.route('/assess', methods=['POST'])
@security_middleware
@bot_detection_middleware
def assess():
    """Assess a single requirement."""
    # Validate JSON request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400
    
    requirement_text = data.get('requirement', '')
    
    # Validate and sanitize input
    valid, error = validate_requirement(requirement_text)
    if not valid:
        log_security_event('invalid_input', {'error': error, 'endpoint': '/assess'})
        return jsonify({'error': error}), 400
    
    # Sanitize input
    sanitized_text, sanitize_error = sanitize_input(requirement_text)
    if sanitize_error:
        log_security_event('sanitization_failed', {'error': sanitize_error, 'endpoint': '/assess'})
        return jsonify({'error': sanitize_error}), 400
    
    try:
        # Assess the requirement (use sanitized text)
        result = auditor.assess_requirement(sanitized_text)
        
        # Format response (use sanitized text for display)
        response = {
            'requirement': sanitized_text,
            'overall_score': round(result.overall, 3),
            'overall_risk_level': get_risk_level(result.overall),
            'confidence': round(result.confidence, 3),
            'scores': {k: round(v, 3) for k, v in result.scores.items()},
            'risk_levels': {k: get_risk_level(v) for k, v in result.scores.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add sprint readiness gate (use sanitized text)
        gate = auditor.assess_for_sprint_readiness(sanitized_text)
        response['sprint_ready'] = bool(gate['sprint_ready'])  # Convert numpy bool to Python bool
        response['gate_decision'] = str(gate['gate_decision'])
        response['gate_reason'] = str(gate['gate_reason'])
        # Add COBIT alignment
        response['cobit_alignment'] = gate.get('cobit_alignment', {})
        
        return jsonify(response)
    
    except Exception as e:
        log_security_event('assessment_error', {'error': str(e), 'endpoint': '/assess'})
        # Don't expose internal error details
        return jsonify({'error': 'Assessment failed. Please check your input and try again.'}), 500

@app.route('/assess_batch', methods=['POST'])
@security_middleware
@bot_detection_middleware
def assess_batch():
    """Assess multiple requirements at once."""
    # Validate JSON request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400
    
    requirements = data.get('requirements', [])
    
    # Validate batch requirements
    valid, error = validate_batch_requirements(requirements)
    if not valid:
        log_security_event('invalid_batch_input', {'error': error, 'endpoint': '/assess_batch'})
        return jsonify({'error': error}), 400
    
    # Sanitize all requirements
    sanitized_requirements = []
    for req in requirements:
        sanitized, sanitize_error = sanitize_input(req)
        if sanitize_error:
            log_security_event('sanitization_failed', {
                'error': sanitize_error, 
                'endpoint': '/assess_batch',
                'requirement_index': len(sanitized_requirements)
            })
            return jsonify({'error': f'Invalid requirement: {sanitize_error}'}), 400
        sanitized_requirements.append(sanitized)
    
    try:
        # Assess all requirements (use sanitized requirements)
        results = auditor.assess_batch(sanitized_requirements)
        
        # Format response
        response = {
            'total_requirements': len(results),
            'results': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for i, result in enumerate(results):
            req_result = {
                'requirement_id': f'R{i+1}',
                'requirement': result.requirement,
                'overall_score': round(result.overall, 3),
                'overall_risk_level': get_risk_level(result.overall),
                'confidence': round(result.confidence, 3),
                'scores': {k: round(v, 3) for k, v in result.scores.items()},
                'risk_levels': {k: get_risk_level(v) for k, v in result.scores.items()}
            }
            
            # Add sprint readiness
            gate = auditor.assess_for_sprint_readiness(result.requirement)
            req_result['sprint_ready'] = bool(gate['sprint_ready'])  # Convert numpy bool to Python bool
            req_result['gate_decision'] = str(gate['gate_decision'])
            req_result['gate_reason'] = str(gate['gate_reason'])
            # Add COBIT alignment
            req_result['cobit_alignment'] = gate.get('cobit_alignment', {})
            
            response['results'].append(req_result)
        
        # Generate heatmap for batch results
        heatmap_base64 = generate_heatmap_base64(response['results'])
        if heatmap_base64:
            response['heatmap'] = heatmap_base64
        
        return jsonify(response)
    
    except Exception as e:
        log_security_event('batch_assessment_error', {'error': str(e), 'endpoint': '/assess_batch'})
        # Don't expose internal error details
        return jsonify({'error': 'Batch assessment failed. Please check your input and try again.'}), 500

@app.route('/export_excel', methods=['POST'])
@security_middleware
@bot_detection_middleware
def export_excel():
    """Export assessment results to Excel file."""
    # Validate JSON request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        # Handle single result
        if 'requirement' in data:
            # Validate and sanitize requirement
            req_text = data.get('requirement', '')
            sanitized_req, sanitize_error = sanitize_input(req_text)
            if sanitize_error:
                return jsonify({'error': f'Invalid requirement: {sanitize_error}'}), 400
            
            # Single result - convert to list format
            results_data = [{
                'requirement_id': 'R1',
                'requirement': sanitized_req,  # Use sanitized requirement
                'overall_score': data.get('overall_score', 0),
                'overall_risk_level': data.get('overall_risk_level', ''),
                'confidence': data.get('confidence', 0),
                'scores': data.get('scores', {}),
                'risk_levels': data.get('risk_levels', {}),
                'sprint_ready': data.get('sprint_ready', False),
                'gate_decision': data.get('gate_decision', ''),
                'gate_reason': data.get('gate_reason', '')
            }]
        elif 'results' in data:
            # Batch results - sanitize all requirements
            results_data = []
            for result in data['results']:
                req_text = result.get('requirement', '')
                sanitized_req, sanitize_error = sanitize_input(req_text)
                if sanitize_error:
                    return jsonify({'error': f'Invalid requirement in batch: {sanitize_error}'}), 400
                
                sanitized_result = result.copy()
                sanitized_result['requirement'] = sanitized_req
                results_data.append(sanitized_result)
        else:
            return jsonify({'error': 'Invalid data format'}), 400
        
        # Create DataFrame
        rows = []
        for result in results_data:
            row = {
                'Requirement ID': result.get('requirement_id', 'N/A'),
                'Requirement': result.get('requirement', ''),
                'Overall Score': result.get('overall_score', 0),
                'Overall Risk Level': result.get('overall_risk_level', ''),
                'Confidence': result.get('confidence', 0),
                'Sprint Ready': 'Yes' if result.get('sprint_ready', False) else 'No',
                'Gate Decision': result.get('gate_decision', ''),
                'Gate Reason': result.get('gate_reason', '')
            }
            
            # Add individual risk scores
            scores = result.get('scores', {})
            category_names = {
                'ambiguity': 'Ambiguity Risk',
                'complexity': 'Complexity Risk',
                'access_security': 'Security Control',
                'io_accuracy': 'Data & I/O Integrity',
                'compliance': 'Compliance & Regulatory',
                'user_error': 'User Interaction Risk',
                'performance': 'Performance & Capacity',
            }
            
            for key, score in scores.items():
                category_name = category_names.get(key, key.replace('_', ' ').title())
                row[f'{category_name} Score'] = score
                risk_levels = result.get('risk_levels', {})
                row[f'{category_name} Level'] = risk_levels.get(key, '')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Assessment Results', index=False)
        
        output.seek(0)
        
        # Generate filename with timestamp
        filename = f"kibo_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return send_file(
            output,
            mimetype='application/vnd.openpyxl-formats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        log_security_event('export_error', {'error': str(e), 'endpoint': '/export_excel'})
        return jsonify({'error': 'Export failed. Please try again.'}), 500

@app.route('/health')
def health():
    """Health check endpoint (no security middleware to allow monitoring)."""
    return jsonify({'status': 'healthy', 'auditor_ready': True})

# Error handlers
@app.errorhandler(413)
def request_too_large(error):
    """Handle request too large errors."""
    return jsonify({'error': 'Request too large. Maximum 10MB allowed.'}), 413

@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit exceeded errors."""
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({'error': 'Bad request. Please check your input.'}), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    log_security_event('internal_error', {'error': str(error)})
    return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*60)
    print("KIBO Requirements Auditor Web UI")
    print("="*60)
    print(f"\nStarting web server on port {port}...")
    if debug:
        print("DEBUG mode enabled")
        print("Open your browser and navigate to: http://localhost:5000")
    else:
        print("Production mode - ensure proper WSGI server is configured")
    print("\nPress Ctrl+C to stop the server.\n")
    app.run(debug=debug, host='0.0.0.0', port=port)

