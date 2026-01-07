"""
Bot Detection Module for KIBO-RA
Prevents automated bots from submitting requirements.
"""

import re
import time
import hashlib
import logging
from typing import Dict, Optional, List
from collections import defaultdict
from datetime import datetime, timedelta
from flask import request, jsonify, g
import requests

# Configure logging
bot_logger = logging.getLogger('bot_detection')

# Bot detection configuration
BOT_CONFIG = {
    # reCAPTCHA v3 (Google)
    'RECAPTCHA_ENABLED': True,
    'RECAPTCHA_SECRET_KEY': None,  # Set via environment variable
    'RECAPTCHA_SITE_KEY': None,    # Set via environment variable
    'RECAPTCHA_THRESHOLD': 0.5,    # Score threshold (0.0 = bot, 1.0 = human)
    
    # Behavioral analysis
    'ENABLE_BEHAVIORAL_ANALYSIS': True,
    'SUSPICIOUS_PATTERN_THRESHOLD': 5,  # Patterns before flagging
    
    # User-Agent validation
    'ENABLE_UA_VALIDATION': True,
    'BLOCK_EMPTY_UA': True,
    'BLOCK_SUSPICIOUS_UA': True,
    
    # Request timing analysis
    'ENABLE_TIMING_ANALYSIS': True,
    'MIN_REQUEST_TIME_MS': 100,   # Minimum time between requests (ms)
    'TOO_FAST_THRESHOLD_MS': 50,   # Too fast = bot
    
    # Honeypot field
    'ENABLE_HONEYPOT': True,
    'HONEYPOT_FIELD_NAME': 'website',  # Common honeypot name
    
    # Browser fingerprinting
    'ENABLE_FINGERPRINTING': True,
    'REQUIRE_JS_ENABLED': True,  # Check for JavaScript execution
}

# Suspicious User-Agent patterns
SUSPICIOUS_USER_AGENTS = [
    r'(?i)(bot|crawler|spider|scraper|curl|wget|python|java|go-http)',
    r'(?i)(headless|phantom|selenium|webdriver|automation)',
    r'(?i)(^$)',  # Empty user agent
]

# Known bot User-Agents (common ones)
KNOWN_BOT_AGENTS = [
    'Googlebot', 'Bingbot', 'Slurp', 'DuckDuckBot', 'Baiduspider',
    'YandexBot', 'Sogou', 'Exabot', 'facebot', 'ia_archiver',
    'curl', 'Wget', 'python-requests', 'Go-http-client', 'Java',
    'PostmanRuntime', 'Apache-HttpClient', 'okhttp'
]

# Request pattern tracking
_request_patterns: Dict[str, List[Dict]] = defaultdict(list)
_bot_fingerprints: Dict[str, Dict] = defaultdict(dict)


def get_client_fingerprint() -> str:
    """Generate a fingerprint from request characteristics."""
    components = [
        request.headers.get('User-Agent', ''),
        request.headers.get('Accept-Language', ''),
        request.headers.get('Accept-Encoding', ''),
        request.headers.get('Accept', ''),
        request.headers.get('Connection', ''),
        request.remote_addr or '',
    ]
    
    fingerprint_string = '|'.join(str(c) for c in components)
    return hashlib.md5(fingerprint_string.encode()).hexdigest()


def validate_user_agent() -> tuple[bool, Optional[str]]:
    """
    Validate User-Agent header.
    Returns (is_valid, error_message)
    """
    if not BOT_CONFIG['ENABLE_UA_VALIDATION']:
        return True, None
    
    user_agent = request.headers.get('User-Agent', '')
    
    # Check for empty User-Agent
    if BOT_CONFIG['BLOCK_EMPTY_UA'] and not user_agent:
        bot_logger.warning(f"Empty User-Agent from {request.remote_addr}")
        return False, "Invalid request headers"
    
    # Check for suspicious patterns
    if BOT_CONFIG['BLOCK_SUSPICIOUS_UA']:
        for pattern in SUSPICIOUS_USER_AGENTS:
            if re.search(pattern, user_agent):
                bot_logger.warning(f"Suspicious User-Agent: {user_agent[:50]} from {request.remote_addr}")
                return False, "Automated requests are not allowed"
        
        # Check against known bot agents
        ua_lower = user_agent.lower()
        for bot_agent in KNOWN_BOT_AGENTS:
            if bot_agent.lower() in ua_lower:
                # Allow search engine bots for health checks only
                if request.endpoint == '/health':
                    continue
                bot_logger.warning(f"Known bot User-Agent: {bot_agent} from {request.remote_addr}")
                return False, "Automated requests are not allowed"
    
    return True, None


def analyze_request_timing(ip: str) -> tuple[bool, Optional[str]]:
    """
    Analyze request timing patterns to detect bots.
    Returns (is_valid, error_message)
    """
    if not BOT_CONFIG['ENABLE_TIMING_ANALYSIS']:
        return True, None
    
    now = time.time()
    key = f"timing:{ip}"
    
    # Get recent requests
    recent_requests = [
        req for req in _request_patterns[key]
        if now - req['timestamp'] < 60  # Last minute
    ]
    
    if len(recent_requests) >= 2:
        # Calculate time between requests
        times = [req['timestamp'] for req in recent_requests[-2:]]
        time_diff = (times[1] - times[0]) * 1000  # Convert to milliseconds
        
        if time_diff < BOT_CONFIG['TOO_FAST_THRESHOLD_MS']:
            bot_logger.warning(f"Too fast requests from {ip}: {time_diff}ms")
            return False, "Requests are being sent too quickly"
        
        if time_diff < BOT_CONFIG['MIN_REQUEST_TIME_MS']:
            # Suspicious but not blocking yet
            bot_logger.info(f"Fast request pattern from {ip}: {time_diff}ms")
    
    # Record this request
    _request_patterns[key].append({
        'timestamp': now,
        'endpoint': request.endpoint,
        'method': request.method
    })
    
    # Keep only last 10 requests
    _request_patterns[key] = _request_patterns[key][-10:]
    
    return True, None


def analyze_request_patterns(ip: str) -> tuple[bool, Optional[str]]:
    """
    Analyze request patterns for bot-like behavior.
    Returns (is_valid, error_message)
    """
    if not BOT_CONFIG['ENABLE_BEHAVIORAL_ANALYSIS']:
        return True, None
    
    fingerprint = get_client_fingerprint()
    key = f"pattern:{ip}:{fingerprint}"
    
    patterns = _request_patterns[key]
    now = time.time()
    
    # Clean old patterns
    patterns = [p for p in patterns if now - p['timestamp'] < 300]  # Last 5 minutes
    _request_patterns[key] = patterns
    
    # Check for suspicious patterns
    suspicious_count = 0
    
    # Pattern 1: Identical requests in quick succession
    if len(patterns) >= 3:
        recent_data = [p.get('data_hash') for p in patterns[-3:]]
        if len(set(recent_data)) == 1:  # All identical
            suspicious_count += 3
    
    # Pattern 2: Very regular timing (machine-like)
    if len(patterns) >= 5:
        timings = [patterns[i+1]['timestamp'] - patterns[i]['timestamp'] 
                  for i in range(len(patterns)-1)]
        if timings:
            avg_timing = sum(timings) / len(timings)
            variance = sum((t - avg_timing) ** 2 for t in timings) / len(timings)
            if variance < 0.1:  # Very regular timing
                suspicious_count += 2
    
    # Pattern 3: Missing common headers
    required_headers = ['Accept', 'Accept-Language']
    missing_headers = sum(1 for h in required_headers if not request.headers.get(h))
    if missing_headers > 0:
        suspicious_count += 1
    
    if suspicious_count >= BOT_CONFIG['SUSPICIOUS_PATTERN_THRESHOLD']:
        bot_logger.warning(f"Suspicious request patterns detected from {ip}: {suspicious_count} patterns")
        return False, "Suspicious request patterns detected"
    
    # Record this request pattern
    data_hash = hashlib.md5(str(request.get_json() or '').encode()).hexdigest()
    patterns.append({
        'timestamp': now,
        'endpoint': request.endpoint,
        'data_hash': data_hash
    })
    _request_patterns[key] = patterns[-20:]  # Keep last 20
    
    return True, None


def verify_recaptcha(token: Optional[str]) -> tuple[bool, Optional[str]]:
    """
    Verify reCAPTCHA v3 token.
    Returns (is_valid, error_message)
    """
    if not BOT_CONFIG['RECAPTCHA_ENABLED']:
        return True, None
    
    secret_key = BOT_CONFIG['RECAPTCHA_SECRET_KEY'] or \
                 request.environ.get('RECAPTCHA_SECRET_KEY')
    
    if not secret_key:
        bot_logger.warning("reCAPTCHA secret key not configured")
        return True, None  # Don't block if not configured
    
    if not token:
        return False, "reCAPTCHA verification required"
    
    try:
        # Verify with Google
        verify_url = "https://www.google.com/recaptcha/api/siteverify"
        response = requests.post(verify_url, data={
            'secret': secret_key,
            'response': token,
            'remoteip': request.remote_addr
        }, timeout=5)
        
        result = response.json()
        
        if not result.get('success'):
            bot_logger.warning(f"reCAPTCHA verification failed: {result.get('error-codes', [])}")
            return False, "reCAPTCHA verification failed"
        
        # Check score (v3 returns score 0.0-1.0)
        score = result.get('score', 0.0)
        if score < BOT_CONFIG['RECAPTCHA_THRESHOLD']:
            bot_logger.warning(f"reCAPTCHA score too low: {score} from {request.remote_addr}")
            return False, "reCAPTCHA verification failed"
        
        return True, None
    
    except Exception as e:
        bot_logger.error(f"reCAPTCHA verification error: {str(e)}")
        # Don't block on verification errors, but log them
        return True, None


def check_honeypot(data: Dict) -> tuple[bool, Optional[str]]:
    """
    Check honeypot field (should be empty for humans).
    Returns (is_valid, error_message)
    """
    if not BOT_CONFIG['ENABLE_HONEYPOT']:
        return True, None
    
    honeypot_field = BOT_CONFIG['HONEYPOT_FIELD_NAME']
    
    # Check if honeypot field is filled (bots often fill all fields)
    if honeypot_field in data and data[honeypot_field]:
        bot_logger.warning(f"Honeypot field filled from {request.remote_addr}")
        return False, "Invalid form submission"
    
    return True, None


def check_javascript_enabled(data: Dict) -> tuple[bool, Optional[str]]:
    """
    Check if JavaScript is enabled (required for legitimate browsers).
    Returns (is_valid, error_message)
    """
    if not BOT_CONFIG['REQUIRE_JS_ENABLED']:
        return True, None
    
    # Check for JavaScript token (set by frontend)
    js_token = data.get('_js_token') or request.headers.get('X-JS-Token')
    
    if not js_token:
        # Not blocking, but logging
        bot_logger.info(f"No JS token from {request.remote_addr}")
        # Don't block - some legitimate users may have JS disabled
    
    return True, None


def detect_bot(recaptcha_token: Optional[str] = None, request_data: Optional[Dict] = None) -> tuple[bool, Optional[str]]:
    """
    Comprehensive bot detection.
    Returns (is_human, error_message)
    
    Args:
        recaptcha_token: reCAPTCHA v3 token from frontend
        request_data: Request JSON data for honeypot check
    """
    ip = request.remote_addr or 'unknown'
    
    # 1. Validate User-Agent
    ua_valid, ua_error = validate_user_agent()
    if not ua_valid:
        return False, ua_error
    
    # 2. Check honeypot (if data provided)
    if request_data:
        honeypot_valid, honeypot_error = check_honeypot(request_data)
        if not honeypot_valid:
            return False, honeypot_error
    
    # 3. Verify reCAPTCHA
    recaptcha_valid, recaptcha_error = verify_recaptcha(recaptcha_token)
    if not recaptcha_valid:
        return False, recaptcha_error
    
    # 4. Analyze request timing
    timing_valid, timing_error = analyze_request_timing(ip)
    if not timing_valid:
        return False, timing_error
    
    # 5. Analyze request patterns
    pattern_valid, pattern_error = analyze_request_patterns(ip)
    if not pattern_valid:
        return False, pattern_error
    
    # 6. Check JavaScript (informational only)
    if request_data:
        check_javascript_enabled(request_data)
    
    return True, None


def bot_detection_middleware(f):
    """Decorator to add bot detection to routes."""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get reCAPTCHA token from request
        recaptcha_token = None
        if request.is_json:
            data = request.get_json() or {}
            recaptcha_token = data.get('recaptcha_token') or data.get('g-recaptcha-response')
        else:
            data = {}
        
        # Detect bots
        is_human, error = detect_bot(recaptcha_token, data)
        
        if not is_human:
            bot_logger.warning(f"Bot detected from {request.remote_addr}: {error}")
            return jsonify({
                'error': 'Bot detected',
                'message': error or 'Automated requests are not allowed'
            }), 403
        
        # Continue with request
        return f(*args, **kwargs)
    
    return decorated_function


def get_recaptcha_site_key() -> Optional[str]:
    """Get reCAPTCHA site key for frontend."""
    return BOT_CONFIG['RECAPTCHA_SITE_KEY'] or \
           request.environ.get('RECAPTCHA_SITE_KEY')


# Cleanup function
def cleanup_bot_detection():
    """Clean up old bot detection data."""
    now = time.time()
    keys_to_delete = []
    
    for key, patterns in _request_patterns.items():
        # Keep only patterns from last hour
        _request_patterns[key] = [
            p for p in patterns if now - p['timestamp'] < 3600
        ]
        if not _request_patterns[key]:
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        del _request_patterns[key]

