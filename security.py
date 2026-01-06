"""
Security module for KIBO-RA application.
Protects against: DDoS, code injection, MitM, ransomware, malware, bulk abuse.
"""

import re
import time
import hashlib
import logging
from functools import wraps
from collections import defaultdict
from datetime import datetime, timedelta
from flask import request, jsonify, g
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger('security')

# Rate limiting storage (in-memory, use Redis for production)
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)
_blocked_ips: Dict[str, datetime] = {}
_suspicious_patterns: Dict[str, int] = defaultdict(int)

# Security configuration
SECURITY_CONFIG = {
    # Rate limiting
    'RATE_LIMIT_REQUESTS_PER_MINUTE': 30,  # Max requests per minute per IP
    'RATE_LIMIT_BATCH_PER_HOUR': 10,  # Max batch requests per hour per IP
    'RATE_LIMIT_BURST': 5,  # Allow burst of 5 requests
    
    # Request size limits
    'MAX_REQUEST_SIZE': 10 * 1024 * 1024,  # 10MB max request size
    'MAX_REQUIREMENT_LENGTH': 5000,  # Max characters per requirement
    'MAX_BATCH_SIZE': 25,  # Max requirements per batch
    
    # Abuse detection
    'ABUSE_THRESHOLD': 100,  # Requests before blocking
    'ABUSE_WINDOW_MINUTES': 60,  # Time window for abuse detection
    'BLOCK_DURATION_MINUTES': 60,  # How long to block abusive IPs
    
    # Security headers
    'ENABLE_CSP': True,
    'ENABLE_HSTS': True,
    'ENABLE_XSS_PROTECTION': True,
}

# Dangerous patterns for code injection detection
DANGEROUS_PATTERNS = [
    # SQL Injection patterns
    r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into|update\s+set)",
    r"(?i)(--|/\*|\*/|;|'|'|xp_|sp_)",
    
    # Command injection patterns
    r"(?i)(\||&|;|`|\$\(|<\s*\(|>\s*\(|exec\(|eval\(|system\(|shell_exec)",
    r"(?i)(cmd\.exe|/bin/sh|/bin/bash|powershell|wscript|vbscript)",
    
    # Script injection patterns
    r"(?i)(<script|javascript:|onerror=|onload=|onclick=|onmouseover=)",
    r"(?i)(<iframe|document\.cookie|window\.location|eval\()",
    
    # Path traversal
    r"(\.\./|\.\.\\|\.\.%2f|\.\.%5c)",
    
    # Ransomware/malware indicators
    r"(?i)(encrypt|decrypt|ransom|malware|virus|trojan|backdoor)",
    r"(?i)(\.exe|\.bat|\.cmd|\.ps1|\.sh|\.pyc|\.dll)",
    
    # Suspicious file operations
    r"(?i)(file_get_contents|fopen|fwrite|fput|unlink|rmdir|mkdir)",
    
    # Base64 encoded payloads (common in attacks)
    r"([A-Za-z0-9+/]{100,}={0,2})",  # Long base64 strings
    
    # Hex encoded payloads
    r"(\\x[0-9a-fA-F]{2}){10,}",  # Multiple hex escapes
]

# Suspicious keywords
SUSPICIOUS_KEYWORDS = [
    'password', 'secret', 'key', 'token', 'credential', 'admin',
    'root', 'sudo', 'privilege', 'escalation', 'exploit', 'hack'
]


def get_client_ip() -> str:
    """Get client IP address, handling proxies."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr or 'unknown'


def is_ip_blocked(ip: str) -> bool:
    """Check if IP is currently blocked."""
    if ip in _blocked_ips:
        block_until = _blocked_ips[ip]
        if datetime.now() < block_until:
            return True
        else:
            # Block expired, remove it
            del _blocked_ips[ip]
    return False


def block_ip(ip: str, duration_minutes: int = None):
    """Block an IP address."""
    duration = duration_minutes or SECURITY_CONFIG['BLOCK_DURATION_MINUTES']
    block_until = datetime.now() + timedelta(minutes=duration)
    _blocked_ips[ip] = block_until
    security_logger.warning(f"IP {ip} blocked until {block_until}")


def check_rate_limit(ip: str, endpoint: str):
    """
    Check if request exceeds rate limit.
    Returns (allowed, error_message)
    """
    # Check if IP is blocked
    if is_ip_blocked(ip):
        return False, "IP address is temporarily blocked due to abuse"
    
    now = time.time()
    key = f"{ip}:{endpoint}"
    
    # Clean old entries (older than 1 minute)
    _rate_limit_store[key] = [
        timestamp for timestamp in _rate_limit_store[key]
        if now - timestamp < 60
    ]
    
    # Check rate limit
    requests_count = len(_rate_limit_store[key])
    
    if endpoint == '/assess_batch':
        # Stricter limit for batch endpoint
        max_requests = SECURITY_CONFIG['RATE_LIMIT_BATCH_PER_HOUR']
        window = 3600  # 1 hour
        # Clean entries older than 1 hour
        _rate_limit_store[key] = [
            timestamp for timestamp in _rate_limit_store[key]
            if now - timestamp < window
        ]
        requests_count = len(_rate_limit_store[key])
        
        if requests_count >= max_requests:
            # Track abuse
            _suspicious_patterns[ip] += 1
            if _suspicious_patterns[ip] >= SECURITY_CONFIG['ABUSE_THRESHOLD']:
                block_ip(ip)
                return False, "Excessive batch requests detected. IP blocked."
            return False, f"Rate limit exceeded. Maximum {max_requests} batch requests per hour."
    else:
        # Standard rate limit
        max_requests = SECURITY_CONFIG['RATE_LIMIT_REQUESTS_PER_MINUTE']
        if requests_count >= max_requests:
            _suspicious_patterns[ip] += 1
            if _suspicious_patterns[ip] >= SECURITY_CONFIG['ABUSE_THRESHOLD']:
                block_ip(ip)
                return False, "Excessive requests detected. IP blocked."
            return False, f"Rate limit exceeded. Maximum {max_requests} requests per minute."
    
    # Record this request
    _rate_limit_store[key].append(now)
    
    return True, None


def sanitize_input(text: str, max_length: int = None):
    """
    Sanitize and validate input text.
    Returns (sanitized_text, error_message)
    """
    if not isinstance(text, str):
        return "", "Input must be a string"
    
    # Check length
    max_len = max_length or SECURITY_CONFIG['MAX_REQUIREMENT_LENGTH']
    if len(text) > max_len:
        return "", f"Input too long. Maximum {max_len} characters allowed."
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Check for dangerous patterns
    text_lower = text.lower()
    
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text):
            security_logger.warning(f"Dangerous pattern detected: {pattern[:50]}")
            return "", "Input contains potentially dangerous content"
    
    # Check for suspicious keywords in context (not just presence)
    # Only flag if they appear in suspicious contexts
    suspicious_contexts = [
        r'(?i)(password\s*=|secret\s*=|key\s*=)',
        r'(?i)(exec\s*\(|eval\s*\(|system\s*\()',
    ]
    
    for context_pattern in suspicious_contexts:
        if re.search(context_pattern, text):
            security_logger.warning(f"Suspicious context detected in input")
            return "", "Input contains suspicious content"
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace (prevent hidden characters)
    text = ' '.join(text.split())
    
    return text, None


def validate_requirement(requirement: str):
    """Validate a single requirement."""
    if not requirement or not isinstance(requirement, str):
        return False, "Requirement must be a non-empty string"
    
    # Sanitize input
    sanitized, error = sanitize_input(requirement)
    if error:
        return False, error
    
    # Check minimum length (prevent empty or whitespace-only)
    if len(sanitized.strip()) < 10:
        return False, "Requirement must be at least 10 characters"
    
    return True, None


def validate_batch_requirements(requirements: List[str]):
    """Validate batch requirements."""
    if not isinstance(requirements, list):
        return False, "Requirements must be a list"
    
    # Check batch size
    if len(requirements) > SECURITY_CONFIG['MAX_BATCH_SIZE']:
        return False, f"Maximum {SECURITY_CONFIG['MAX_BATCH_SIZE']} requirements per batch"
    
    if len(requirements) == 0:
        return False, "At least one requirement is required"
    
    # Validate each requirement
    for i, req in enumerate(requirements):
        if not isinstance(req, str):
            return False, f"Requirement {i+1} must be a string"
        
        valid, error = validate_requirement(req)
        if not valid:
            return False, f"Requirement {i+1}: {error}"
    
    return True, None


def check_request_size():
    """Check if request size exceeds limits."""
    if request.content_length:
        if request.content_length > SECURITY_CONFIG['MAX_REQUEST_SIZE']:
            return False, f"Request too large. Maximum {SECURITY_CONFIG['MAX_REQUEST_SIZE'] / 1024 / 1024:.1f}MB allowed"
    return True, None


def setup_security_headers(response):
    """Add security headers to response."""
    # Content Security Policy
    if SECURITY_CONFIG['ENABLE_CSP']:
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # Needed for Flask templates
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers['Content-Security-Policy'] = csp
    
    # HTTP Strict Transport Security
    if SECURITY_CONFIG['ENABLE_HSTS']:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # XSS Protection
    if SECURITY_CONFIG['ENABLE_XSS_PROTECTION']:
        response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Permissions policy
    response.headers['Permissions-Policy'] = (
        'geolocation=(), microphone=(), camera=(), '
        'payment=(), usb=(), magnetometer=(), gyroscope=()'
    )
    
    return response


def security_middleware(f):
    """Decorator to add security checks to routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP
        ip = get_client_ip()
        g.client_ip = ip
        
        # Check if IP is blocked
        if is_ip_blocked(ip):
            security_logger.warning(f"Blocked IP {ip} attempted access")
            return jsonify({
                'error': 'Access denied',
                'message': 'Your IP address has been temporarily blocked due to suspicious activity'
            }), 403
        
        # Check request size
        size_ok, size_error = check_request_size()
        if not size_ok:
            security_logger.warning(f"Oversized request from {ip}")
            return jsonify({'error': size_error}), 413
        
        # Check rate limit
        rate_ok, rate_error = check_rate_limit(ip, request.endpoint or 'unknown')
        if not rate_ok:
            security_logger.warning(f"Rate limit exceeded for {ip} on {request.endpoint}")
            return jsonify({'error': rate_error}), 429
        
        # Execute the route
        try:
            response = f(*args, **kwargs)
            
            # Add security headers
            if hasattr(response, 'headers'):
                response = setup_security_headers(response)
            
            return response
        except Exception as e:
            security_logger.error(f"Error in {request.endpoint}: {str(e)}")
            # Don't expose internal errors
            return jsonify({
                'error': 'An error occurred processing your request'
            }), 500
    
    return decorated_function


def log_security_event(event_type: str, details: Dict):
    """Log security events for monitoring."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'ip': get_client_ip(),
        'endpoint': request.endpoint,
        'details': details
    }
    security_logger.info(f"Security event: {log_entry}")


# Cleanup function to run periodically (call from app startup)
def cleanup_rate_limits():
    """Clean up old rate limit entries."""
    now = time.time()
    keys_to_delete = []
    
    for key, timestamps in _rate_limit_store.items():
        # Keep only entries from last hour
        _rate_limit_store[key] = [
            ts for ts in timestamps if now - ts < 3600
        ]
        if not _rate_limit_store[key]:
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        del _rate_limit_store[key]
    
    # Clean expired blocks
    now_dt = datetime.now()
    ips_to_unblock = [
        ip for ip, block_until in _blocked_ips.items()
        if now_dt >= block_until
    ]
    for ip in ips_to_unblock:
        del _blocked_ips[ip]
        _suspicious_patterns[ip] = 0  # Reset abuse counter

