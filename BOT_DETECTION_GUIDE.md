# Bot Detection Guide

## 🤖 **Bot Protection Implemented**

Your KIBO-RA application now includes comprehensive bot detection to prevent automated submissions!

---

## 🛡️ **Bot Detection Features**

### **1. reCAPTCHA v3 (Google)**
- **Invisible CAPTCHA** - No user interaction required
- **Score-based** - Returns 0.0 (bot) to 1.0 (human)
- **Threshold**: 0.5 (configurable)
- **Status**: Optional (works without keys, but recommended)

### **2. User-Agent Validation**
- Blocks empty User-Agents
- Detects suspicious patterns (bot, crawler, scraper, etc.)
- Blocks known bot agents (curl, wget, python-requests, etc.)
- Allows search engines for health checks only

### **3. Request Timing Analysis**
- Detects too-fast requests (< 50ms between requests)
- Tracks request patterns
- Identifies machine-like timing (too regular)

### **4. Behavioral Pattern Analysis**
- Detects identical requests in quick succession
- Identifies regular timing patterns (bot-like)
- Checks for missing browser headers
- Tracks suspicious patterns per IP

### **5. Honeypot Field**
- Hidden field that humans won't fill
- Bots often fill all fields automatically
- Field name: `website` (common honeypot name)

### **6. JavaScript Token**
- Proves JavaScript is enabled
- Bots without JS can't generate token
- Informational only (doesn't block)

---

## ⚙️ **Configuration**

### **Enable reCAPTCHA (Recommended)**

1. **Get reCAPTCHA Keys:**
   - Go to https://www.google.com/recaptcha/admin/create
   - Create a new site (reCAPTCHA v3)
   - Copy Site Key and Secret Key

2. **Set Environment Variables:**
   ```bash
   export RECAPTCHA_SITE_KEY="your-site-key"
   export RECAPTCHA_SECRET_KEY="your-secret-key"
   ```

3. **Or Set in Cloud Run:**
   ```bash
   gcloud run services update kibo-ra \
     --set-env-vars RECAPTCHA_SITE_KEY=your-site-key,RECAPTCHA_SECRET_KEY=your-secret-key \
     --region us-central1
   ```

### **Configure Bot Detection**

Edit `bot_detection.py`:

```python
BOT_CONFIG = {
    'RECAPTCHA_ENABLED': True,  # Set to False to disable
    'RECAPTCHA_THRESHOLD': 0.5,  # Lower = stricter (0.0-1.0)
    'ENABLE_BEHAVIORAL_ANALYSIS': True,
    'ENABLE_UA_VALIDATION': True,
    'ENABLE_TIMING_ANALYSIS': True,
    'ENABLE_HONEYPOT': True,
    'REQUIRE_JS_ENABLED': True,
}
```

---

## 🔍 **How It Works**

### **Request Flow:**

1. **User submits form** → Frontend gets reCAPTCHA token
2. **Request sent** → Includes reCAPTCHA token + JS token
3. **Backend checks:**
   - ✅ User-Agent validation
   - ✅ Honeypot field (should be empty)
   - ✅ reCAPTCHA verification (if enabled)
   - ✅ Request timing analysis
   - ✅ Behavioral pattern analysis
4. **If bot detected** → Request rejected (403 error)
5. **If human** → Request processed normally

---

## 📊 **Detection Methods**

| Method | What It Detects | Blocking |
|--------|----------------|----------|
| **reCAPTCHA** | Google's bot detection | ✅ Yes |
| **User-Agent** | Empty/suspicious UAs | ✅ Yes |
| **Timing** | Too-fast requests | ✅ Yes |
| **Patterns** | Identical/regular requests | ✅ Yes |
| **Honeypot** | Bots filling hidden fields | ✅ Yes |
| **JS Token** | JavaScript disabled | ⚠️ Logs only |

---

## 🧪 **Testing**

### **Test Bot Detection:**

```bash
# Test with curl (should be blocked)
curl -X POST http://your-app/assess \
  -H "Content-Type: application/json" \
  -d '{"requirement": "Test"}'

# Expected: 403 Forbidden - "Automated requests are not allowed"
```

### **Test with Browser:**

1. Open your app in a browser
2. Submit a requirement
3. Should work normally (human user)

---

## 🚨 **What Gets Blocked**

### **Automatically Blocked:**
- ✅ curl, wget, python-requests
- ✅ Empty User-Agent
- ✅ Suspicious User-Agent patterns
- ✅ Requests faster than 50ms apart
- ✅ Identical requests in quick succession
- ✅ Honeypot field filled
- ✅ Low reCAPTCHA score (< 0.5)

### **Allowed:**
- ✅ Legitimate browsers (Chrome, Firefox, Safari, etc.)
- ✅ Search engine bots (for /health endpoint only)
- ✅ Normal human behavior

---

## 📝 **Files Modified**

### **New Files:**
- ✅ `bot_detection.py` - Bot detection module

### **Modified Files:**
- ✅ `app.py` - Added bot detection middleware
- ✅ `templates/index.html` - Added reCAPTCHA script + honeypot
- ✅ `static/script.js` - Added reCAPTCHA token + JS token
- ✅ `requirements.txt` - Added `requests` library

---

## 🔧 **Troubleshooting**

### **reCAPTCHA Not Working:**

1. **Check keys are set:**
   ```python
   # In bot_detection.py
   secret_key = BOT_CONFIG['RECAPTCHA_SECRET_KEY'] or \
                request.environ.get('RECAPTCHA_SECRET_KEY')
   ```

2. **Check frontend loads script:**
   - Open browser console
   - Check for reCAPTCHA errors

3. **If not configured:** Bot detection still works without reCAPTCHA!

### **Legitimate Users Blocked:**

1. **Lower reCAPTCHA threshold:**
   ```python
   'RECAPTCHA_THRESHOLD': 0.3  # More lenient
   ```

2. **Disable timing analysis temporarily:**
   ```python
   'ENABLE_TIMING_ANALYSIS': False
   ```

3. **Check logs** for why users were blocked

---

## 📈 **Monitoring**

Bot detection logs all events:

```python
# Logged events:
- "Empty User-Agent from IP"
- "Suspicious User-Agent: ..."
- "Known bot User-Agent: ..."
- "Too fast requests from IP"
- "Suspicious request patterns detected"
- "Honeypot field filled"
- "reCAPTCHA score too low"
```

**Check Cloud Run logs** to monitor bot detection events.

---

## ✅ **Summary**

Your application now has **multi-layer bot protection**:

1. ✅ **reCAPTCHA v3** - Google's bot detection (optional)
2. ✅ **User-Agent validation** - Blocks suspicious agents
3. ✅ **Timing analysis** - Detects too-fast requests
4. ✅ **Pattern analysis** - Identifies bot-like behavior
5. ✅ **Honeypot field** - Catches form-filling bots
6. ✅ **JS token** - Proves JavaScript enabled

**All features work automatically!** No configuration needed (except reCAPTCHA keys for best protection).

---

## 🎯 **Next Steps**

1. **Deploy** - Upload updated files
2. **Configure reCAPTCHA** (optional but recommended)
3. **Test** - Verify bot detection works
4. **Monitor** - Check logs for bot attempts
5. **Adjust** - Tune thresholds if needed

**Your application is now protected against bots!** 🤖🚫

