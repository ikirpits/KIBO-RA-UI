# Fix Cloud Run Startup Timeout Issue

## 🚨 **Problem**

Your deployment is failing with:
```
The user-provided container failed to start and listen on the port defined 
provided by the PORT=8080 environment variable within the allocated timeout.
```

## 🔍 **Root Cause**

The ML models (SentenceTransformer + BERT4RE) take **2-5 minutes** to download and load at startup. Cloud Run's default timeout is too short.

## ✅ **Solutions Applied**

### **1. Lazy Model Loading** ✅
- Models now load **on first request**, not at startup
- Health endpoint works immediately
- Faster startup time

### **2. Increased Timeout** ✅
- Cloud Run timeout increased to 600 seconds (10 minutes)
- Startup CPU boost enabled for faster model loading

### **3. Better Error Handling** ✅
- Health endpoint shows model loading status
- Graceful degradation if models fail to load

---

## 🔧 **Files Updated**

1. ✅ **`app.py`** - Changed to lazy initialization
2. ✅ **`cloudbuild.yaml`** - Increased timeout to 600s

---

## 🚀 **Deploy Again**

### **Option 1: Update Existing Deployment**

```bash
# Update Cloud Run service with longer timeout
gcloud run services update kibo-ra-ui \
    --timeout 600 \
    --startup-cpu-boost \
    --region europe-west1 \
    --project kibo-ra
```

### **Option 2: Redeploy with Updated Code**

If you're using GitHub integration, just push the updated `app.py`:

```bash
git add app.py cloudbuild.yaml
git commit -m "Fix startup timeout - lazy model loading"
git push
```

Cloud Build will automatically rebuild and redeploy.

### **Option 3: Manual Redeploy**

```bash
# Build and deploy again
gcloud builds submit --tag gcr.io/kibo-ra/kibo-ra

# Deploy with longer timeout
gcloud run deploy kibo-ra-ui \
    --image gcr.io/kibo-ra/kibo-ra \
    --platform managed \
    --region europe-west1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 600 \
    --startup-cpu-boost \
    --port 8080
```

---

## 📊 **What Changed**

### **Before:**
```python
# Models loaded immediately at startup
auditor = KIBORequirementsAuditor(RISK_PROTOTYPES)  # Takes 2-5 minutes!
```

### **After:**
```python
# Models loaded on first request
def get_auditor():
    global _auditor
    if _auditor is None:
        _auditor = KIBORequirementsAuditor(RISK_PROTOTYPES)
    return _auditor
```

**Benefits:**
- ✅ Health endpoint responds immediately
- ✅ Container starts listening on port quickly
- ✅ Models load in background on first request
- ✅ No timeout errors

---

## ⚠️ **Important Notes**

1. **First Request:** Will take 2-5 minutes (model loading)
2. **Subsequent Requests:** Fast (< 5 seconds)
3. **Health Check:** Works immediately (doesn't require models)

---

## 🧪 **Test After Deployment**

```bash
# 1. Test health endpoint (should work immediately)
curl https://your-service-url.run.app/health

# Expected: {"status": "healthy", "auditor_ready": false}

# 2. Make first assessment request (will load models)
curl -X POST https://your-service-url.run.app/assess \
  -H "Content-Type: application/json" \
  -d '{"requirement": "Test requirement"}'

# 3. Check health again (should show ready)
curl https://your-service-url.run.app/health

# Expected: {"status": "healthy", "auditor_ready": true}
```

---

## ✅ **Summary**

**Problem:** Models loading at startup → timeout  
**Solution:** Lazy loading + increased timeout  
**Result:** Container starts quickly, models load on demand

**Deploy again and it should work!** 🚀

