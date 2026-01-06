# Quick Fix for Deployment Timeout

## 🚨 **The Problem**

Your deployment failed because:
- ML models take **2-5 minutes** to download and load
- Cloud Run default timeout is **too short**
- Container doesn't start listening on port in time

## ✅ **The Fix**

I've updated your code to:
1. ✅ **Lazy load models** - Load on first request, not at startup
2. ✅ **Increased timeout** - 600 seconds (10 minutes)
3. ✅ **Startup CPU boost** - Faster model loading

---

## 🚀 **Deploy the Fix**

### **If Using GitHub (Your Current Setup):**

```bash
# 1. Commit the updated files
git add app.py cloudbuild.yaml
git commit -m "Fix startup timeout - lazy model loading"
git push

# Cloud Build will automatically rebuild and redeploy
```

### **If Deploying Manually:**

```bash
# 1. Update the service timeout
gcloud run services update kibo-ra-ui \
    --timeout 600 \
    --startup-cpu-boost \
    --region europe-west1 \
    --project kibo-ra

# 2. Or redeploy with updated code
gcloud builds submit --tag gcr.io/kibo-ra/kibo-ra

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

## 📝 **What Changed**

### **app.py:**
- Changed from: `auditor = KIBORequirementsAuditor(...)` at startup
- Changed to: `get_auditor()` function - loads on first request

### **cloudbuild.yaml:**
- Timeout: 300s → 600s
- Added: `--startup-cpu-boost`

---

## ✅ **After Deployment**

1. **Health endpoint** works immediately (no models needed)
2. **First request** takes 2-5 minutes (loading models)
3. **Subsequent requests** are fast (< 5 seconds)

---

## 🧪 **Test**

```bash
# Health check (should work immediately)
curl https://kibo-ra-ui-xxxxx.run.app/health

# First assessment (will load models - takes 2-5 min)
curl -X POST https://kibo-ra-ui-xxxxx.run.app/assess \
  -H "Content-Type: application/json" \
  -d '{"requirement": "Test requirement"}'
```

---

**Push the updated code and redeploy!** 🚀

