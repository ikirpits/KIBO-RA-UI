# Quick Fix: Update Service Timeout

## 🚨 **Problem**

Your GitHub repo hasn't been updated with the lazy loading code yet, but we can fix the timeout issue immediately by updating the Cloud Run service directly.

## ✅ **Immediate Fix (No Code Push Required)**

Update the Cloud Run service timeout **right now**:

```bash
gcloud run services update kibo-ra-ui \
    --timeout 600 \
    --startup-cpu-boost \
    --region europe-west1 \
    --project kibo-ra
```

This will allow the current code to start (even with model loading at startup).

---

## 🔄 **Then Push Updated Code**

After updating the timeout, push the lazy loading code to GitHub:

```bash
# 1. Check what files changed
git status

# 2. Add updated files
git add app.py cloudbuild.yaml

# 3. Commit
git commit -m "Fix startup timeout - lazy model loading"

# 4. Push to GitHub
git push
```

Cloud Build will automatically rebuild with the lazy loading code.

---

## 📋 **What Changed**

**app.py:**
- Models now load **on first request** (lazy loading)
- Health endpoint works immediately
- Faster startup

**cloudbuild.yaml:**
- Timeout: 300s → 600s
- Added: `--startup-cpu-boost`

---

## ✅ **After Both Steps**

1. **Service timeout updated** → Current code can start
2. **Lazy loading pushed** → Future deployments are faster

---

**Run the gcloud command above first, then push the code!** 🚀

