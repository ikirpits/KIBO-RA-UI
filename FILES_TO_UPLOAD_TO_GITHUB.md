# Files to Upload to GitHub

## 🎯 **What You Need to Upload**

Since Cloud Build pulls from GitHub, you need to upload these **2 updated files**:

### **1. `app.py`** ✅ (Updated with lazy loading)
- **Location:** `c:\Users\ikirpits\kibo-ra\app.py`
- **Key Change:** Models now load on first request, not at startup
- **Lines 96-111:** Contains the `get_auditor()` function

### **2. `cloudbuild.yaml`** ✅ (Updated with 600s timeout)
- **Location:** `c:\Users\ikirpits\kibo-ra\cloudbuild.yaml`
- **Key Change:** Timeout increased to 600s + startup CPU boost
- **Line 46:** `--timeout 600`
- **Line 47:** `--startup-cpu-boost`

---

## 📤 **How to Upload**

### **Option 1: GitHub Web Interface**

1. Go to: https://github.com/ikirpits/KIBO-RA-UI
2. Click on `app.py`
3. Click **Edit** (pencil icon)
4. Copy the entire contents of your local `app.py`
5. Paste and **Commit changes**
6. Repeat for `cloudbuild.yaml`

### **Option 2: GitHub Desktop / Git CLI**

If you have GitHub Desktop or Git CLI set up:

```bash
cd C:\Users\ikirpits\kibo-ra
git init
git remote add origin https://github.com/ikirpits/KIBO-RA-UI.git
git add app.py cloudbuild.yaml
git commit -m "Fix startup timeout: lazy model loading + 600s timeout"
git push -u origin main
```

---

## ✅ **After Uploading**

1. Cloud Build will **automatically detect** the new commit
2. It will **rebuild** (takes ~17 minutes as you mentioned)
3. This time it should **deploy successfully** ✅

---

## 📋 **Why This Will Work**

**Before (Current GitHub Code):**
- Models load at startup → Takes 2-5 minutes
- Container times out before models finish loading ❌

**After (Updated Code):**
- Container starts immediately (no model loading)
- Health endpoint works right away ✅
- Models load on first request (user waits, not Cloud Run)

---

## ⏱️ **Timeline**

1. **Upload files:** ~2 minutes
2. **Cloud Build detects:** ~30 seconds
3. **Build process:** ~17 minutes (as you mentioned)
4. **Deploy:** ~2 minutes
5. **Total:** ~20 minutes

---

**Upload these 2 files to GitHub and Cloud Build will automatically redeploy!** 🚀

