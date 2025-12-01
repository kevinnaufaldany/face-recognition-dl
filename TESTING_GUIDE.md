# ✅ TESTING & TROUBLESHOOTING GUIDE

## 🧪 Quick Test

### **Test 1: Run the App**

```bash
cd d:\Abckup_desktop\Semester7\DL\tubes
streamlit run app.py
```

**Expected Output:**

```
You can now view your Streamlit app in your browser.
URL: http://localhost:8501
```

### **Test 2: Upload a Test Image**

1. Click "Pilih foto wajah"
2. Select any JPG/PNG from your computer
3. Upload

**Expected Behavior:**

- ✅ App IMMEDIATELY starts preprocessing (you should see spinner)
- ✅ After a few seconds, show both images:
  - Left: Original photo
  - Right: Processed 224×224
- ✅ Below says either:
  - "✅ Wajah Terdeteksi & Di-Crop" OR
  - "⚠️ Wajah Tidak Terdeteksi (Resize Langsung)"

### **Test 3: Click "Prediksi Sekarang"**

1. Click the button
2. Wait for inference

**Expected Behavior:**

- ✅ Shows a name in large text
- ✅ Shows confidence percentage
- ✅ Shows Top-5 table
- ✅ Shows bar chart

---

## 🔍 Debugging: If preprocessing fails

### **Issue 1: App crashes on upload**

**Check:**

```python
# Run this in Python shell
python -c "
from utils.face_crop import FaceCropper
cropper = FaceCropper()
print('✅ FaceCropper initialized successfully')
"
```

**Expected:** No error, print success message

**If error:**

- Check MediaPipe is installed: `pip list | grep mediapipe`
- Check OpenCV is installed: `pip list | grep opencv`

---

### **Issue 2: "Wajah tidak terdeteksi" always appears**

**Possible causes:**

1. Image quality too low
2. Face is too small or too far
3. Multiple faces (system detects first)
4. Face is partially hidden

**Test different images:**

- Clear, front-facing portrait ✅ Should detect
- Side profile → May not detect
- Multiple people → May detect 1
- Very small face → Likely not detect

---

### **Issue 3: Prediction gives wrong name**

**This could be:**

1. **Model issue**: Model not trained well (dataset too small)
2. **Preprocessing issue**: Face not cropped correctly
3. **Confidence too low**: User in dataset might have similar face

**Check:**

- Is confidence > 50%? If < 50%, might be uncertain
- Check Top-5 table: Is correct name in top-5?
- If not, dataset might not have enough samples (4 per class is very low)

---

### **Issue 4: "Error load model"**

**Check:**

```python
python -c "
import torch
from model_convnext import create_model
model = create_model(num_classes=70, pretrained=False, dropout=0.3, device='cpu')
print('✅ Model created successfully')
print(f'Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
"
```

**Expected:** Success, ~28M parameters

**If checkpoint error:**

```python
python -c "
import torch
state = torch.load('checkpoints/convnext_tiny_20251201_144518/best_epoch7.pth',
                   map_location='cpu')
print('✅ Checkpoint loaded')
print(f'Keys: {list(state.keys())}')
"
```

---

### **Issue 5: Face detection never works**

**Debug step-by-step:**

```python
# Test 1: Can import FaceCropper?
from utils.face_crop import FaceCropper
cropper = FaceCropper()
print("✅ Step 1: FaceCropper imported")

# Test 2: Can load image?
from PIL import Image
img = Image.open("test_image.jpg").convert("RGB")
print(f"✅ Step 2: Image loaded, size: {img.size}")

# Test 3: Can detect & crop?
face, success = cropper.detect_and_crop_face_from_pil(img)
print(f"✅ Step 3: Detection result: success={success}")
if success:
    print(f"   Face image shape: {face.shape}")
else:
    print("   Face not detected, fallback used")
```

**Each step should succeed**

---

## 📊 Performance Check

### **Check 1: Image Format**

```python
from PIL import Image
img = Image.open("photo.jpg")
print(f"Format: {img.format}")  # Should be JPEG, PNG, etc
print(f"Size: {img.size}")      # Should be (width, height)
print(f"Mode: {img.mode}")      # Should be RGB, RGBA, etc

# Try convert to RGB
img_rgb = img.convert("RGB")
print(f"✅ Converted to RGB: {img_rgb.mode}")
```

### **Check 2: Preprocessing Output**

```python
processed_img = img.resize((224, 224))
print(f"Processed size: {processed_img.size}")  # Should be (224, 224)
print(f"Processed format: {type(processed_img)}")  # Should be PIL Image
```

### **Check 3: Model Input**

```python
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

tensor = transform(processed_img).unsqueeze(0)
print(f"Tensor shape: {tensor.shape}")  # Should be (1, 3, 224, 224)
print(f"Tensor dtype: {tensor.dtype}")  # Should be torch.float32
print(f"Tensor range: {tensor.min():.2f} to {tensor.max():.2f}")
```

---

## 🚀 Real-World Testing

### **Test Scenario 1: Good Conditions**

**Image:** Clear, front-facing, good lighting, 1 person
**Expected:**

- ✅ Face detected
- ✅ Confidence ≥ 70%
- ✅ Correct name predicted

### **Test Scenario 2: Medium Conditions**

**Image:** Slight angle, normal lighting, 1 person
**Expected:**

- ✅ Face detected (might use strategy 2/3)
- ✅ Confidence 50-70%
- ✅ Name predicted (might be uncertain)

### **Test Scenario 3: Poor Conditions**

**Image:** Far away, side profile, multiple people
**Expected:**

- ⚠️ Face might not detect (use fallback resize)
- ⚠️ Confidence < 50%
- ⚠️ Prediction might be wrong

### **Test Scenario 4: No Face**

**Image:** Landscape, object, animal
**Expected:**

- ⚠️ Face not detected (use strategy 4 fallback)
- ⚠️ Shows warning message
- ⚠️ Still tries to predict (but likely wrong)

---

## 📈 Confidence Score Guide

| Confidence | Meaning              | Action                  |
| ---------- | -------------------- | ----------------------- |
| 95-100%    | Model very sure      | ✅ Trust the prediction |
| 80-95%     | Model quite sure     | ✅ Likely correct       |
| 70-80%     | Model confident      | ✅ Probably correct     |
| 50-70%     | Model uncertain      | ⚠️ Check top-5          |
| 30-50%     | Model very uncertain | ❌ Verify manually      |
| < 30%      | Model guessing       | ❌ Don't trust          |

---

## 🐛 Common Issues & Solutions

| Issue                          | Cause              | Solution                           |
| ------------------------------ | ------------------ | ---------------------------------- |
| Always "Face tidak terdeteksi" | Image quality low  | Try clearer photo                  |
| Wrong prediction every time    | Dataset too small  | Add more training data             |
| App crashes on upload          | Import error       | Check pip install all dependencies |
| "Error load model"             | Checkpoint missing | Check path in app.py               |
| Very low confidence (< 30%)    | Model uncertain    | Try better image                   |
| Slow inference                 | CPU only           | Use GPU if available               |

---

## ✅ Validation Checklist

- [ ] App runs without errors: `streamlit run app.py`
- [ ] Can upload image: Click file uploader
- [ ] Preprocessing shows immediately
- [ ] See both original + processed image
- [ ] Shows face detection status
- [ ] Click prediksi button works
- [ ] Shows top-1 prediction
- [ ] Shows confidence percentage
- [ ] Shows top-5 table
- [ ] Shows bar chart
- [ ] Confidence is reasonable (not all 1-2%)

---

## 🔧 If Everything Fails

**Nuclear option - reset and test from scratch:**

```bash
# 1. Stop app (Ctrl+C)

# 2. Clear cache
del %APPDATA%\Roaming\.streamlit\*

# 3. Reinstall packages
pip install --upgrade streamlit torch torchvision mediapipe opencv-python pillow

# 4. Test imports
python -c "import torch; import streamlit; import mediapipe; print('✅ All imports OK')"

# 5. Run app again
streamlit run app.py
```

---

## 📞 If Still Issues

**Provide this information:**

1. **Screenshot of error message**
2. **App.py line that fails** (if known)
3. **Image you're testing with** (size, format)
4. **Output from debug script above**
5. **Python version**: `python --version`
6. **Package versions**:
   ```
   pip list | grep -E "streamlit|torch|mediapipe|opencv"
   ```

---

## 🎯 Success Indicators

✅ **You'll know it's working when:**

1. Upload image → Immediately see preprocessing spinner
2. Preprocessing shows both original + processed side-by-side
3. Status shows "✅ Wajah Terdeteksi" or "⚠️ Tidak terdeteksi"
4. Click button → See prediction with reasonable confidence
5. Top-5 table shows other likely candidates
6. Bar chart visualizes the confidence distribution

---

**Last Updated**: 2025-12-01 17:00 UTC
**Version**: 1.0
**Status**: ✅ Ready for Testing
