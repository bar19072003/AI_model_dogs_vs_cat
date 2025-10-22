# Model Experiment Report — Dogs vs Cats CNN

This report documents the model improvement process, experiments, and reasoning behind each design decision made while developing the Dogs vs Cats CNN classifier.

---

## 1. Baseline Model

**Setup**
- 3 convolutional layers  
- No Batch Normalization  
- No Dropout  
- Strong Data Augmentation (high rotation, zoom, and shear)  
- 10 epochs  
- Batch size: **512**  
- Learning rate: default (`1e-3`)

**Results**
- Training accuracy: ~70%  
- Validation accuracy: slightly lower (~65–68%)  
- Overfitting was minimal, but the model underperformed — learning capacity was limited.

**Observation**
- The model struggled to converge properly; large batch size caused gradients to average out too much.  
- Strong augmentations made learning slower and noisier.

---

## 2. Reduced Batch Size

**Change:** Decreased batch size from **512 → 64**

**Reasoning:**  
Smaller batches introduce more gradient noise, often improving generalization and helping the optimizer escape local minima.

**Result:**  
- Accuracy improved by roughly **+5%** (to around 75%)  
- Loss decreased faster during early epochs.  
- Training became more stable.

---

## 3. Lower Learning Rate

**Change:** Reduced learning rate from `1e-3` → `3e-4`

**Reasoning:**  
The learning rate was too high, causing the optimizer to overshoot minima. A smaller rate allows smoother convergence.

**Result:**  
- Additional **+3%** improvement in accuracy (~78%)  
- Validation curves became smoother and more stable.  
- Reduced oscillations in loss.

---

## 4. Softer Augmentation

**Change:** Reduced the augmentation strength:
- Lower `rotation_range`, `zoom_range`, and `shear_range`  
- Kept horizontal flips

**Reasoning:**  
The original augmentations distorted images too much and made it harder for the model to generalize correctly.

**Result:**  
- Validation accuracy increased by another few percent (~80–82%)  
- Model began to converge faster and reach consistent plateaus.  

---

## 5. Added a Fourth Convolutional Layer + Improvements

**Changes:**
- Added **one extra Conv2D layer** to increase feature extraction depth.  
- Added **Batch Normalization** after every Conv layer.  
- Set **padding='same'** for better spatial preservation.  
- Increased training epochs to **15**.  

**Reasoning:**  
The deeper architecture captures more complex patterns and textures.  
BatchNorm stabilizes training and accelerates convergence by normalizing activations.  
More epochs give the model time to converge to its optimal weights.

**Result:**
- Validation accuracy reached **~90%** at the final epoch.  
- Training accuracy around 93%.  
- Model showed minimal overfitting; curves of train vs. val accuracy were close.  

---

## 6. Final Summary

| Stage | Change | Validation Accuracy | Improvement |
|--------|---------|---------------------|-------------|
| Baseline | 3 conv layers, strong aug, lr=1e-3, batch=512 | ~70% | — |
| ↓ Smaller batch (64) | Better gradient flow | ~75% | +5% |
| ↓ Lower LR (3e-4) | Smoother optimization | ~78% | +3% |
| ↓ Weaker augmentation | Less distortion | ~81% | +3% |
| + Added Conv layer + BatchNorm + 15 epochs | Deeper learning & stabilization | ~90% | +9% |

**Total improvement:** ~20% absolute gain in validation accuracy.

---

## 7. Lessons Learned

- **Batch size** plays a big role in stability; smaller batches improved gradient quality.  
- **Learning rate** reduction was key to achieving smooth convergence.  
- **Data augmentation** should be balanced — too strong can hurt generalization.  
- **Deeper networks with BatchNorm** handle variability in lighting and textures better.  
- **Gradual experimentation** yields measurable, interpretable improvements — each change built logically on the previous step.

---

## 8. Final Configuration Snapshot

- **Architecture:** 4 Conv2D (16→128 filters) + BatchNorm + MaxPool  
- **Dense:** 512 (ReLU) + BatchNorm + Dropout(0.4)  
- **Optimizer:** Adam (lr=3e-4, clipnorm=1.0)  
- **Loss:** Binary Crossentropy  
- **Epochs:** 15  
- **Batch size:** 64  
- **Final validation accuracy:** ≈ 90%

---

### Author
Bar Bibi — 2025  