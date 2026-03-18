# 🧠 Bone Fracture Detection Model Architecture

---

# 🖼️ Architecture Diagram

![Bone Fracture Model Architecture](./assets/model_architecture.png)

> 📌 *Note:* Place your image in `assets/model_architecture.png` or update the path accordingly.

---

# 📌 1. High-Level Overview

The BoneFractureAI system is built using a **Convolutional Neural Network (CNN)** with **Transfer Learning** based on **ResNet50**.

### 🎯 Objective

* Detect bone fractures from X-ray images
* Provide classification: **Normal vs Fractured**
* Generate **visual explanations using Grad-CAM**

---

## 🔑 Key Highlights

| Component             | Description                    |
| --------------------- | ------------------------------ |
| **Architecture Type** | CNN with Transfer Learning     |
| **Base Model**        | ResNet50 (ImageNet Pretrained) |
| **Task**              | Binary Classification          |
| **Input Size**        | 224 × 224 × 3                  |
| **Output**            | 2 Classes (Normal, Fracture)   |
| **Explainability**    | Grad-CAM                       |

---

# 🏗️ 2. Complete Model Pipeline

```text
Input Image → Preprocessing → ResNet50 → Custom Head → Softmax Output
```

---

# 🔍 3. Detailed Architecture

---

## 🧾 3.1 Stage 1: Input & Preprocessing

### Input

* X-ray Image (RGB or Grayscale)

### Processing Steps

1. Resize → **224 × 224**
2. Normalize using ImageNet statistics

### 📐 Formula

```math
x_normalized = (x - mean) / std
```

Where:

* mean = [0.485, 0.456, 0.406]
* std = [0.229, 0.224, 0.225]

---

## 🧠 3.2 Stage 2: ResNet50 Backbone (Feature Extractor)

### 🔹 Key Idea

ResNet50 extracts **high-level features** like:

* edges
* textures
* bone structures

---

### 📊 Architecture Summary

| Layer   | Output Shape | Parameters |
| ------- | ------------ | ---------- |
| Conv1   | 112×112×64   | 9,408      |
| MaxPool | 56×56×64     | -          |
| Layer1  | 56×56×256    | 215K       |
| Layer2  | 28×28×512    | 1.2M       |
| Layer3  | 14×14×1024   | 7.1M       |
| Layer4  | 7×7×2048     | 15M        |
| AvgPool | 1×1×2048     | -          |

👉 Total Parameters: **23.5M**

---

### 🔁 Residual Learning (Core Concept)

```math
Output = ReLU(F(x) + x)
```

✅ Solves **vanishing gradient problem**
✅ Enables **deep networks**

---

## 🎯 3.3 Stage 3: Custom Classification Head

### Input

Feature vector: **2048**

---

### Layer Breakdown

#### 🔹 Fully Connected Layer

* 2048 → 512

```math
h1 = W1 · x + b1
```

#### 🔹 Activation

```math
h1 = max(0, h1)
```

#### 🔹 Dropout

```math
h1 = h1 · mask / 0.5
```

#### 🔹 Final Layer

* 512 → 2

```math
logits = W2 · h1 + b2
```

---

### 🔹 Output (Softmax)

```math
P(class) = exp(z) / Σ exp(z)
```

---

### 📊 Parameters

| Layer          | Parameters |
| -------------- | ---------- |
| FC1            | 1,049,088  |
| FC2            | 1,026      |
| **Total Head** | ~1.05M     |

---

### 🔢 Total Model Parameters

👉 **24.5 Million Parameters**

---

# 📐 4. Mathematical Formulation

---

## 🔁 Forward Pass

```math
F = ResNet50(I_norm)
h1 = ReLU(W1 · F + b1)
z = W2 · Dropout(h1) + b2
ŷ = Softmax(z)
```

---

## 📉 Loss Function (Cross Entropy)

```math
L = -[y log(ŷ1) + (1-y) log(ŷ0)]
```

---

## ⚙️ Optimization (Adam)

```math
θ = θ - α * m̂ / (√v̂ + ε)
```

### Hyperparameters

* Learning Rate: **1e-4**
* β1 = 0.9
* β2 = 0.999

---

# 🎨 5. Grad-CAM (Explainable AI)

---

## 🎯 Purpose

To **visualize important regions** in X-ray images.

---

## 📐 Mathematical Steps

```math
α = avg(∂y / ∂A)
L = ReLU(Σ αk · Ak)
```

---

## 🎯 Output Interpretation

| Color     | Meaning         |
| --------- | --------------- |
| 🔴 Red    | High importance |
| 🟡 Yellow | Medium          |
| 🔵 Blue   | Low             |

---

# 🏋️ 6. Training Strategy

---

## 🔹 Phase 1: Feature Extraction

* Freeze ResNet50
* Train classifier only

## 🔹 Phase 2: Fine-Tuning

* Unfreeze last layers
* Lower learning rate

---

## 🔹 Data Augmentation

* Flip
* Rotation
* Brightness changes

---

## 🔹 Hyperparameters

| Parameter  | Value  |
| ---------- | ------ |
| Batch Size | 16     |
| Epochs     | 10–30  |
| Optimizer  | Adam   |
| LR         | 0.0001 |
| Dropout    | 0.5    |

---

# 📊 7. Evaluation Metrics

| Metric    | Formula       | Meaning                      |
| --------- | ------------- | ---------------------------- |
| Accuracy  | (TP+TN)/Total | Overall correctness          |
| Precision | TP/(TP+FP)    | Correct fracture predictions |
| Recall    | TP/(TP+FN)    | Detection ability            |
| F1 Score  | Harmonic mean | Balance                      |

---

# 🤔 8. Why This Architecture?

* ✅ ResNet50 → Strong pretrained features
* ✅ Transfer Learning → Less data required
* ✅ Dropout → Prevent overfitting
* ✅ Grad-CAM → Medical explainability

---

# 💻 9. Code Implementation

```python
class BoneFractureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)

        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.backbone(x)
```

---

# 🎓 10. Final Summary

* Model: **ResNet50 + Custom Head**
* Parameters: **~24.5M**
* Task: **Binary Classification**
* Key Feature: **Explainable AI (Grad-CAM)**

---

# 🚀 Conclusion

This architecture provides:

* High accuracy
* Interpretability
* Scalability

👉 Ideal for **AI-powered medical diagnosis systems**

---
