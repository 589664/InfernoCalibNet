---
title: "Classification task"
---

# Machine Learning Classification: Binary, Multiclass, and Multilabel

## **Summary of Classification Tasks**

### **1. Binary Classification**

- **Task**: Decide between two classes (e.g., Dog vs. Not Dog).
- **Output Layer**: Single logit (`nn.Linear(in_features, 1)`).
- **Activation Function**: Sigmoid (compresses logits into probabilities between 0 and 1).
- **Decision**:
  - `P > 0.5`: Positive class (e.g., Dog).
  - `P ≤ 0.5`: Negative class (e.g., Not Dog).

### **2. Multiclass Classification**

- **Task**: Decide between multiple classes, only one can be chosen (e.g., Dog, Cat, Bird).
- **Output Layer**: Multiple logits, one per class (`nn.Linear(in_features, num_classes)`).
- **Activation Function**: Softmax (converts logits into a probability distribution across classes).
- **Decision**: Choose the class with the highest probability.

### **3. Multilabel Classification**

- **Task**: Detect multiple labels per image (e.g., Dog, Cat, Bird — any combination is possible).
- **Output Layer**: Multiple logits, one per label (`nn.Linear(in_features, num_labels)`).
- **Activation Function**: Sigmoid (applied independently to each logit).
- **Decision**: Apply a threshold (e.g., 0.5) for each probability to decide label presence.

---

## **Summary Table**

| **Task**                  | **Input**       | **Logits (Output)** | **Post Activation**  | **Final Prediction**           |
| ------------------------- | --------------- | ------------------- | -------------------- | ------------------------------ |
| Binary Classification     | `(3, 224, 224)` | `2.3`               | `P(dog) ≈ 0.91`      | Dog (P > 0.5)                  |
| Multiclass Classification | `(3, 224, 224)` | `[1.2, 3.8, -0.5]`  | `[0.07, 0.90, 0.03]` | Cat (highest probability)      |
| Multilabel Classification | `(3, 224, 224)` | `[2.1, -1.3, 1.8]`  | `[0.89, 0.21, 0.86]` | Dog and Bird (threshold > 0.5) |

---

## **Differences Between Tasks**

| **Aspect**              | **Binary Classification** | **Multiclass Classification** | **Multilabel Classification**        |
| ----------------------- | ------------------------- | ----------------------------- | ------------------------------------ |
| **Output Shape**        | `(N, 1)`                  | `(N, C)`                      | `(N, C)`                             |
| **Activation Function** | Sigmoid                   | Softmax                       | Sigmoid                              |
| **Loss Function**       | BCEWithLogitsLoss         | CrossEntropyLoss              | BCEWithLogitsLoss                    |
| **Prediction Type**     | Single binary decision    | Single class decision         | Independent decisions for each label |
| **Example Labels**      | Dog vs. Not Dog           | Dog, Cat, or Bird             | Dog, Cat, and/or Bird                |

---
