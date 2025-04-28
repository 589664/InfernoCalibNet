---
title: "Specifications"
---

# **Batch Size, Epochs, and Dataset Size Considerations**

### 1. Understanding the Relations:

- **Dataset Size**: I have 112,121 grayscale X-ray images.
- **Hardware Specs**:
  - **CPU**: I have 24 cores and 64 GB of memory.
  - **GPU**: I'm using an NVIDIA RTX A4000 with 16 GB memory and Tensor cores.

### 2. Batch Size Calculation:

- **Memory Considerations**: The batch size depends on the available **GPU memory** and the size of my images. Since these are grayscale (1 channel), they are less memory-intensive compared to RGB (3 channels).
  - Estimate **per image memory**: The input size depends on the version of EfficientNet being used:
    - **EfficientNet B3**: Input size is **300x300** pixels.
    - **EfficientNet B4**: Input size is **380x380** pixels.
    - **EfficientNet B5**: Input size is **456x456** pixels.
  - **Memory Requirement per Image**:
    - Assuming the **EfficientNet B3** input size of **300x300** pixels, with 1 channel, each image would take **300x300x1 bytes** or **90 KB** (approx). Accounting for floating-point precision and overhead during training, the **effective memory per image** could be closer to **200 KB**.
    - For **EfficientNet B4** (380x380), each image would be approximately **144 KB**, and, with overhead, **effective memory could be around 288 KB**.
    - For **EfficientNet B5** (456x456), each image would be approximately **205 KB**, and, with overhead, **effective memory could be around 410 KB**.
  - **Memory Requirement per Batch**:
    - **Batch Size = 64** (for EfficientNet B3):
      - **64 images × 200 KB** = **12.8 MB** of GPU memory (just for the images).
      - Adding model weights, activations, and other overhead, I estimate **8-10 GB of total memory requirement** for **batch size of 64**.
    - **Batch Size = 128** (for EfficientNet B3):
      - **128 images × 200 KB** = **25.6 MB** of GPU memory.
      - Adding other overhead, this could push memory usage closer to **16 GB**, but this depends on the model size and complexity.
    - For **EfficientNet B4** and **B5**, the memory requirements per batch will increase proportionally due to the larger input sizes.
- **Ideal Starting Point**: Batch size of **64** is a reasonable starting point given my **16 GB GPU memory**, but I may need to experiment with **smaller or larger batch sizes** depending on the utilization shown by `nvidia-smi` and the specific EfficientNet model I'm using.

### 3. Epochs and Steps per Epoch Calculation:

- **Steps per Epoch** = `total_images / batch_size`
  - For **batch size of 64**:
    - `112,121 / 64 ≈ 1752` steps per epoch.
  - For **batch size of 128**:
    - `112,121 / 128 ≈ 876` steps per epoch.
- **Total Number of Epochs**:
  - Based on **convergence** and **overfitting** observations, I plan to set epochs to **50-100** as a starting point. I will use **early stopping** to prevent unnecessary training if the validation loss plateaus.
  - This means the model will see the entire dataset **50-100 times**, but with batch-level updates.

### 4. Time Estimation for Training:

- **Estimation per Batch**:
  - The **forward and backward pass time per batch** depends on the model, and since I am using a pretrained model like EfficientNet B3 to B5 with a batch size of **64** on my GPU, a typical forward-backward pass might take around **0.5 - 1 second**.
- **Total Time per Epoch**:
  - For **batch size of 64** and **1752 steps per epoch**:
    - Time per epoch = `0.5 seconds × 1752` ≈ **14-15 minutes**.
    - For **100 epochs**, the total estimated time = `15 minutes × 100` = **1500 minutes** or **25 hours**.
  - This is a rough estimation, but it helps me gauge how long the training process might take.

### 5. CPU Utilization for Data Loading:

- **DataLoader Considerations**:
  - I have a powerful **24-core CPU** with **64 GB RAM**, which I should leverage to ensure that **data loading** is not a bottleneck.
  - Setting the `num_workers` in the **DataLoader** to **8-12** will allow me to utilize multiple CPU cores, which will help ensure the GPU is fully utilized without waiting for data.

### 6. GPU Considerations - Mixed Precision:

- Given the **Tensor cores** on the **NVIDIA RTX A4000**, enabling **mixed precision training** can significantly improve both speed and memory utilization.
- This allows me to potentially use a **larger batch size** or accelerate training time while reducing precision requirements for intermediate calculations.

### 7. Summary of Key Parameters:

- **Batch Size**: I will start with **64**, and adjust based on memory utilization.
- **Epochs**: Set to **50-100**, using **early stopping** to optimize training time.
- **Steps per Epoch**: Varies based on batch size; for batch size **64**, expect **1752 steps** per epoch.
- **CPU Parallelism**: Set `num_workers` to **8-12** to maximize data throughput.
- **Mixed Precision**: Enable to leverage Tensor cores and optimize memory usage.
- **EfficientNet Model Input Sizes**:
  - **EfficientNet B3**: 300x300 pixels.
  - **EfficientNet B4**: 380x380 pixels.
  - **EfficientNet B5**: 456x456 pixels.
