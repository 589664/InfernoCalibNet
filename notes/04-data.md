**Modifying Networks for Grayscale Images**

- **ImageNet Models Expect RGB**: EfficientNet and other CNNs pretrained on **ImageNet** use RGB images, which have distinct mean and standard deviation values per channel.
- **My Approach**: Modified EfficientNet to accept **1-channel grayscale input** by adjusting the first convolutional layer. This reduces **memory usage** but means the model must **relearn** early features.
- **Pseudo-RGB Approach**: Others use pseudo-RGB by replicating the grayscale channel across R, G, B, which allows them to use **pretrained weights**, often resulting in **faster convergence**.
- **Trade-offs**: The 1-channel approach is more efficient in terms of **resources** but sacrifices pretrained benefits; pseudo-RGB uses more memory but retains **transfer learning advantages**.
- **Next Steps**: Try both approaches to determine which works best for the **X-ray classification** task.

---
