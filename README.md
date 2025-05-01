<h1 align="center">
  🧠InfernoCalibNet🔥
</h1>

<p align="center">
  <img src="assets/banner.png" alt="InfernoCalibNet Banner" style="width:100%; margin:auto;">
</p>

> **Calibration modeling for CNN outputs using Bayesian nonparametric inference**

[![License: GPL v3](https://img.shields.io/badge/License:%20Covenant-GPL%20v3-f3eef8.svg?logo=read-the-docs&logoColor=white)](LICENSE.md)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenan-2.0-fcd53f.svg?logo=read-the-docs&logoColor=white)](LICENSE.md)
[![InfernoCalibNet](https://img.shields.io/badge/InfernoCalibNet%20Docs-1.0-074057.svg?logo=read-the-docs&logoColor=white)](https://m4siko.github.io/InfernoCalibNet/)


---

## 🔍 Table of Contents

- [📊 Overview](#overview)
- [🧠 System Architecture (project workflow)](#system-architecture-project-workflow)
- [📂 Repository Structure](#repository-structure)
- [🔧 Installation](#installation)
- [👣 Quick Start](#quick-start)
- [🔄 Usage Examples](#usage-examples)
- [📄 Documentation](#documentation)
- [🌟 Acknowledgements](#acknowledgements)
- [📖 Citation](#citation)

---

## 📊 Overview

<div style="border-left: 4px solid #074057; padding: 1em;">

Through uncertainty-aware modeling, this research explores the application of Bayesian regression and convolutional neural networks (CNNs) to assist medical decision-making. The CNN serves as a probability converter in this configuration, converting complex visual input into a vector of real-valued class values. These ratings are treated as structured summaries of the visual information rather than as final judgments.
<br>

Putting these outputs into Inferno, a Bayesian transducer that transforms raw logits into calibrated probability distributions, is the project's main goal. By adjusting for patient-specific base rates and incorporating previous data, this statistical post-processing stage allows clinicians to make decisions based on presented benefit rather than strict classification. Because the final decision is visible, individualized, and based on Bayesian reasoning, the design eliminates the need to "explain" the CNN itself by separating prediction from action.

Through a series of hypothetical experiments created to mirror the needs of personalized medicine, the study investigates Inferno's durability and medicinal usefulness. Grad-CAM heatmaps, which provide visual explanations that localize model attention, are added to calibrated outputs. The combined probabilistic and location outputs are meant to help clinicians go beyond classification and provide informed, personalized medicine by helping them customize treatment choices for specific patients.

</div>

## 🧠 System Architecture (project workflow)

<p align="center">
  <img src="assets/system_architecture.svg" alt="InfernoCalibNet System Architecture" style="width:100%; margin:auto;">
</p>


## 📂 Repository Structure


## 🔧 Installation


## 👣 Quick Start

> **Note:** The Inferno R-package provides Bayesian nonparametric calibration software package for CNN outputs.

### Clone the repository without submodules

```bash
git clone https://github.com/589664/InfernoCalibNet.git
```

### Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/589664/InfernoCalibNet.git
```

### If already cloned without submodules

```bash
git submodule update --init --recursive
```

## 🔄 Usage Examples

## 📖 Citation

If you use this software or refer to its documentation, please consider citing:

```bibtex
@software{infernoCalibNet,
  author       = {Maksim Ohvrill and PierGianLuca Porta Mana},
  title        = {InfernoCalibNet: Bayesian Calibration of CNN Outputs to Aid Clinical Decision-Making},
  year         = {2024},
  version      = {1.0},
  date         = {2024-06-02},
  url          = {https://m4siko.github.io/InfernoCalibNet/}
}
```
> You can also click the **"Cite this repository"** button on GitHub for other formats.

## 🌟 Acknowledgements

*Maintained with love & passion by [@m4siko](https://github.com/m4siko)* 🚀
