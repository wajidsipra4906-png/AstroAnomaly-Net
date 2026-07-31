# 🔭 AstroAnomaly-Net: Deep Learning Unsupervised Light Curve Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AstroAnomaly-Net** is an end-to-end unsupervised deep learning pipeline designed to search NASA TESS (Transiting Exoplanet Survey Satellite) light curves for rare, unclassified stellar anomalies. 

By training a **1D Convolutional Autoencoder** to compress and reconstruct standard stellar flux signals, the system isolates high-reconstruction-loss targets and cross-references them against multi-wavelength catalog databases (**MAST**, **SIMBAD**, and **Gaia DR3**).

---

## 🌟 Key Features

* **Data Pipeline:** Ingests TESS Sector 1 Full-Frame Image (FFI) light curves (2,048 cadence points per curve).
* **Architecture:** Custom PyTorch 1D ConvAutoencoder utilizing a compressed 32-dimensional bottleneck representation.
* **Astrophysical Verification:** Automated TAP queries to MAST, SIMBAD, and Gaia DR3 to verify parallax, Renormalised Unit Weight Error (RUWE), and variability flags.
* **Interactive Dashboard:** Built-in Plotly & Streamlit web application for real-time visual assessment of anomaly reconstructions.

---

## 🏗️ Model Architecture
Input (1, 2048)
└─► Conv1D (16) + LeakyReLU + MaxPool
└─► Conv1D (32) + LeakyReLU + MaxPool
└─► Conv1D (64) + LeakyReLU + MaxPool
└─► Linear Bottleneck (32-dim)
└─► ConvTranspose1D Reconstruction ──► Output (1, 2048)
Target reconstruction error is evaluated via Mean Squared Error (MSE).

---

## 📊 Discovery Findings (Top Candidates)

| Rank | TIC ID | MSE Loss | SIMBAD Type | Gaia RUWE | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **38573584** | **0.004537** | `*` (Standard Star) | **1.035** | 🌟 Unclassified Candidate |
| **2** | **396697394** | **0.004491** | `*` (Standard Star) | **1.033** | 🌟 Unclassified Candidate |
| **3** | **393747997** | **0.004423** | `PM*` (High Proper Motion) | **0.944** | 🌟 Unclassified Candidate |
| **4** | **38696111** | **0.004420** | `*` (Standard Star) | **1.026** | 🌟 Unclassified Candidate |

*Note: All top candidates exhibit clean astrometric fits (RUWE ≈ 1.0) with zero recorded variability flags in existing astronomical catalogs.*

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/wajidsipra4906-png/AstroAnomaly-Net.git
cd AstroAnomaly-Net
pip install -r requirements.txt
```
### 2. Launch Interactive Dashboard
```bash
streamlit run app.py
```
---
## 📜 License
Distributed under the MIT License. See LICENSE for more information.
