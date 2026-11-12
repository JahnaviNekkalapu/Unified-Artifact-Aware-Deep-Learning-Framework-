# 🛰️ Unified Artifact Detection and Restoration for Remote Sensing Images

### Earth Observation Studio – An AI-Powered Satellite Image Restoration System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-Web%20Application-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Computer-Vision-purple" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Remote-Sensing-green" alt="Remote Sensing">
</p>

---

## 🌍 1. Project Overview

Satellite and remote sensing images play an important role in environmental monitoring, agriculture, urban planning, disaster management, and geographical analysis.

However, satellite images frequently contain different types of degradations caused by atmospheric conditions, sensor limitations, and transmission errors. These degradations reduce image quality and can affect the accuracy of further image analysis.

This project presents a **Unified Artifact Detection and Restoration Framework** that uses deep learning to identify image degradation conditions and restore affected satellite images.

The system uses an **Artifact-Aware U-Net architecture** to process degraded satellite images and generate restored RGB images. An interactive web application developed using Streamlit provides facilities for image restoration, artifact analysis, performance evaluation, and restoration history.

## 🎯 2. Problem Statement

Remote sensing images frequently suffer from different types of degradation, including noise, blur, haze, and missing image regions.

Traditional image restoration techniques are often designed to handle individual degradation types and may not perform effectively when multiple artifacts occur simultaneously.

Therefore, this project aims to develop a unified deep learning-based framework that can handle multiple image degradation conditions and improve the visual quality of satellite images while preserving important spatial information.

## 🎯 3. Project Objectives

- Develop a unified deep learning model for satellite image restoration.
- Identify and handle different types of image degradation.
- Restore degraded satellite images while preserving important spatial details.
- Use an artifact-aware mechanism to provide degradation information to the restoration network.
- Evaluate restoration quality using standard image quality metrics.
- Develop an interactive web application for practical image restoration and analysis.
- Provide a user-friendly platform for visualizing restoration results and performance.

## ✨ 4. Key Features

| Feature | Description |
|---|---|
| Image Restoration | Restores degraded satellite images using an Artifact-Aware U-Net model. |
| Artifact Analysis | Provides analysis of image degradation conditions. |
| Artifact-Aware Conditioning | Uses artifact-condition maps to guide the restoration network. |
| Performance Evaluation | Evaluates restored images using PSNR, SSIM, MAE, RMSE, and SAM. |
| Restoration History | Displays records of previous restoration operations. |
| Interactive Dashboard | Provides a Streamlit-based interface for accessing the system. |
| Model Information | Displays details about the restoration model and its architecture. |

## 🧪 5. Supported Image Artifacts

The current V5 model supports five artifact conditions.

| Artifact Type | Description |
|---|---|
| Gaussian Noise | Random additive noise affecting image pixels. |
| Speckle Noise | Multiplicative noise that creates granular patterns. |
| Blur | Loss of image sharpness and fine details. |
| Haze | Atmospheric interference that reduces image contrast. |
| Missing Regions | Missing or unavailable image information in selected regions. |

The artifact-condition maps help the restoration network understand the degradation conditions present in the input image.

## 🏗️ 6. System Architecture

The proposed system follows an artifact-aware image restoration pipeline.

### Restoration Workflow

1. **Input Image:** A satellite image is provided to the application.
2. **Preprocessing:** The image is converted into RGB format and normalized.
3. **Artifact Conditioning:** Artifact-condition maps are prepared and combined with the RGB input.
4. **Feature Extraction:** The encoder extracts important image features.
5. **Feature Reconstruction:** The decoder reconstructs image information using skip connections.
6. **Image Restoration:** The network generates a restored RGB image.
7. **Performance Evaluation:** Restoration quality is measured using image quality metrics.
8. **Visualization:** Original and restored images are displayed through the web application.

### Overall Workflow

```text
       Sentinel-2 Satellite Image
                   |
                   v
          Image Preprocessing
                   |
                   v
          Artifact Conditioning
                   |
                   v
          RGB + Artifact Maps
                   |
                   v
          Artifact-Aware U-Net
                   |
                   v
           Restored RGB Image
                   |
                   v
          Performance Evaluation
                   |
                   v
          Streamlit Dashboard
```

## 🧠 7. Deep Learning Model – Artifact-Aware U-Net

The project uses an enhanced U-Net-based architecture for satellite image restoration.

U-Net is an encoder-decoder architecture widely used for image processing tasks. It extracts important image features and reconstructs the output using skip connections.

### 7.1 Main Components

**1. Encoder**
- Extracts features from the input image.
- Learns spatial patterns and degradation characteristics.
- Uses convolutional layers and downsampling operations.

**2. Bottleneck**
- Captures deeper and more complex image representations.
- Learns high-level contextual information.

**3. Decoder**
- Reconstructs the restored image.
- Uses upsampling and convolution operations.
- Uses skip connections to preserve spatial information from the encoder.

### 7.2 Artifact-Aware Input Design

The V5 model uses an 8-channel input.

| Component | Number of Channels | Purpose |
|---|---:|---|
| RGB Input | 3 | Original degraded satellite image |
| Artifact-Condition Maps | 5 | Provides degradation-condition information |
| Combined Input | 8 | Input supplied to the U-Net |
| Restored Output | 3 | Restored RGB satellite image |

### 7.3 Model Specifications

| Parameter | Value |
|---|---|
| Model Architecture | Artifact-Aware U-Net |
| Input Channels | 8 |
| Output Channels | 3 |
| Input Patch Size | 128 × 128 |
| RGB Channels | 3 |
| Artifact-Condition Channels | 5 |
| Output Activation | Sigmoid |
| Total Parameters | 8,096,426 |
| Deep Learning Framework | PyTorch |

## 🛰️ 8. Dataset and Preprocessing

The project uses Sentinel-2 satellite imagery.

### 8.1 Sentinel-2 Band Configuration

The RGB representation is generated using three Sentinel-2 spectral bands.

| Sentinel-2 Band | Color Channel | Description |
|---|---|---|
| B04 | Red | Red spectral band |
| B03 | Green | Green spectral band |
| B02 | Blue | Blue spectral band |

### 8.2 Dataset Distribution

The current patch index contains 7,913 patches.

| Dataset Split | Number of Patches |
|---|---:|
| Training | 5,766 |
| Validation | 1,199 |
| Testing | 948 |
| **Total** | **7,913** |

The dataset contains satellite scenes from the following Indian cities:

| City |
|---|
| Ahmedabad |
| Bengaluru |
| Delhi |
| Hyderabad |
| Mumbai |

### 8.3 Preprocessing Steps

| Step | Operation | Description |
|---|---|---|
| 1 | Data Loading | Load Sentinel-2 satellite image data. |
| 2 | Band Extraction | Extract B04, B03, and B02 bands. |
| 3 | RGB Conversion | Combine the three bands into an RGB image. |
| 4 | Normalization | Divide pixel values by 10,000. |
| 5 | Clipping | Restrict normalized values to the range [0, 1]. |
| 6 | Patch Extraction | Divide images into 128 × 128 patches. |

## 📊 9. Model Evaluation

The V5 model was evaluated on 200 cases using standard image restoration metrics.

### 9.1 Evaluation Results

| Evaluation Metric | Obtained Value | Purpose |
|---|---:|---|
| PSNR | 34.3802 dB | Measures reconstruction quality. |
| SSIM | 0.9219 | Measures structural similarity. |
| MAE | 0.0130 | Measures average absolute error. |
| RMSE | 0.0204 | Measures reconstruction error. |
| SAM | 0.0260 rad | Measures spectral-angle difference. |

### 9.2 Understanding the Evaluation Metrics

| Metric | Full Form | What It Measures | Desired Direction |
|---|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio | Quality of the restored image compared with the reference image. | Higher is better |
| SSIM | Structural Similarity Index | Similarity in luminance, contrast, and structure. | Higher is better |
| MAE | Mean Absolute Error | Average absolute pixel difference. | Lower is better |
| RMSE | Root Mean Square Error | Magnitude of reconstruction errors. | Lower is better |
| SAM | Spectral Angle Mapper | Angular difference between spectral vectors. | Lower is better |

**Note:** These are the saved V5 evaluation results. Model comparisons should use the same test data and evaluation protocol. Since the current RGB representation uses three bands, SAM represents the angle between the available RGB spectral vectors.

## 💻 10. Technology Stack

| Category | Technology Used |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | U-Net, Artifact-Aware Network |
| Image Processing | NumPy, Pillow |
| Data Handling | Pandas |
| Web Application | Streamlit |
| Visualization | Matplotlib |
| Development Environment | Google Colab, Visual Studio Code |
| Version Control | Git and GitHub |

## 📂 11. Project Structure

The following is an illustrative application structure. Adjust the model checkpoint filename and folder names to match the actual repository.

```text
Satellite-Restoration/
│
├── app.py
│
├── components/
│   └── styles.py
│
├── services/
│   └── model_service.py
│
├── pages_rl/
│   ├── overview.py
│   ├── restoration.py
│   ├── artifact_analysis.py
│   ├── performance.py
│   ├── history.py
│   └── about.py
│
├── model/
│   ├── inference.py
│   └── best_damage_aware_v5_model.pth
│
├── checkpoints/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

## ⚙️ 12. Installation and Setup

Follow these steps to run the application locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### Step 2: Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure PyTorch is installed with a version compatible with your operating system.

### Step 5: Configure the Model Checkpoint

Place the trained V5 model checkpoint in the expected model directory and ensure that the checkpoint path in the model service configuration is correct.

### Step 6: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 🚀 13. Model Inference

The trained model can be loaded using the inference class.

```python
from inference import ReliefLinkRestoration

restorer = ReliefLinkRestoration(
    "model/best_damage_aware_v5_model.pth"
)

restored, artifact_logits = restorer.restore(image)
```

The model should be loaded once when the backend starts instead of being reloaded for every restoration request.

**Note:** The input image must follow the preprocessing and input-channel requirements of the trained V5 model.

## 🖥️ 14. Application Modules

The Streamlit application provides the following modules.

| Module | Functionality |
|---|---|
| Overview Dashboard | Displays the system overview and restoration statistics. |
| Image Restoration | Allows users to upload and restore satellite images. |
| Artifact Analysis | Provides information about image degradation. |
| Performance Reports | Displays restoration evaluation metrics. |
| Restoration History | Maintains records of previous restoration operations. |
| About the Model | Displays model architecture and technical information. |

## 🎬 15. Application Demo

An interactive Streamlit application provides access to image restoration and analysis features.

**Live Demo:** https://jahnavinekkalapu-unified-artifact-aware-deep-learnin-app-af4egi.streamlit.app/

**Demo Link:** To be added.

## 🔮 16. Future Enhancements

| Enhancement | Description |
|---|---|
| Additional Artifact Types | Extend support to cloud occlusion and stripe noise. |
| Multi-Artifact Restoration | Improve restoration when multiple degradations occur together. |
| Advanced Architectures | Explore Restormer, SwinIR, and other restoration models. |
| Multispectral Restoration | Extend the framework to additional Sentinel-2 spectral bands. |
| Generalization | Evaluate performance across different geographical regions. |
| Deployment Optimization | Improve inference speed and deployment efficiency. |

## 👩‍💻 17. Author

**Jahnavi Nekkalapu**

Computer Science and Engineering

**Areas of Interest:**
- Artificial Intelligence and Machine Learning
- Deep Learning
- Computer Vision
- Remote Sensing and Satellite Image Processing

## 📜 18. License

This project is intended for academic and research purposes.

An appropriate open-source license can be added after confirming that the project code, datasets, and included assets can be distributed under its terms.

## 🙏 19. Acknowledgements

- **European Space Agency (ESA):** Sentinel-2 satellite imagery.
- **PyTorch:** Deep learning framework.
- **Streamlit:** Interactive web application framework.
- **Open-source research community:** Research and resources related to satellite image restoration and computer vision.

---

<p align="center">
  <b>⭐ If you find this project interesting, consider giving the repository a star on GitHub!</b>
</p>
