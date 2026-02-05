# Domain Scanner API (Malicious Domain Detection)

This project features a specialized API designed for real-time URL analysis and cybersecurity threat detection. It utilizes a state-of-the-art ensemble of Deep Learning and Gradient Boosting models to identify phishing, DGA (Domain Generation Algorithms), and other malicious domains.
# 🚀 Methodology

The system employs a sophisticated multi-modal approach to network security through the following pipeline:

    Feature Engineering: Extraction of lexical features from URLs and DNS metadata. Textual data is processed for NLP models, while Image Reshaping techniques transform these features into visual representations for CNN analysis.

    Ensemble Architecture: The system utilizes a Stacking Ensemble of three specialized models:

        ConvNeXt Nano: A modern, lightweight Convolutional Neural Network used to process the reshaped visual patterns of domain data.

        DeBERTa: A high-performance Transformer model used to capture complex linguistic and sequential patterns within the URL strings.

        XGBoost (Extreme Gradient Boosting): A powerful tabular model that processes structured DNS and lexical metadata to provide a robust classification baseline.

    Meta-Classification: A final stacking layer aggregates the predictions from the visual (ConvNeXt), textual (DeBERTa), and tabular (XGBoost) components to maximize detection accuracy and minimize false positives.

    Real-Time Analysis: Optimized inference execution allowing for the immediate classification of incoming network requests.

# 🛠️ System Architecture

The project is structured to bridge the gap between complex AI research and production-grade security tools:

    Backend (Flask API): The core engine responsible for:

        Processing raw URLs and extracting relevant security features.

        Running the PyTorch (ConvNeXt/DeBERTa) and XGBoost ensemble for threat prediction.

        Serving a RESTful interface for real-time domain scanning.

    Frontend (React): Interactive page for users to submit URLs and visualize the security risk assessment in real-time.

# 💻 Tech Stack

    Language: Python

    Models: ConvNeXt Nano, DeBERTa, XGBoost

    Deep Learning Framework: PyTorch

    Machine Learning: Scikit-learn (Stacking) & XGBoost

    API Framework: Flask

    Frontend: React

# 🚀 Getting Started
1. Backend Configuration
Bash
```sh
cd src
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
2. Frontend Configuration
```sh

cd front-end
npm install
npm run dev
```

# Running Online 

The project is curently online at Github Pages, with communication with the API in Hugging Face Spaces. Link: https://lhlobato.github.io/Domain-Attack-Detection-API/
