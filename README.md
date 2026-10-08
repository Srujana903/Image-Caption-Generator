# 🖼️ Image Caption Generator using CNN + LSTM

## Project Overview

This project generates descriptive captions for images using a deep learning pipeline. A pretrained **ResNet50** extracts image features, an **LSTM language model** predicts the caption word by word, and a **Streamlit** web application allows users to upload images and generate captions.

## Technologies Used

- Python 3.11
- TensorFlow / Keras
- ResNet50 (CNN)
- LSTM (RNN)
- NumPy
- Pillow
- Streamlit
- Git & GitHub

## Dataset

**Flickr8k Dataset**

- 8,092 images
- 40,455 captions
- Captions cleaned and tokenized before training

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
