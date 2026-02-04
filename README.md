# 🧠 Handwritten Digit Classification with MNIST (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A beginner-friendly deep learning project for handwritten digit recognition using PyTorch.**  
The model classifies digits (0–9) and supports **custom image prediction with confidence score** and a **Streamlit web app**.

---

## 📌 Real-Time Digit Recognition with Deep Learning
A simple yet complete end-to-end deep learning workflow:
**training → evaluation → prediction → deployment**

---

## 🔗 Quick Links
- 🚀 **Features**
- ⚡ **Quick Start**
- 🧩 **Installation**
- 📊 **Dataset**
- 🧠 **Model**
- 🧪 **Results**
- 🌐 **Web App**
- 📁 **Project Structure**
- 📜 **License**

---

## 📑 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)
- [Prediction](#prediction)
- [Web Application](#web-application)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🔍 Overview
This project implements a **fully connected neural network** trained on the **MNIST dataset** to recognize handwritten digits.  
It also includes:
- Manual image prediction
- Confidence percentage
- Streamlit-based web interface

---

## ✨ Key Features
- ✅ PyTorch-based training pipeline
- ✅ MNIST handwritten digit dataset
- ✅ ~95% test accuracy
- ✅ Custom image prediction
- ✅ Confidence score output
- ✅ Streamlit web app
- ✅ Beginner-friendly & well-commented code

---

## 📊 Dataset
**MNIST Dataset**
- 60,000 training images
- 10,000 testing images
- Image size: `28×28`
- Classes: `0–9`

Dataset is **automatically downloaded** using `torchvision`.

---

## 🧠 Model Architecture
**Fully Connected Neural Network**

## 📂 Project Structure

digit-classification/
│
├── Dataset/ # MNIST dataset (auto-downloaded)
├── model.py # Model architecture
├── train.py # Training code
├── predict.py # Predict digit from image
├── requirements.txt # Dependencies
├── README.md # Project description
├── .gitignore # Ignored files
└── classifier.pth # Trained model