# Data-Preprocessing-Machine-Learning-Pipeline-

## Overview

This project aims to develop a **Speech Emotion Recognition** (SER) model using Convolutional Neural Networks (CNNs) to classify speech emotions based on audio data. The model leverages the **RAVDESS** dataset, which consists of emotional speech recordings from professional actors. The project demonstrates how audio features, particularly **spectrograms**, can be used to predict emotional states such as happy, sad, angry, fearful, surprised, disgust, calm, and neutral. Link to project model design: https://github.com/Elhameed/Speech_Emotion_Recognition_model

## Project Details

- **Project Title**: Speech Emotion Recognition (SER)
- **Student Name**: Abdulhameed Teniola Ajani
- **Dataset Used**: [RAVDESS Dataset on Kaggle](https://www.kaggle.com/datasets/uw-madison/ravdess-emotional-speech-audio)
- **Technologies**: Python, TensorFlow, Keras, Librosa, Scikit-learn, Matplotlib
- **Model Used**: Convolutional Neural Network (CNN)
  
## Table of Contents

1. [Data Sources](#data-sources)
2. [Data Processing](#data-processing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Model Architecture](#model-architecture)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Bias Mitigation](#bias-mitigation)
8. [How to Run the Project](#how-to-run-the-project)

## Data Sources

The primary dataset for this project is the **RAVDESS** dataset, which contains **1440 audio files** representing eight emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/uw-madison/ravdess-emotional-speech-audio).

An additional dataset considered for further experimentation is **CREMA-D**, available [here](https://github.com/CheyneyComputerScience/CREMA-D), which offers more variety in accents and pronunciations.

## Data Processing

1. **Data Preprocessing**: 
    - The raw audio files were reorganized by emotion, moving them into respective emotion folders for easier processing.
    - Each audio file was converted into a **spectrogram** using Short-Time Fourier Transform (STFT) with `librosa`.
    - The spectrograms were resized to a consistent shape of **64x64** pixels for CNN input.
    - **Normalization**: The pixel values of the spectrograms were standardized using z-score normalization.

2. **Data Splitting**:
    - The data was split into training, validation, and test sets using `train_test_split` from **Scikit-learn**. The training set consisted of 70% of the data, with 15% each allocated to validation and test sets.

## Exploratory Data Analysis

The dataset contains emotional speech data in the form of **audio files**. After preprocessing, the key feature used for classification is the **spectrogram** derived from these audio files.

**Key Observations**:
- The spectrograms visually represent the frequency components of speech, capturing emotional cues in voice tone and pitch.
- The dataset's labels are categorical (emotion types), and each emotion is equally represented in the dataset.

Sample spectrograms were visualized using `matplotlib` to ensure proper extraction and feature representation.

## Model Architecture

The model used for emotion classification is a **Convolutional Neural Network (CNN)**. The architecture is designed to classify the spectrograms into one of the eight emotions.

The CNN architecture consists of:
- Several **convolutional layers** to extract spatial features from the spectrogram images.
- **Max pooling layers** to reduce dimensionality and retain the most important features.
- **Dense layers** for classification, with **softmax activation** to output probabilities for each emotion class.

## Model Training

1. **Training**:
    - The CNN model was trained on the preprocessed spectrograms.
    - The model was compiled using the **Adam optimizer** and **categorical cross-entropy loss** for multi-class classification.
    - **Accuracy** was used as the evaluation metric.

2. **Hyperparameters**:
    - Learning rate: 0.001
    - Batch size: 32
    - Epochs: 30

## Evaluation

The model’s performance was evaluated using several metrics:
- **Accuracy**: Overall percentage of correctly classified emotions.
- **Confusion Matrix**: To visualize how well the model distinguishes between different emotions.
- **Precision, Recall, and F1-Score**: To assess the model’s ability to predict each emotion accurately.

## Bias Mitigation

The dataset is balanced in terms of emotional classes, ensuring fairness during training. Each emotion is represented by an equal number of samples to mitigate any class imbalance issues.

## How to Run the Project

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/speech-emotion-recognition.git
    cd speech-emotion-recognition
    ```

2. **Install Dependencies**:
    Make sure to install all the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Notebook**:
    Open the Jupyter notebook and run the preprocessing, training, and evaluation steps.
    ```bash
    jupyter notebook
    ```

    Follow the instructions in the notebook for step-by-step execution.

4. **Dataset**:
    Download the **RAVDESS dataset** from Kaggle and place it in the `/data` folder. 
