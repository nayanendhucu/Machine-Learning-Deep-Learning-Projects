# Machine-Learning-Deep-Learning-Projects
A collection of machine learning and deep learning projects covering facial emotion detection using CNN, student depression prediction using Logistic Regression, and ADHD classification using Random Forest. The projects demonstrate data preprocessing, feature engineering, image processing, model training, evaluation
#  Machine Learning & Deep Learning Projects

This repository contains three practical machine learning and deep learning projects developed using Python in Google Colab. The projects cover **computer vision, deep learning, classical machine learning, data preprocessing, feature engineering, model evaluation, serialization, and prediction on new data**.

The projects demonstrate different approaches to solving classification problems using both image-based and tabular datasets.

---

#  Projects

## 1.  Facial Emotion Detection using CNN

### Overview

A deep learning-based computer vision project that classifies human facial expressions into **7 emotion categories** using a Convolutional Neural Network (CNN).

The model processes **48×48 grayscale facial images**, performs image normalization, trains a CNN using TensorFlow/Keras, evaluates the model on test images, and predicts the emotion of a newly uploaded facial image.

###  Objective

To automatically recognize facial emotions from images using a CNN-based image classification model.

###  Dataset

The project uses the **FER2013 (Facial Expression Recognition 2013)** dataset.

The notebook loaded:

* **28,709 training images**
* **7,178 test images**
* **7 emotion classes**
* Image size: **48 × 48**
* Image type: **Grayscale**

The seven classes are:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

###  Workflow

```text
FER2013 Dataset
      ↓
Image Loading
      ↓
Resize to 48 × 48
      ↓
Grayscale Conversion
      ↓
Pixel Normalization
      ↓
CNN Model
      ↓
Training
      ↓
Test Evaluation
      ↓
New Image Upload
      ↓
Emotion Prediction
```

###  CNN Architecture

```text
Input: 48 × 48 × 1
        ↓
Conv2D – 32 Filters
        ↓
MaxPooling
        ↓
Conv2D – 64 Filters
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense – 128 Neurons
        ↓
Dense – 7 Classes
        ↓
Softmax
```

### ⚙️ Training Configuration

| Parameter  | Value                    |
| ---------- | ------------------------ |
| Framework  | TensorFlow / Keras       |
| Model      | CNN                      |
| Input Size | 48 × 48 × 1              |
| Batch Size | 64                       |
| Optimizer  | Adam                     |
| Loss       | Categorical Crossentropy |
| Epochs     | 5                        |
| Classes    | 7                        |

### 📈 Result

After 5 epochs:

* Training Accuracy: **55.32%**
* Test/Validation Accuracy: **50.38%**
* Test Loss: **1.2950**

The notebook also demonstrates prediction on a new uploaded image. For the tested `sad.jpg` image, the model predicted:

**Fear**

### 🔗 Dataset

[FER2013 — Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013?utm_source=chatgpt.com)

---

# 2. 🎓 Student Depression Prediction

### Overview

A machine learning classification project that predicts whether a student is classified as having depression based on selected academic, lifestyle, and financial factors.

The project demonstrates data loading, feature selection, missing-value handling, numerical conversion, categorical encoding, train-test splitting, pipeline-based preprocessing, Logistic Regression training, and model evaluation.

###  Objective

To build a machine learning model capable of predicting the `Depression` target using selected student-related attributes.

###  Dataset

The project uses the **Student Depression Dataset**.

The notebook loads:

```text
student_depression_dataset.csv
```

The dataset contains student-related attributes including:

* Academic Pressure
* Work Pressure
* CGPA
* Study Satisfaction
* Job Satisfaction
* Sleep Duration
* Work/Study Hours
* Financial Stress
* Depression

The original dataset contains approximately **27,900 student records**.

###  Selected Features

The model uses:

```text
Academic Pressure
Work Pressure
CGPA
Study Satisfaction
Job Satisfaction
Sleep Duration
Work/Study Hours
Financial Stress
```

Target:

```text
Depression
```

###  Preprocessing

The notebook performs:

* Missing-value handling
* Numerical conversion
* Replacement of invalid numeric values
* Mean imputation
* One-hot encoding for `Sleep Duration`
* Numerical feature passthrough
* Train-test splitting

###  Model

The project uses:

**Logistic Regression**

A Scikit-learn pipeline combines preprocessing and model training:

```text
Raw Student Data
       ↓
Feature Selection
       ↓
Missing Value Handling
       ↓
Numerical Conversion
       ↓
One-Hot Encoding
       ↓
Train / Test Split
       ↓
Logistic Regression
       ↓
Depression Prediction
```

### Training Configuration

| Parameter            | Value               |
| -------------------- | ------------------- |
| Model                | Logistic Regression |
| Test Size            | 20%                 |
| Random State         | 42                  |
| Max Iterations       | 1000                |
| Categorical Encoding | One-Hot Encoding    |
| Missing Values       | Mean Imputation     |

###  Result

The notebook achieved:

**Test Accuracy: 77.46%**

### Model Serialization

The notebook also attempts to save the trained model using Python `pickle` for later use.

###  Dataset

[Student Depression Dataset — Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset?utm_source=chatgpt.com)

> **Important:** This is a machine learning classification project and should not be considered a clinical diagnostic system.

---

# 3.  ADHD Prediction using Random Forest

### Overview

A classification project demonstrating how a Random Forest model can be used to classify ADHD-related labels from behavioral and lifestyle variables.

Unlike the first two projects, **this notebook does not use an external Kaggle dataset**. Instead, it generates a synthetic dataset programmatically for demonstrating the machine learning workflow.

### Dataset

The notebook generates **300 synthetic records** containing:

* Attention Span
* Sleep Hours
* Screen Time
* Task Completion
* Restlessness

Example structure:

```text
attention_span
sleep_hours
screen_time
task_completion
restlessness
```

### 🏷️ Target Creation

The `adhd_label` is generated using predefined rules:

```text
attention_span < 20
AND
restlessness > 6
AND
task_completion < 50
```

If all conditions are satisfied:

```text
ADHD = 1
```

Otherwise:

```text
ADHD = 0
```

Therefore, this project is primarily a demonstration of a classification workflow rather than a clinically validated ADHD prediction system.

### 🔄 Workflow

```text
Synthetic Data Generation
        ↓
Feature Creation
        ↓
Rule-Based Target Generation
        ↓
Train / Test Split
        ↓
Random Forest Classifier
        ↓
Model Training
        ↓
Accuracy Evaluation
        ↓
New Sample Prediction
```

### 🧠 Model

**Random Forest Classifier**

Configuration used in the notebook:

```python
RandomForestClassifier(n_estimators=100)
```

### 📈 Result

The notebook achieved:

**Test Accuracy: 96.67%**

The trained model was also serialized using `pickle`:

```text
adhd_model.pkl
```

A new sample was passed to the trained model for prediction.

### ⚠️ Important Limitation

The high accuracy should **not** be interpreted as real-world ADHD diagnostic performance because the target label was created from predefined rules applied to synthetic data.

There is therefore **no Kaggle dataset link for this project**.

---

# 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* CNN
* Image Processing

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* One-Hot Encoding
* Train-Test Split
* Pipeline

### Data Processing

* Pandas
* NumPy

### Computer Vision

* OpenCV

### Environment

* Google Colab

### Model Serialization

* Pickle

---

# 📁 Repository Structure

```text
Machine-Learning-Projects/
│
├── Facial_Emotion_Detection/
│   └── Facial_Emotion_Detection.ipynb
│
├── Student_Depression_Prediction/
│   └── Student_Depression_Prediction.ipynb
│
├── ADHD_Prediction/
│   └── ADHD_Prediction.ipynb
│
└── README.md
```

---

# 📚 Project Skills Demonstrated

Through these projects, I gained practical experience in:

* Data preprocessing
* Missing-value handling
* Feature selection
* Categorical encoding
* Image preprocessing
* Image normalization
* CNN architecture
* Deep learning model training
* Multi-class image classification
* Logistic Regression
* Random Forest classification
* Train-test splitting
* Model evaluation
* Model serialization
* Prediction on new data
* Working with Kaggle datasets
* Working with synthetic datasets
* Google Colab-based development

---

# ⚠️ Disclaimer

These projects were developed for **learning, experimentation, and demonstration purposes**. The medical and mental-health-related projects are not intended to provide clinical diagnoses or replace evaluation by qualified healthcare professionals.
