# Handwritten-Digit-Recognition-Using-Deep-Learning-MNIST-Keras-TensorFlow-
 a machine learning model that can accurately recognize handwritten digits (0–9) using a deep neural network trained on the popular MNIST dataset
### 📘 Project Description: Handwritten Digit Recognition with Deep Learning

---

#### **Title**:

**Handwritten Digit Recognition Using Deep Learning (MNIST + Keras + TensorFlow)**

---

#### **Objective**:

To build a machine learning model that can accurately recognize handwritten digits (0–9) using a deep neural network trained on the popular **MNIST dataset**. The final model can be used in applications like digit-based form scanning, postal address recognition, or calculator input on touch devices.

---

#### **Dataset Used**:

* **MNIST Dataset**

  * Contains 70,000 grayscale images (28x28 pixels)
  * 60,000 training samples, 10,000 test samples
  * Each image contains **one digit (0 to 9)** written by different people
  * Each digit is centered in the image, written in white on a black background

---

#### **Technologies Used**:

* **Python** (core programming language)
* **TensorFlow/Keras** (for model training and saving)
* **NumPy** (for data manipulation)
* **PIL (Pillow)** and **OpenCV** (for image processing if needed later)
* (Optional: **Flask** for building a web app interface)

---

#### **Model Architecture**:

A basic feedforward neural network (Multilayer Perceptron) with the following layers:

1. **Flatten**: Converts the 28x28 image into a 784-element vector
2. **Dense Layer** (128 units, ReLU activation): Learns high-level features
3. **Dense Layer** (10 units, Softmax activation): Outputs probabilities for digits 0–9

---

#### **Workflow**:

1. **Data Loading**: Load MNIST data from `tensorflow.keras.datasets`.
2. **Preprocessing**: Normalize pixel values to range \[0, 1].
3. **Label Encoding**: Convert digit labels into one-hot encoded vectors.
4. **Model Building**: Create a Sequential model using Keras.
5. **Model Training**: Fit the model on training data for 5+ epochs.
6. **Evaluation**: Validate the model on test data.
7. **Model Saving**: Save the trained model as `model.h5` for reuse.

---

#### **Expected Output**:

* Training logs showing accuracy and loss over epochs
* A saved model file: `model.h5`
* Once deployed (e.g., in a GUI or web app), the system should:

  * Accept an image or drawing of a digit
  * Predict which digit it is (0–9)
  * Display the prediction


