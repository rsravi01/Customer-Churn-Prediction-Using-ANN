# Customer-Churn-Prediction-Using-ANN
Overview
This project demonstrates how to build and train an Artificial Neural Network (ANN) for predicting customer churn using the Churn_Modelling.csv dataset. The model is implemented using Python and TensorFlow/Keras, with preprocessing and evaluation steps included.

Requirements
Before running the script, ensure you have the following libraries installed:

tensorflow==2.12.0
numpy
matplotlib
pandas
scikit-learn
Install the dependencies using:

bash
Copy code
pip install tensorflow==2.12.0 numpy matplotlib pandas scikit-learn
Steps in the Project
1. Environment Setup
Uninstall any older or GPU versions of TensorFlow and install version 2.12.0.
Verify TensorFlow installation and GPU availability.
2. Dataset Loading and Preprocessing
Load the Churn_Modelling.csv dataset.
Extract feature variables (x) and target variable (y).
Perform one-hot encoding for categorical variables (Geography and Gender).
Normalize the features using StandardScaler.
3. Splitting the Data
Split the dataset into training and testing sets using an 80:20 ratio.
4. Building the ANN
Initialize a sequential model.
Add:
An input layer with 11 neurons and ReLU activation.
Two hidden layers with 7 and 6 neurons, respectively, and ReLU activation.
An output layer with 1 neuron and sigmoid activation.
Compile the model with:
Optimizer: Adam
Loss: binary_crossentropy
Metrics: accuracy
5. Training the Model
Train the model with:
Validation split: 33%
Batch size: 10
Early stopping callback to prevent overfitting.
Plot accuracy and loss graphs for both training and validation.
6. Evaluation
Predict on the test set.
Compute the confusion matrix and accuracy score.
7. Model Weights
Retrieve the trained weights of the model.
Usage Instructions
Clone the repository or download the script.
Place the dataset (Churn_Modelling.csv) in the same directory as the script.
Run the script in your Python environment.
Output and Results
Graphs:
Training and validation accuracy vs. epochs.
Training and validation loss vs. epochs.
Metrics:
Confusion matrix.
Accuracy score of the model on the test set.
File Descriptions
Churn_Modelling.csv: Dataset used for training and testing.
ann_script.py: Python script containing the code for data preprocessing, model training, and evaluation.
