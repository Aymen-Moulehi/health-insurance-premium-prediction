# Health Insurance Premium Prediction

This project provides a **Machine Learning model** to predict health insurance premiums based on various factors such as age, BMI, smoking habits, and region. The model uses a **Random Forest Regressor** to estimate the insurance premium for an individual. Additionally, the project includes a **Flask web application** that allows users to input their information and receive a prediction in real-time.


## Description

This project includes:
1. **Health_Insurance_Premium_Modeling.ipynb**:
   - A Jupyter notebook where the model is trained on the provided insurance dataset.
   - The notebook generates and saves a trained model (`insurance_premium_predictor.pkl`), which is used by the Flask app to make predictions.
   
2. **Flask Application**:
   - A simple web application built with **Flask**, where users can input their details (e.g., age, BMI, smoker status, etc.).
   - The app uses the trained model to predict the insurance premium based on the input data.
   - The web interface is styled using **CSS** to create a user-friendly experience.

## Prerequisites

- Python 3.12.7 (or any version compatible with Python 3)
- Required dependencies in `requirements.txt`

## Setup & Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Aymen-Moulehi/health-insurance-premium-prediction.git
cd health-insurance-premium-prediction
 ```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
```

3. **Activate the virtual environment:**
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```
4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Training the Model
Before running the Flask app, you need to train the model. Follow these steps:

1. **Navigate to the notebooks/ folder:**

```bash
cd notebooks
```

2. **Open the Jupyter notebook Health_Insurance_Premium_Modeling.ipynb**

```bash
jupyter notebook Health_Insurance_Premium_Modeling.ipynb
```
3. **In the notebook, run all the cells to train the model and save it as insurance_premium_predictor.pkl in the app/model/ folder**

The notebook will preprocess the data, train the Random Forest Regressor model, and save the model using joblib.

## Project Overview
- **Health_Insurance_Premium_Modeling.ipynb:** 
 Trains the model using data preprocessing, feature engineering, and machine learning algorithms.
- **Flask Web Application:**
    * Provides a user interface for users to input their personal details.
    * Uses the trained model to make predictions on the user's input.
    * Displays the predicted premium value.

## Dependencies
This project requires the following Python libraries:

- Flask: For creating the web application.
- joblib: For saving and loading the machine learning model.
- pandas: For data manipulation.
- numpy: For numerical operations.
- scikit-learn: For training and evaluating the model.

You can install all dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

