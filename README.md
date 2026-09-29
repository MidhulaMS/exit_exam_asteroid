# ☄️ Asteroid Diameter Prediction Using Deep Learning

## 📌 Project Overview

This project focuses on predicting the **diameter of asteroids using Deep Learning**. A regression-based Artificial Neural Network (ANN) is developed using asteroid orbital and physical characteristics to learn the relationship between the input features and asteroid diameter.

The project includes **data preprocessing, exploratory data analysis, feature selection, feature encoding, scaling, Deep Neural Network regression, hyperparameter tuning, and model evaluation**.

---

## 🎯 Objective

The main objective of this project is to develop a **Deep Learning regression model** capable of predicting asteroid diameter from available asteroid characteristics.

### Target Variable

* `diameter` — Continuous asteroid diameter value

### Problem Type

**Supervised Learning → Regression**

---

## 📊 Dataset

The project uses an `asteroid.csv` dataset containing information about asteroids, including physical and orbital characteristics.

Some of the important features include:

* `H`
* `albedo`
* `moid`
* `moid_ld`
* `q`
* `a`
* `ad`
* `n`
* `per`
* `per_y`
* `om`
* `w`
* `ma`
* `epoch`
* `neo`
* `pha`
* `class`
* `diameter`

The target variable is:

```text
diameter
```

---

## 🔍 Exploratory Data Analysis

The dataset was analyzed to understand its structure and the relationship between the features and asteroid diameter.

The EDA included:

* Dataset shape and structure
* Statistical summary
* Missing-value analysis
* Duplicate-value checking
* Target-variable distribution
* Feature distributions
* Relationship between numerical features and diameter
* Correlation analysis
* Mutual Information analysis
* Identification of important features

Mutual Information was used to examine the dependency between input features and the target variable.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Missing Target Values

Rows where the target variable `diameter` was missing were removed because the model requires a known target value during supervised training.

### 2. Irrelevant Columns

Identifier and non-predictive columns were removed where appropriate, including:

* `id`
* `name`
* `full_name`
* `pdes`
* `orbit_id`
* `prefix`
* `equinox`

`diameter_sigma` was also excluded from the predictive features because it is directly associated with the diameter measurement.

### 3. Boolean Encoding

Boolean features were converted into numerical values:

```text
True  → 1
False → 0
```

### 4. Categorical Encoding

Categorical features such as `class` were converted into numerical representations using one-hot encoding.

### 5. Missing Feature Values

Missing values in predictor variables were handled using appropriate numerical imputation.

### 6. Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

A fixed random state was used to make the split reproducible.

---

## 📏 Feature Scaling

The input features were standardized using `StandardScaler`.

The scaler was fitted only on the training data and then applied to both training and testing data.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling helps the neural network train more efficiently when features have different numerical ranges.

---

# 🧠 Deep Learning Regression Architecture

A **Deep Neural Network (DNN)** based regression architecture was developed using fully connected Dense layers.

### Architecture

```text
Input Features
      ↓
Dense(128, ReLU)
      ↓
Batch Normalization
      ↓
Dropout(0.20)
      ↓
Dense(64, ReLU)
      ↓
Batch Normalization
      ↓
Dropout(0.20)
      ↓
Dense(32, ReLU)
      ↓
Dense(16, ReLU)
      ↓
Dense(1, Linear)
      ↓
Predicted Asteroid Diameter
```

### Model Characteristics

* Fully connected Dense layers
* ReLU activation in hidden layers
* Batch Normalization
* Dropout regularization
* Linear activation in output layer
* Adam optimizer
* Mean Squared Error (MSE) loss

The final `Dense(1)` layer produces a single continuous value representing the predicted asteroid diameter.

---

## ⚙️ Model Compilation

The model is compiled using:

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
```

### Loss Function

**Mean Squared Error (MSE)** is used as the regression loss function.

### Evaluation Metric

**Mean Absolute Error (MAE)** is monitored during training to measure the average prediction error.

---

# 🔧 Hyperparameter Tuning

Two important hyperparameters were tuned:

### 1. Learning Rate

The following learning rates were evaluated:

```text
0.001
0.0005
0.0001
```

### 2. Dropout Rate

The following dropout rates were evaluated:

```text
0.1
0.2
0.3
```

This resulted in multiple combinations of learning rate and dropout rate.

The configurations were compared using:

* MAE
* RMSE
* R² Score

The configuration with the lowest RMSE was selected as the final model configuration.

---

## 🛑 Overfitting Prevention

The following techniques were used to improve generalization:

* Dropout
* Batch Normalization
* Early Stopping
* Validation data
* Learning-rate reduction

Early stopping was used to stop training when the validation loss stopped improving.

---

# 📈 Model Evaluation

The final model is evaluated using standard regression metrics.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted diameter.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

### Root Mean Squared Error (RMSE)

Provides the square root of MSE and expresses the error in the same unit as the target.

### R² Score

Measures how much of the variation in asteroid diameter is explained by the model.

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

y_pred = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²   :", r2)
```

---

## 📊 Visualization

The following visualizations can be used to analyze model performance:

### Training vs Validation Loss

Helps identify learning behavior and possible overfitting.

### Training vs Validation MAE

Shows how prediction error changes during training.

### Actual vs Predicted Values

Compares the actual asteroid diameter with the model's predictions.

### Residual Plot

Shows the difference between actual and predicted values.

---

## 🗂️ Project Structure

```text
Asteroid-Diameter-Prediction/
│
├── asteroid.csv
│
├── asteroid_diameter_prediction.ipynb
│
├── asteroid_diameter_prediction_model.keras
│
├── asteroid_diameter_scaler.pkl
│
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras
* Google Colab / Jupyter Notebook
* Git & GitHub

---

## 🚀 Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Missing Value Handling
   ↓
Feature Selection
   ↓
Categorical / Boolean Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Deep Neural Network
   ↓
Hyperparameter Tuning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Asteroid Diameter Prediction
```

---

## 💡 Key Learning Outcomes

Through this project, the following concepts were applied:

* Regression problem formulation
* Exploratory Data Analysis
* Missing-value handling
* Feature selection
* Categorical and Boolean encoding
* Feature scaling
* Deep Neural Networks
* Regression architecture design
* Dropout regularization
* Batch Normalization
* Early Stopping
* Hyperparameter tuning
* Regression evaluation metrics
* Model performance visualization

---

## 🔮 Future Enhancements

Potential improvements include:

* Applying log transformation to the highly skewed diameter target
* Comparing the DNN with traditional regression models
* Comparing Random Forest and Gradient Boosting models
* Performing more extensive hyperparameter optimization
* Deploying the trained model using Streamlit
* Creating an interactive asteroid diameter prediction interface

---

## 👩‍💻 Author

**Midhula M S**

B.Tech Computer Science and Engineering

---

## 📜 License

This project is intended for educational and academic purposes.
