# Credit Card Customer Churn Prediction

A deep learning project that predicts whether a credit card customer is likely to churn using a **TensorFlow/Keras neural network**.

## 📌 Overview

Customer churn prediction helps financial institutions identify customers who are likely to leave their services. This project uses customer demographic, financial, and account information to build a binary classification model that predicts customer churn.

The model is trained on the **Credit Card Customer Churn Prediction** dataset and achieves **85.75% accuracy** on the test set.

## 📊 Dataset

The dataset contains **10,000 customer records** and 14 original features.

### Target Variable

* `Exited` — Indicates whether the customer left the bank

  * `0` → Customer stayed
  * `1` → Customer exited

### Features Used

* `CreditScore`
* `Geography`
* `Gender`
* `Age`
* `Tenure`
* `Balance`
* `NumOfProducts`
* `HasCrCard`
* `IsActiveMember`
* `EstimatedSalary`

The columns `RowNumber`, `CustomerId`, and `Surname` were removed as they were not used for prediction.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* Google Colab
* KaggleHub

## 🔄 Workflow

```text
Kaggle Dataset
      ↓
Data Loading
      ↓
Data Exploration
      ↓
Remove Unnecessary Features
      ↓
Categorical Feature Encoding
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Neural Network
      ↓
Model Training
      ↓
Prediction
      ↓
Model Evaluation
```

## 🧹 Data Preprocessing

Categorical variables were converted into numerical features using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=['Geography', 'Gender'],
    drop_first=True
)
```

The data was then divided into training and testing sets using an **80/20 split** and standardized using `StandardScaler`.

## 🧠 Model Architecture

The project uses a simple feed-forward neural network built with Keras:

```text
Input Layer
    ↓
Dense Layer — 11 neurons — ReLU
    ↓
Dense Layer — 11 neurons — ReLU
    ↓
Output Layer — 1 neuron — Sigmoid
```

The model contains **276 trainable parameters**.

### Compilation

The model uses:

* **Loss:** Binary Crossentropy
* **Optimizer:** Adam
* **Metric:** Accuracy

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)
```

## 🚀 Training

The model was trained for **100 epochs** with 20% of the training data used for validation.

The training accuracy reached approximately **86.73%** by the final epoch.

## 📈 Results

| Metric           |         Result |
| ---------------- | -------------: |
| Test Accuracy    |     **85.75%** |
| Training Epochs  |            100 |
| Dataset Size     |         10,000 |
| Train/Test Split |          80/20 |
| Model            | Neural Network |

The final test accuracy was calculated using the model's predictions with a classification threshold of 0.5.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Credit-Card-Customer-Churn-Prediction.git
cd Credit-Card-Customer-Churn-Prediction
```

### 2. Open the notebook

Open:

```text
Customer_Churn_Prediction.ipynb
```

The notebook is designed for **Google Colab**.

### 3. Install KaggleHub

```python
!pip -q install kagglehub
```

### 4. Download the dataset

```python
import kagglehub

path = kagglehub.dataset_download(
    "rjmanoj/credit-card-customer-churn-prediction"
)
```

### 5. Run the notebook

Execute the cells sequentially to reproduce the preprocessing, training, predictions, and evaluation.

## 📁 Project Structure

```text
Credit-Card-Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
└── README.md
```

## 🔮 Future Improvements

* Compare the neural network with Logistic Regression, Random Forest, XGBoost, and SVM.
* Perform hyperparameter tuning.
* Add precision, recall, F1-score, and ROC-AUC evaluation.
* Address class imbalance using appropriate techniques.
* Add a Streamlit web application for real-time churn prediction.
* Save and deploy the trained model.

## 👨‍💻 Author

**Devansh Verma**

If you found this project useful, consider giving the repository a ⭐.
