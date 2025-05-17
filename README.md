# Breast Cancer Prediction Web App

## Project Description
This project is a machine learning-based web application designed to predict breast cancer diagnosis (malignant or benign) using the Breast Cancer Wisconsin dataset. The application preprocesses the data, trains multiple models (Logistic Regression, Random Forest, and SVM), evaluates their performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC, and then deploys the best model (Random Forest) via a Flask web service.

## Key Terms and Concepts
- **ROC (Receiver Operating Characteristic):** A graphical plot used to evaluate the performance of binary classifiers. It shows the trade-off between the true positive rate (sensitivity) and the false positive rate (1-specificity).
- **AUC (Area Under the Curve):** A measure of the model's ability to distinguish between classes. A higher AUC indicates better model performance.
- **KNN (K-Nearest Neighbors):** A non-parametric classification algorithm that classifies data points based on the majority class of their k nearest neighbors.
- **Flask:** A lightweight web framework for Python used to build web applications quickly and easily.
- **Models Used:**
  - **Logistic Regression:** A linear model for binary classification that predicts the probability of a binary outcome.
  - **Random Forest:** An ensemble learning method that constructs multiple decision trees and outputs the class that is the mode of the classes of the individual trees.
  - **SVM (Support Vector Machine):** A supervised learning model that finds the hyperplane that best divides a dataset into classes.

## How It Works
1. **Data Preprocessing & Analysis:**
   - The dataset is loaded, cleaned, and normalized using StandardScaler.
   - Visualizations (class distribution and correlation heatmap) are generated to understand the data.

2. **Model Training & Evaluation:**
   - Three models (Logistic Regression, Random Forest, SVM) are trained on the preprocessed data.
   - Each model is evaluated using classification reports, confusion matrices, and ROC curves.
   - The best model (Random Forest) is selected based on ROC-AUC score.

3. **Model Saving:**
   - The best model and scaler are saved using joblib for later use in the web app.

4. **Web Service (Flask):**
   - A Flask web app (`app.py`) provides a user interface for inputting new data.
   - The app scales the input data and generates real-time predictions.
   - Results are displayed along with model evaluation metrics, and users can download their prediction as a CSV.

## Expected Results
- **Model Performance:** The Random Forest model achieves high accuracy, precision, recall, and ROC-AUC scores, indicating robust performance in distinguishing between malignant and benign cases.
- **Web App:** Users can input new data and receive immediate predictions, along with a downloadable CSV of their results.

## Project Structure
- `train_model.py`: Script for data preprocessing, model training, evaluation, and saving.
- `app.py`: Flask web application for real-time predictions.
- `templates/index.html`: HTML template for the user interface.
- `static/`: Directory for storing generated CSV files and visualizations.
- `breast_cancer_model.pkl`: Saved Random Forest model.
- `scaler.pkl`: Saved StandardScaler for data normalization.
- `model_metrics.json`: JSON file containing model evaluation metrics.

## How to Run the Project
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   ```bash
   python train_model.py
   ```

3. **Run the Web App:**
   ```bash
   python app.py
   ```

4. **Access the Web App:**
   Open your browser and go to `http://127.0.0.1:5000`.

## Dependencies
- Python 3.x
- Flask
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
- joblib

## Future Improvements
- Add more models (KNN, Decision Tree, Gradient Boosting, XGBoost, Naive Bayes).
- Enhance error handling and logging.
- Deploy the app to a production server (e.g., Heroku, Render).

## Project Report
### Dataset Overview
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Shape:** 569 samples, 31 features
- **Target:** Binary classification (0 = Malignant, 1 = Benign)

### Data Visualization
- **Class Distribution:** A countplot shows the distribution of malignant and benign cases.
- **Correlation Heatmap:** A heatmap visualizes the correlation between features.

### Model Performance
#### Metrics for the Best Model (Random Forest)
- **Accuracy:** 0.96
- **Precision:** 0.96
- **Recall:** 0.99
- **F1 Score:** 0.97
- **ROC-AUC:** 0.997

#### Confusion Matrix
```
[[40  3]
 [ 1 70]]
```

### ROC Curve
- The ROC curve compares the performance of Logistic Regression, Random Forest, and SVM models.
- Random Forest achieves the highest ROC-AUC score.

### Conclusion
The Random Forest model demonstrates robust performance in predicting breast cancer diagnosis, with high accuracy and ROC-AUC scores. The web application successfully provides real-time predictions and allows users to download their results. 