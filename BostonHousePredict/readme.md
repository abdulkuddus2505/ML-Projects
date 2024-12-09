# Boston Housing Price Prediction

This project uses machine learning to predict housing prices in Boston. It explores various regression models and techniques, including data preprocessing, feature selection, and hyperparameter tuning, to achieve the best possible prediction accuracy.

## Project Structure

The project is organized as follows:

- `boston.csv`: The dataset containing information about housing prices in Boston.
- `boston_model.pkl`: The saved trained model for predictions.
- `boston_app.ipynb`: The Jupyter Notebook containing the code for data analysis, model training, and evaluation.

## Steps

1. **Data Loading and Exploration:** The project starts by loading the Boston housing dataset and performing exploratory data analysis. This involves examining the distribution of variables, identifying correlations, and visualizing relationships between features and the target variable.

2. **Data Preprocessing:**  Data preprocessing techniques are applied to prepare the data for model training. This includes:
    - Handling missing values (if any).
    - Feature scaling using StandardScaler to normalize the input features.
    - One-hot encoding of categorical variables using OneHotEncoder to represent them numerically.

3. **Feature Selection:**  Feature selection is performed to identify the most relevant features for prediction. Pairwise correlation analysis is used to remove highly correlated features, which can improve model performance and reduce overfitting.

4. **Model Selection and Training:** Various regression models are explored, including:
    - Linear Regression
    - SGD Regressor
    - KNN Regression
    - Random Forest Regression
    - Ridge Regression

5. **Hyperparameter Tuning:** Optuna is used to perform hyperparameter optimization for KNN Regression, Random Forest Regression, and Ridge Regression. This process aims to find the best combination of hyperparameters that minimize the prediction error.

6. **Model Evaluation:** The performance of the trained models is evaluated using metrics such as R-squared and Mean Squared Error (MSE). The best-performing model is selected based on its evaluation scores.

7. **Model Deployment:**  The best-performing model (Random Forest Regressor in this case) is saved as `boston_model.pkl` using pickle for future use.

8. **Prediction with Saved Model** Instructions for making a prediction using the trained model.  Load data, apply transformations, then make predictions.

## Usage

To run this project:

1. Upload the `boston.csv` file to your Google Colab environment.
2. Open the `boston_app.ipynb` file in Google Colab.
3. Run the cells in the notebook sequentially to perform data analysis, model training, and evaluation.
4. Use the saved model (`boston_model.pkl`) to make predictions on new data.

## Dependencies

This project requires the following libraries:
