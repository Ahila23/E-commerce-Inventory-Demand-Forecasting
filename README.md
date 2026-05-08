# Demand Forecasting Using LSTM

## Project Overview
This project focuses on building an end-to-end demand forecasting system using Deep Learning (LSTM) to predict future product sales from historical retail data. The project includes data preprocessing, feature engineering, time-series analysis, model building, evaluation, and deployment using Streamlit.

---

# Workflow Step by Step

## 1. Data Loading
- Loaded the retail sales dataset using Pandas
- Inspected the dataset structure, shape, and column information

```python
df = pd.read_csv("Neural_stock_cleaned_dataset.csv")
----
## 2. Data Inspection and Cleaning
*Checked missing values using isnull().sum()
*Converted date column from string to datetime format
*Sorted data based on product_id and date

Missing Value Handling
*units_sold → Forward fill within each product
*stock_on_hand → Filled using median values
*Outlier Handling
*Used IQR method to detect extreme outliers in unit_price
*Corrected price outliers using capping
*Retained high units_sold values because they were associated with promotional sales (is_promotion = 1)


3. Exploratory Data Analysis (EDA)

Performed:
*Data inspection (info, describe)
*Missing value analysis
*Histogram visualizations
*Box plots for outlier detection
*Correlation heatmap
*Seasonal decomposition plots
*ACF/PACF plots for time-series dependency analysis

Libraries used:

*Matplotlib
*Seaborn
*Statsmodels

4. Feature Engineering

Created time-based features:
*day
*month
*quarter
*day_of_week
*is_weekend

Created lag features:
*lag_7
*lag_14

Created rolling statistics:
*rolling_mean_7
*rolling_std_7
*rolling_mean_30

Applied one-hot encoding for:
*product_category
*day_of_week

5. Data Preprocessing
Removed rows generated with NaN after lag creation
Applied MinMaxScaler
Fit scaler only on training data to prevent data leakage

6. Train-Test Split
Maintained chronological order
No shuffling applied
Used last 20% of data as test set
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]
7. Sliding Window Dataset Creation

Created sequential datasets for LSTM training using a sliding window approach.

def create_sequences(data, seq_len):
8. Model Building
LSTM Model

Built a deep learning model using:
*PyTorch
*LSTM layer
*Fully connected output layer
*MLP Baseline
*Implemented a Multi-Layer Perceptron baseline model for comparison.

9. Model Training
Used:
*Adam optimizer
*MSELoss function
*Mini-batch training using DataLoader

Tracked:

*Training loss
*Best model checkpoint

Saved best model using:

torch.save(model.state_dict(), 'models/lstm_model.pt')
10. Model Evaluation

Evaluated model performance using:
*MAE (Mean Absolute Error)
*RMSE (Root Mean Squared Error)

Visualized:
*Actual vs Predicted demand
*Training loss curve

11. Model Saving
Saved:
Trained LSTM model (.pt)
Scaler (.pkl)
Cleaned dataset (.csv, .pkl)

12. Streamlit Deployment
Developed a Streamlit web application that:
*Loads trained model
*Takes product input from user
*Predicts future demand
*Displays demand forecast chart

Run application:
streamlit run app.py

Technologies Used
*Python
*Pandas
*NumPy
*Matplotlib
*Seaborn
*Statsmodels
*Scikit-learn
*PyTorch
*Streamlit

Key Concepts Applied
*Time Series Forecasting
*Deep Learning (LSTM)
*Feature Engineering
*Sliding Window Technique
*Data Preprocessing
*Outlier Detection
*Model Deployment

Future Improvements
*Hyperparameter tuning
*Attention-based forecasting models
*Multi-step forecasting
*TensorBoard integration
Cloud deployment using AWS EC2
