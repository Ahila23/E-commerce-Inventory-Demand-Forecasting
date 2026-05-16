# NeuralStock: Deep Learning for E-commerce Inventory Demand Forecasting

## Project Overview

NeuralStock is a deep learning-based inventory demand forecasting system developed using LSTM (Long Short-Term Memory) networks. The project predicts future product demand to help e-commerce businesses optimize inventory management, reduce stockouts, and improve procurement planning.

The application includes:
- Demand prediction using LSTM
- Interactive Streamlit dashboard
- Forecast visualization
- Reorder alert system
- Downloadable forecast reports
- Cloud deployment using AWS EC2

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- Streamlit
- AWS EC2

# Project Workflow

## Step 1 — Data Collection

Collected historical e-commerce inventory dataset containing:
- Product ID
- Units Sold
- Unit Price
- Stock on Hand
- Reorder Point
- Promotion Information
- Supplier Lead Days
- Date Information

## Step 2 — Data Preprocessing

Performed data cleaning and preprocessing:
- Converted date column into datetime format
- Removed duplicate records
- Handled missing values using:
  - Forward Fill
  - Backward Fill
- Sorted dataset by date and product_id

### Code Example

```python
df['date'] = pd.to_datetime(df['date'])

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

## Step 3 — Exploratory Data Analysis (EDA)

Analyzed the dataset using:
- Demand distribution plots
- Trend analysis
- Seasonal analysis
- Correlation heatmaps
- Time-series visualization

This helped identify:
- sales trends
- seasonal demand patterns
- stock behavior

### Code Example

```python
plt.figure(figsize=(10,5))
plt.plot(df['units_sold'])
plt.title("Units Sold Trend")
plt.show()

## Step 4 — Feature Engineering

Created additional features for better forecasting:
- lag_1 feature
- lag_7 feature
- rolling mean
- month
- quarter
- weekday
- weekend indicator

### Code Example

```python
df['lag_1'] = df['units_sold'].shift(1)
df['lag_7'] = df['units_sold'].shift(7)

df['rolling_mean'] = df['units_sold'].rolling(window=7).mean()

df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['day'] = df['date'].dt.day

df['is_weekend'] = df['date'].dt.weekday >= 5

## Step 5 — Data Scaling

Used MinMaxScaler to normalize numerical features.

Why scaling?
- LSTM models perform better on normalized data
- prevents large values from dominating training

### Code Example

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

pickle.dump(scaler, open("scaler.pkl", "wb"))
```

---

## Step 6 — Sequence Creation

Created sliding window sequences for time-series forecasting.

Example:
- Previous 10 days data → predict next demand value

### Code Example

```python
def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])

    return np.array(X), np.array(y)

SEQ_LENGTH = 10


## Step 7 — LSTM Model Building

Built LSTM model using PyTorch.

Model Architecture:
- LSTM Layer
- Fully Connected Dense Layer

### Code Example

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])

        return out


Loss Function:

```python
criterion = nn.MSELoss()


Optimizer:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


## Step 8 — Model Training

Trained model using:
- batch processing
- multiple epochs
- gradient optimization

Training process:
1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Weight updates

### Code Example

```python
epochs = 10

for epoch in range(epochs):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(loss.item())
```

---

## Step 9 — Model Evaluation

Evaluated model performance using:
- MAE
- RMSE
- Actual vs Predicted comparison

Visualized:
- training loss curve
- demand forecast chart

### Code Example

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print(mae)


## Step 10 — Save Trained Model

Saved trained LSTM model.

### Code Example

```python
torch.save(model.state_dict(), "lstm_model.pt")

# Streamlit Dashboard

Built an interactive dashboard using Streamlit.

Features:
- Product/category selection
- Date range filtering
- Demand prediction
- Forecast visualization
- Reorder alerts
- Downloadable CSV reports

### Run Application

```bash
streamlit run app.py

# Streamlit Dashboard Features

## Category Selection

```python
category = st.selectbox(
    "Select Product Category/product_id",
    product_list
)

## Date Range Input

```python
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

## Reorder Alert
```python
threshold = 50

if predicted_demand < threshold:
    st.warning("Reorder Alert: Stock may run low!")

## Download CSV Output

```python
csv = chart_df.to_csv(index=False)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="forecast.csv",
    mime="text/csv"
)

# AWS EC2 Deployment

Deployed application on AWS EC2 Ubuntu server.

## Deployment Steps

1. Created EC2 instance
2. Configured security groups
3. Opened port 5000
4. Installed Python dependencies
5. Uploaded project files
6. Ran Streamlit application

## Install Dependencies

```bash
sudo apt update

sudo apt install python3-pip -y

## Install Libraries

```bash
pip install -r requirements.txt

## Run Streamlit Application

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0

## Public URL
http://52.15.168.35:5000/



# requirements.txt
streamlit
torch
pandas
numpy
matplotlib
scikit-learn
pickle-mixin


# Project Structure
NeuralStock/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── lstm_model.pt
│   ├── scaler.pkl
│   └── Demand cleaned_data.pkl

# Future Improvements
- Transformer-based forecasting
- Attention mechanisms
- Multi-step forecasting
- Docker deployment
- Kubernetes scaling
- Real-time inventory integration

# Business Impact
This project helps:
- reduce stockouts
- optimize inventory levels
- improve procurement planning
- increase operational efficiency
- enhance customer satisfaction

# Conclusion
NeuralStock successfully demonstrates an end-to-end deep learning pipeline for e-commerce inventory demand forecasting using LSTM networks, interactive dashboards, and cloud deployment.
