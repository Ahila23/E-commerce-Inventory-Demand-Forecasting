import pandas as pd
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

#Page configuration
st.set_page_config(page_title="Demand forecasting App",layout="wide")
st.title("Demand Forecasting using LSTM")

#Loading the data
df=pd.read_pickle("models/_Demand cleaned_data.pkl")
print("pickle file successfully uploaded")

#Load scaler
with open("models/scaler.pkl",'rb')as f:
    scaler=pickle.load(f)

#LSTM Model Class

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        _, (h, _) = self.lstm(x)

        return self.fc(h[-1])

#Features
features=[col for col in df.columns if col not in ['date','product_id']]
input_size=len(features)

#Loading the model
model = LSTMModel(input_size)
model.load_state_dict(
    torch.load('models/lstm_model.pt',map_location=torch.device('cpu'))                          
)

model.eval()

print("Load LSTM Model pt successfully")

#product selction
product_list=df['product_id'].unique()
selected_product=st.selectbox(
    "Select Product Category/product_id",
    product_list
)

#filter product area
product_data=df[df['product_id']==selected_product]
st.subheader("Product data")
st.dataframe(product_data.tail())

#preparing the input
data=product_data[features]
scaled_data=scaler.transform(data)
SEQ_LEN=10

#last sequences
last_sequence=scaled_data[-SEQ_LEN:]
X_input=np.array([last_sequence])
X_input=torch.tensor(X_input,dtype=torch.float32)


#prediction
if st.button("Predict Demand"): 
    
    with torch.no_grad():
        prediction=model(X_input)
    
    pred_value=prediction.numpy()[0][0]

    #inverse scaling
    dummy_array=np.zeros((1,len(features)))
    dummy_array[0,0]=pred_value

    predicted_demand=scaler.inverse_transform(dummy_array)[0][0]

    st.success(f"Predicted Demand: {predicted_demand:.2f}")
    #forecasting line chart
    history=product_data['units_sold'].tail(30).values
    forecast=np.append(history,predicted_demand)
    chart_df=pd.DataFrame({
        'Index':range(len(forecast)),
        'Demand':forecast
})
    st.subheader("Demand Forecast Chart")
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(chart_df['Index'],chart_df['Demand'])
    ax.set_xlabel("Time")
    ax.set_ylabel("Units_sold")
    ax.set_title("Demand Forecast")
    st.pyplot(fig)
