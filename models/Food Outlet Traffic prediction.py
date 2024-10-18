pip install streamlit scikit-learn pandas


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set the title for the app
st.title('Traffic Prediction for Food Outlet')

# Sidebar inputs for model features
st.sidebar.header('User Input Features')

def user_input_features():
    day_of_week = st.sidebar.selectbox('Day of the Week', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
    time_of_day = st.sidebar.slider('Time of Day (24-hour format)', 0, 23, 12)
    promotion = st.sidebar.selectbox('Promotional Offer', ('No', 'Yes'))
    
    # Convert inputs to a DataFrame
    data = {
        'Day_of_Week': day_of_week,
        'Time_of_Day': time_of_day,
        'Promotion': promotion
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Input features from sidebar
input_df = user_input_features()

# Simulated historical data
data = {
    'Day_of_Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'Time_of_Day': [8, 12, 18, 10, 14, 19, 21],
    'Promotion': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Traffic': [200, 300, 250, 400, 320, 450, 500]
}
df = pd.DataFrame(data)

# Encoding categorical variables
df['Day_of_Week'] = df['Day_of_Week'].astype('category').cat.codes
df['Promotion'] = df['Promotion'].apply(lambda x: 1 if x == 'Yes' else 0)
input_df['Day_of_Week'] = input_df['Day_of_Week'].astype('category').cat.codes
input_df['Promotion'] = input_df['Promotion'].apply(lambda x: 1 if x == 'Yes' else 0)

# Features and target
X = df[['Day_of_Week', 'Time_of_Day', 'Promotion']]
y = df['Traffic']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(input_df)

# Output
st.subheader('Predicted Traffic')
st.write(f'Estimated traffic: {int(prediction[0])} people')

# Display the actual dataset used (for illustration)
st.subheader('Training Data')
st.write(df)


streamlit run app.py

