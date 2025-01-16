# Importing Libraries

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading the model

pipe = joblib.load('Pipeline.joblib')

# Importing the Data
df = pd.read_csv('true_car_listings.csv')

# Create the Page


# creating two cols
col1, col2 = st.columns([1, 3], gap='small', border=False, vertical_alignment='top')

# Insert an Image Under col1
with col1:
    st.image("C:/Users/Madhavz/Desktop/My Details/img2.jpg", width=100)

# Insert a Title Under Col2

with col2:
    st.title(':car: :green[Car Price Predictor]', )

# Creating an Expander

with st.expander(':violet[***ABOUT THE PROJECT***]'):

    # Writing About the Project
    st.write('**:car: :green[Car Price Predictor]** is a **Supervised Machine Learning Model** which predicts the **Price of Cars** based on three features namely **CAR MAKER, MODEL AND MILEAGE**.')
    st.write('**:blue[Linear Regression]** algorithm is working behind this model.')

# Show the Model
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_count():
    st.session_state.count += 1

# create a header for the model
st.success(f'**MODEL** ( {st.session_state.count} Liked )', icon=':material/model_training:')

# create a container for input and output
with st.container():

    col1, col2 = st.columns([2.5,1], border=True,vertical_alignment='top')

    # use col1 for input
    with col1:
        st.info('**INPUT**', icon=':material/input:')

        inp1, inp2= st.columns(2, border=True)
        inp3, inp4 = st.columns(2, border=True)

        with inp1:
            maker = st.selectbox(':blue[CAR MAKER]', options=df['Make'].unique())

        with inp2:
            if maker:
                model_name = df[df['Make'] == maker]['Model'].unique()
            else:
                model_name = None
                
            model_name = st.selectbox(':blue[CAR MODEL]', options=model_name)
            model_name = str(model_name)

        with inp3:
            mini = df['Mileage'].min()
            maxi = df['Mileage'].max()
            mileage = st.text_input(':blue[CAR MILEAGE]', placeholder=f'{mini} - {maxi}')
            if mileage:
                mileage = np.int64(mileage)

        with inp4:
            st.write('**CLICK ON :red[PREDICT PRICE]**')
            st.button('PREDICT PRICE', type='primary', use_container_width=True)
            if maker and model_name and mileage:
                input_data = pd.DataFrame(data=[[maker, model_name, mileage]], columns=['Make', 'Model', 'Mileage'])
                prediction = pipe.predict(input_data)

    # Use col2 for output
    with col2:
        st.success('**OUTPUT**', icon=':material/output:')

        with st.container():
            try:
                st.image(f'Car Images/{str.lower(maker)}.jpg')
            except FileNotFoundError:
                st.warning("Image not available for this car maker.")

        with st.container():
            if mileage:
                price = np.round(prediction[0], 2)
                st.info(f'##### **:material/currency_rupee: {price}**')
                

# container for like, share, video, repository, connection, hf space
with st.container(border=False):
    col1, col2, col3, col4, col5, col6 = st.columns([.85,1,.95,.95,1.15,1.15], border=False, vertical_alignment='top')

    with col1:
        st.button('**LIKE**', icon=':material/favorite:', on_click=increment_count())

    with col2:
        st.button('**SHARE**', icon=':material/share:')

    with col3:
        st.link_button('**VIDEO**', icon=':material/slideshow:', url='https://www.youtube.com/@madee2020')

    with col4:
        st.link_button('**REPO**', icon=':material/code_blocks:', url='https://github.com/madhavpani/Car-Price-Prediction')

    with col5:
        st.link_button('**CONNECT**', icon=':material/connect_without_contact:', url='https://www.linkedin.com/in/madhavpani')

    with col6:
        st.link_button('**HF SPACE**', icon=':material/sentiment_satisfied:', url='https://huggingface.co/spaces')

