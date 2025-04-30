#get needed libraries
import streamlit as st 
import tensorflow as tf
import pandas as pd

#get model that was saved
model=tf.keras.models.load_model('health_model.keras')

#input from streamlit web app
st.write("Provide the following information to predict medical expenses:")
n_age=st.number_input("Enter your age:", 33)
n_sex=st.radio("Sex:",("male","female"))
n_bmi=st.number_input("Enter your Body Mass Index:", 22.7)
n_children=st.number_input("Enter the number of children:",0)
n_smoker=st.radio("Do you smoke?:",("yes","no"))
n_region=st.selectbox("Which region are you from:", ["southwest","southeast","northwest","northeast"])

# From the Label Encoder in the model created the columns sex, smoker and region were encoded now we make sure they are given the same values

sex={'female':0, 'male':1}
smoker={'no':0, 'yes':1}
region={'northeast':0, 'northwest':1,'southeast':2,'southwest':3}

def get_mapped_value(value, mapping_dict):
    return mapping_dict.get(value, None)

m_sex=get_mapped_value(n_sex, sex)
m_smoker=get_mapped_value(n_smoker, smoker)
m_region=get_mapped_value(n_region, region)

data={
    'age':[n_age],
    'sex':[m_sex],
    'bmi':[n_bmi],
    'children':[n_children],
    'smoker':[m_smoker],
    'region':[m_region]
}
df=pd.DataFrame(data)
expenses=model.predict(df)
st.write("The predicted expenses is:", expenses)
