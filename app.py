# import all the pevious library plus streamlit
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from tensorflow.keras.models import load_model
import streamlit as st

#load the trained model 
model=load_model('model.h5')

#load the all the pickel file into veriable
with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)
with open('one_hot_encoder.pkl','rb') as file:
    geo_encoder=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler_transformer=pickle.load(file)

#how to load the title 
st.title("Model Churn Prediction")


#create input options so that user can select 
#for catagorical veriables mostly we use
geography=st.selectbox('Geography',geo_encoder.categories_[0])#we use category beacuse it is onehotencoder here [0] it is taking columns
gender=st.selectbox('Gender',label_encoder.classes_)#we use classes becuse we have used label encoder
HasCrCard=st.selectbox('HasCrCard',[0,1])
IsActiveMember=st.selectbox('IsActiveMember',[1,0])


#we use slider when we have limited no of options we have to mention the range beside them
Age=st.slider('Age',0,100)
Tenure=st.slider('Tenure',0,10)
NumOfProducts=st.slider('NumOfProducts',0,4)


#we use number imput when we do not have exact range 
Balance=st.number_input("Balance")
CreditScore=st.number_input("CreditScore")
EstimatedSalary=st.number_input("EstimatedSalary")


#now take the input data and convert them into data frame
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [label_encoder.transform([gender])[0]],
    'Geography': [geography],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

#we cannot do onehotencoding so we do it here individually
geoencoder=geo_encoder.transform(input_data[['Geography']])
geoencoder=pd.DataFrame(geoencoder.toarray(),columns=geo_encoder.get_feature_names_out(['Geography']))
input_data=input_data.drop(['Geography'],axis=1)
input_data=pd.concat([geoencoder,input_data],axis=1)
input_data


x=model.predict(input_data)
st.write(x)
if x>0.5 :
    st.write("customer is likely to churn")
else:
    st.write("customer is not likely to churn")








