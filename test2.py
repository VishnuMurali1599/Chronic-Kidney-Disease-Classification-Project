import streamlit as st
import pickle
import numpy as np
import json

# import the model
with open('CKD_Model.pkl', 'rb') as model_file:
    pipe = pickle.load(model_file)
    #pipe = pickle.load(open('CKD_Model.pkl','rb'))
#df = pickle.load(open('df.pkl','rb'))

st.title("Chronic Kidney Disease Classification")



# weight
age = st.number_input('Age')
blood_pressure = st.number_input('Blood Pressure')
specific_gravity = st.number_input('Specific Gravity')
albumin = st.number_input('Albuim')
sugar = st.number_input('Sugar')
red_blood_cells = st.number_input('Red Blood Cells')
pus_cell = st.number_input('Pus Cell')
pus_cell_clumps = st.number_input('Pus Cell Clumps')
bacteria = st.number_input('Bacteria')
blood_glucose_random = st.number_input('Blood Glucose Random')
blood_urea = st.number_input('Blood Urea')
serum_creatinine = st.number_input('Serum Creatinine')
sodium = st.number_input('Sodium')
potassium = st.number_input('Potassium')
haemoglobin = st.number_input('Haemoglobin')
packed_cell_volume = st.number_input('Packed cell volume')
white_blood_cell_count = st.number_input('white_blood_cell_count')
red_blood_cell_count = st.number_input('red_blood_cell_count')
hypertension = st.number_input('hypertension')
diabetes_mellitus = st.number_input('diabetes_mellitus')
coronary_artery_disease = st.number_input('coronary_artery_disease')
appetite = st.number_input('appetite')
peda_edema = st.number_input('peda_edema')
aanemia = st.number_input('aanemia')

if st.button('Check  Wether Chronic Kidney Disease or Not'):
    
    query = np.array([age,blood_pressure,specific_gravity,albumin,sugar,red_blood_cells,pus_cell,pus_cell_clumps,bacteria,blood_glucose_random,blood_urea,serum_creatinine,sodium,potassium,haemoglobin,packed_cell_volume,white_blood_cell_count,red_blood_cell_count,hypertension,diabetes_mellitus,coronary_artery_disease,appetite,peda_edema,aanemia])

    query = query.reshape(1,24)
    st.title("The Predicition of Disease is " + str(int(np.exp(pipe.predict(query)[0]))))
            