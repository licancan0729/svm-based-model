import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
st.title("A svm-based model to predict coronary heart disease")

gensini= st.sidebar.slider('gensini',0,300)
gen=gensini

obesity_list=['yes','no']
obesity_name = st.sidebar.selectbox(
    "obesity?",
    obesity_list
)
if obesity_name=='yes':
    obesity_num =1
else:
    obesity_num =0


GENDER_list=['male','female']
GENDER_name = st.sidebar.selectbox(
    "GENDER?",
    GENDER_list
)
if GENDER_name=='male':
    GENDER_num =1
else:
    GENDER_num =2

age_list=[1,2,3]
age_name = st.sidebar.selectbox(
    "AGE?",
    age_list
)
if age_name==1:
    age_num =1
elif age_name==2:
    age_num =2
else:
    age_num=3

smoke_list=['yes','no']
smoke_name = st.sidebar.selectbox(
    "smoke?",
    smoke_list
)
if smoke_name=='yes':
    smoke_num =2
else:
    smoke_num =1


FAMILY_list=['yes','no']
FAMILY_name = st.sidebar.selectbox(
    "FAMILY?",
    FAMILY_list
)
if FAMILY_name=='yes':
    FAMILY_num =2
else:
    FAMILY_num =1



hypertension_list=['yes','no']
hypertension_name = st.sidebar.selectbox(
    "hypertension?",
    hypertension_list
)
if hypertension_name=='yes':
    hypertension_num =1
else:
    hypertension_num =0

from joblib import dump, load
svm = load('svm.joblib')
# c=[[0, 0, 1, 1, 1, 1, 0],]
c=[[gen,obesity_num, GENDER_num, age_num,smoke_num, FAMILY_num,hypertension_num],]
a=svm.predict(c)
b=svm.predict_proba(c)
print(a)
#print(obesity_num)
st.write(f"根据你的选择，诊断结果为{a}{b}")