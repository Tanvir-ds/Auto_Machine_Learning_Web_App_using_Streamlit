# importing libraries
import streamlit as st
import pandas as pd
import os
# importing profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
# importing Machine Learning stuff
from pycaret.classification import setup,compare_models,pull,save_model
import pickle as pkl

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto Streamlit Machine Learning Web App")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This Appication allows you to build an automated ML Pipeline using Stramlit,Pandas_Profiling and PyCaret.")
    
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv('sourcedata.csv',index_col=None)

if choice=='Upload':
    st.title("Upload your data for Modeling!")
    file = st.file_uploader("Upload your dataset here...")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)

if choice=='Profiling':
    st.title("Automated Exploratory Data Analysis!")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice=='ML':
    st.title("Machine Learning Model go!")
    target = st.selectbox("Select Your Target Column...",df.columns)
    if st.button("Train the Model..."):
        setup(df,target=target,silent=True)
        setup_df = pull()
        st.info("This is the Machine Learning Experiment settings!")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the Machine Learning Model!")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")

if choice=='Download':
    with open("best_model.pkl","rb") as f:
        st.download_button("Download the Model...",f,"trained_model.pkl")
    