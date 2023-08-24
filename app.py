import streamlit as st
import numpy as np


import recommendation as recom

st.write("# Book Recommendation system")

user_id = st.text_input("Enter the user ID")
st.write("Type Selection : 0 - Memory Based, 1 - Model Based ")
    
Type = st.text_input("Enter the type")

if  len(Type) != 0:
    print("user id is ",user_id) 
    recom_df = recom.get_recommendation(user_id,Type)
    if "Similarity_score(0-1)" in recom_df.columns:
        st.dataframe(recom_df[["Recommended_title","Similarity_score(0-1)","avg_rating"]])
    else:
        st.dataframe(recom_df[["Recommended_title","num_of_ratings","avg_rating"]])
        
    for i in range(0,len(recom_df)-3,3):
        for col in st.columns(3):
            with col:
                st.image(recom_df.iloc[i]["Image_URL_M"],width = 100)
                st.write("Rating" ,recom_df.iloc[i]["avg_rating"])
                st.write(recom_df.iloc[i]["Recommended_title"])
            i +=  1
    st.write("End of list")
