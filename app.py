import streamlit as st

import recommendation as recom
st.write("# Book Recommendation system")

user_id = st.text_input("Enter the user ID")
st.write("Type Selection \n Type 0 : Memory Based \n Type 1 : Model Based \n")

Type = st.text_input("Enter the type")

if  len(Type) is not 0:

    recom_df = recom.get_recommendation(user_id,Type)
    st.dataframe(recom_df)
    for i in range(0,len(recom_df)-3,3):
        for col in st.columns(3):
            with col:
                st.image(recom_df.iloc[i]["Image_URL_M"],width = 100)
                st.write("Rating" ,recom_df.iloc[i]["avg_rating"])
                st.write(recom_df.iloc[i]["Recommended_title"])
            i +=  1
            # if i > len(recom_df):
            #     break
    st.write("End of list")