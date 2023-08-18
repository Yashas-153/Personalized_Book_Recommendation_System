import streamlit as st

import recommendation as recom
st.write("# Book Recommendation system")

user_id = st.text_input("Enter the user ID")
st.write("Type Selection \n Type 0 : Memory Based \n Type 1 : Model Based \n")

Type = st.text_input("Enter the type")

if  len(Type) is not 0:
    recom_df = recom.get_recommendation(user_id,Type)
    st.dataframe(recom_df)


# Sample data (replace this with your DataFrame)
# df = recom_df.head(10)

# Define the number of books to display per row
books_per_row = 5

# Streamlit app
# st.title("Netflix-like Book Interface")

# # Loop through the books in rowsst
# for i in range(0, len(df), books_per_row):
#     for col , book  in list(zip(st.columns(3),df.iterrows())):

#         #books = df.iloc[i:i+books_per_row]

#         # Display each book's cover image and title
#         st.image(book['Image_URL_M'], caption=book['Title'], use_column_width='auto')

