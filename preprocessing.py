import pandas as pd
import numpy as np
import warnings
import nltk
import re
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    
warnings.filterwarnings("ignore")


books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")

books = books.set_axis(["ISBN","Title","Author","Year","Publisher","Image_URL_S","Image_URL_M","Image_URL_L"], axis = "columns")

def preprocessing(text):
    #removing all numbers
    text = re.sub(r"[^a-zA-Z ]","",str(text))
    #converting everything to lower case
    text = text.lower()
    #remove punctuations
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    #remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    #stemming
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    stemmed_words = [stemmer.stem(word) for word in words]  # applying the Snowball stemmer to each word
    text = ' '.join(stemmed_words) 
    return text


#ratings.groupby(by = "User-ID").sum().sort_values(by = "Book-Rating", ascending = False).head(10)

def get_books_data():
    books_df = books[["ISBN","Title","Author","Publisher","Image_URL_S"]]
    books_df = books_df.merge(ratings,on = "ISBN")
    top_rated_books = books_df.groupby("Title").count()["Book-Rating"] > 20
    books_df = books_df[books_df["Title"].isin(top_rated_books[top_rated_books].index)].drop_duplicates(subset = ["Title"], keep = "first")
    books_df["Features"] = books_df["Title"] + " " + books_df["Author"] + " " + books_df["Publisher"]

    print("preprocessing started")

    books_df["Features"] = books_df["Features"].apply(preprocessing)

    print("preprocessing ended")

    books_df = books_df.reset_index()
    books_data = books_df[["Title","Features"]]
    return books_data

def get_cosine_sim_of_content():

    tvectorizer = TfidfVectorizer(max_features= 20000)
    books_data = get_books_data()
    tfidf_mat = tvectorizer.fit_transform(books_data["Features"])

    print("vectorised")

    cosine_sim = cosine_similarity(tfidf_mat) #applying cosine similarity to tfidf_matrix
    
    return cosine_sim


ratings_with_name = books.merge(ratings,on = "ISBN").drop(["Author","Year","Publisher","Image_URL_M","Image_URL_L"],axis = 1)
    
def get_user_item_pivot_table():
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    top_users_rating = ratings_with_name[ratings_with_name["User-ID"].isin(x[x].index)]

    y = top_users_rating.groupby('Title').count()['Book-Rating'] >= 50
    final_ratings = top_users_rating[top_users_rating["Title"].isin(y[y].index)]

    pivot_table = final_ratings.pivot_table(index='Title',columns='User-ID',values='Book-Rating')
    pivot_table.fillna(0,inplace=True)

    return pivot_table.T

def get_cosine_sim_of_users():
    
    users_pt = get_user_item_pivot_table()
    users_sim_score = cosine_similarity(users_pt)
    
    return users_sim_score


