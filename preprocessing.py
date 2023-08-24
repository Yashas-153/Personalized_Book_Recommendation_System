import pandas as pd
import numpy as np
import warnings
import nltk
import re
import pickle 
print("download from internet   ")
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
    
warnings.filterwarnings("ignore")


books = pd.read_csv("Datasets/Books.csv")
ratings = pd.read_csv("Datasets/Ratings.csv")
ratings = ratings[ratings["Book-Rating"]!= 0]
users = pd.read_csv("Datasets/Users.csv")
books = books.set_axis(["ISBN","Title","Author","Year","Publisher","Image_URL_S","Image_URL_M","Image_URL_L"], axis = "columns")
ratings_with_name = books.merge(ratings,on = "ISBN").drop(["Author","Year","Publisher","Image_URL_M","Image_URL_L"],axis = 1)

BOOKS_TITLE_LIST = books['Title'].to_list()


def preprocessing(text):
    text = re.sub(r"[^a-zA-Z ]","",str(text))
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    stemmed_words = [stemmer.stem(word) for word in words]  
    text = ' '.join(stemmed_words) 
    return text



def get_books_data():
    books_df = books[["ISBN","Title","Author","Publisher","Image_URL_S"]]
    books_df = books_df.merge(ratings,on = "ISBN")
    top_rated_books = books_df.groupby("Title").count()["Book-Rating"]  > 8
    books_df = books_df[books_df["Title"].isin(top_rated_books[top_rated_books].index)].drop_duplicates(subset = ["Title"], keep = "first")
    books_df["Features"] = books_df["Title"] + " " + books_df["Author"] + " " + books_df["Publisher"]

    print("preprocessing started")

    books_df["Features"] = books_df["Features"].apply(preprocessing)

    print("preprocessing ended")

    books_df = books_df.reset_index()
    books_data = books_df[["Title","Features"]]
    
    return books_data

   
def get_memory_based_pt():
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 65
    top_users_rating = ratings_with_name[ratings_with_name["User-ID"].isin(x[x].index)]

    y = top_users_rating.groupby('Title').count()['Book-Rating'] >= 15
    final_ratings = top_users_rating[top_users_rating["Title"].isin(y[y].index)]

    pivot_table = final_ratings.pivot_table(index='Title',columns='User-ID',values='Book-Rating')
    pivot_table.fillna(0,inplace=True)

    return pivot_table.T

def get_model_based_pt():   
    users_pt = get_memory_based_pt()
    nmf_model = pickle.load(open("Models/nmf_model.pkl","rb"))
    W = nmf_model.transform(users_pt)
    print("shape of W is ",W.shape)
    H = nmf_model.components_
    print("H shape is ",H.shape)
    mat = np.dot(W,H)
    model_pt = pd.DataFrame(mat, columns = np.array(users_pt.columns))
    model_pt.set_index(np.array(users_pt.index),inplace = True)
    print("returned model_pt")
    return model_pt


pivot_table = get_memory_based_pt()
