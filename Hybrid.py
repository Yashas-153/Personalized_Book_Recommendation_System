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

#ratings.groupby(by = "User-ID").sum().sort_values(by = "Book-Rating", ascending = False).head(10)

similarity_top = books[["ISBN","Title","Author","Publisher","Image_URL_S"]]
similarity_top = similarity_top.merge(ratings,on = "ISBN")
top_rated_books = similarity_top.groupby("Title").count()["Book-Rating"] > 20
similarity_top = similarity_top[similarity_top["Title"].isin(top_rated_books[top_rated_books].index)]
similarity_top = similarity_top.drop_duplicates(subset = ["Title"], keep = "first")

similarity_top["Features"] = similarity_top["Title"] + " " + similarity_top["Author"] + " " + similarity_top["Publisher"]

print("preprocessing started")
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


#similar content books

similarity_top["Features"] = similarity_top["Features"].apply(preprocessing)
print("preprocessing ended")
similarity_top = similarity_top.reset_index()
books_data = similarity_top[["Title","Features"]]


tvectorizer = TfidfVectorizer(max_features= 20000)
tfidf_mat = tvectorizer.fit_transform(books_data["Features"])
tfidf_mat.shape
print("vectorised")

cosine_sim = cosine_similarity(tfidf_mat) #applying cosine similarity to tfidf_matrix
programme_list=books_data['Title'].to_list() #creating list of movies and tv shows

def recommend_sim_content(title, cosine_similarity= cosine_sim):
    print("Called recommend_sim_content")
    index = programme_list.index(title)         #finds the index of the input title in the programme_list.
    sim_score = list(enumerate(cosine_sim[index])) #creates a list of tuples containing the similarity score and index of the input title and all other programmes in the dataset.
    
    #position 0 is the movie itself, thus exclude
    sim_score = sorted(sim_score, key= lambda x: x[1], reverse=True)[1:10]  #sorts the list of tuples by similarity score in descending order.
    recommend_index = [i[0] for i in sim_score]  #selecting index of recommended movies
    rec_pro = books_data['Title'].iloc[recommend_index]
    rec_score = [round(i[1],4) for i in sim_score]
    rec_table = pd.DataFrame(list(zip(rec_pro,rec_score)), columns=['Recommended Product','Similarity(0-1)'])
    return rec_table


#Item based similarity collaborative filtering

ratings_with_name = books.merge(ratings,on = "ISBN").drop(["Author","Year","Publisher","Image_URL_M","Image_URL_L"],axis = 1)
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
top_users_rating = ratings_with_name[ratings_with_name["User-ID"].isin(x[x].index)]

y = top_users_rating.groupby('Title').count()['Book-Rating'] >= 50
final_ratings = top_users_rating[top_users_rating["Title"].isin(y[y].index)]

title_pt = final_ratings.pivot_table(index='Title',columns='User-ID',values='Book-Rating')
title_pt.fillna(0,inplace=True)

from sklearn.metrics.pairwise import cosine_similarity
title_sim_score = cosine_similarity(title_pt)
title_sim_score.shape

users_pt = title_pt.T
users_sim_score = cosine_similarity(title_pt.T)


def recommend_sim_books(book_name):
    print("Called recommend_sim_books")
    # index fetch
    index = np.where(title_pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(title_sim_score[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Title'] == title_pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Title')['Title'].values))
        item.extend(list(temp_df.drop_duplicates('Title')['Author'].values))
        item.extend(list(temp_df.drop_duplicates('Title')['Image_URL_M'].values))
        
        data.append(item)
    
    return data

#User Based similarity collaborative filtering

def recommend_simi_users(user_id, users_pt, users_sim_score):
    print("Called recommend_simi_users")
    try:
        index = np.where(users_pt.index == user_id)[0][0]  
        user_sim_scores = users_sim_score[index]
        print(index)
        # Sort similar users based on similarity score and get the top similar users
        similar_users = np.argsort(user_sim_scores)[::-1][1:5]
        # Get a list of books that the given user has already read
        read_books = users_pt.columns[users_pt.iloc[index].values != 0]
        print(similar_users)
        
        recommended_books = []
        for sim_user_index in similar_users:
            # Get a list of books that the similar user has read but the given user hasn't
            sim_user_read_books = users_pt.columns[users_pt.iloc[sim_user_index].values != 0]

            unread_books = [book for book in sim_user_read_books if book not in read_books]
            #recommended_books.extend(unread_books)
            unread_books_ratings = ratings_with_name[(ratings_with_name["User-ID"]== users_pt.index[sim_user_index]) & (ratings_with_name['Title'].isin(unread_books))]
            
            # Sort the unread books by their average rating in decreasing order
            unread_books_sorted = unread_books_ratings.groupby('Title')['Book-Rating'].mean().sort_values(ascending=False).index.tolist()[:5]
            
            recommended_books.extend(unread_books_sorted)

            
        recommended_books = list(set(recommended_books))
        return recommended_books
  
    except:
        print("The user have not rated anything explicitly")
        

print("done all")
user_id = 23902
recommended_books = np.array(recommend_simi_users(user_id, users_pt, users_sim_score))
print(recommended_books)