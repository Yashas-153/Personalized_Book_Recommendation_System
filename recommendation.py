import numpy as np
import pandas as pd

import preprocessing as prep
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


print("reading all csv files")
books = pd.read_csv("Datasets/Books.csv")
ratings = pd.read_csv("Datasets/Ratings.csv")
users = pd.read_csv("Datasets/Users.csv")
ratings = ratings[ratings["Book-Rating"]!=0]
print("read all csv files")

books = books.set_axis(["ISBN","Title","Author","Year","Publisher","Image_URL_S","Image_URL_M","Image_URL_L"], axis = "columns")
books_data = prep.get_books_data()

tvectorizer = TfidfVectorizer(max_features= 20000)
    
tfidf_mat = tvectorizer.fit_transform(books_data["Features"]) 
users_pt = prep.get_memory_based_pt()
books_title_list = books_data['Title'].to_list() 
    
ratings_with_name = books.merge(ratings,on = "ISBN").drop(["Author","Year","Publisher","Image_URL_S","Image_URL_L"],axis = 1)
print("done with all data frames")


def get_average_ratings_df():
    print("called get_avr_rating_df")
    high_rated = ratings.groupby("ISBN").count().sort_values("Book-Rating",ascending = False).reset_index().drop(["User-ID"],axis = 1)
    high_rated.rename(columns = {'Book-Rating': 'num_of_ratings'},inplace = True)
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"],errors = "coerce")
    avg_ratings = ratings.groupby("ISBN").mean().sort_values("Book-Rating",ascending = True).reset_index().drop(["User-ID"],axis = 1)
    avg_ratings.rename(columns = {'Book-Rating': 'avg_rating'},inplace = True)
    popular_books = books.merge(high_rated,on = "ISBN").merge(avg_ratings, on = "ISBN").drop(["Image_URL_S","Image_URL_L"],axis = 1)
    return popular_books

popular_books = get_average_ratings_df()

def recommend_top_rated_books():
    print("Called recommend_top_books")
    """
    Function that returns the top rated books from the collections

    Input: 
        None

    Returns:
        A dataframe that containt top books
    
    """
    top_100_books = popular_books.sort_values(by= ["num_of_ratings","avg_rating"],ascending = False)[:101].reset_index()
    top_100_books["avg_rating"] = top_100_books["avg_rating"].round(1)
    top_100_books.rename(columns = {'Title':'Recommended_title'},inplace = True)

    return top_100_books

def recommend_from_past_pref_by_name(user_id):
    """
    Function that recommends books based on users previous interests and ratings

    Input :
        user_id : int
    
    Returns :
        A dataframe of recommended books and related information.
    
    """
    
    print("called recommend_from_past_pref_by_name from recommendation.py")
    
    prev_read_books = ratings_with_name[ratings_with_name["User-ID"] == user_id].groupby("Title").sum().sort_values("Book-Rating",
        ascending = False).reset_index()["Title"]
    top = min(len(prev_read_books),15)
    prev_read_books = prev_read_books[:top]
    top_recom_books = pd.DataFrame(columns = ['Recommended_title','Similarity_score(0-1)','avg_rating','Image_URL_M'])
    cosine_sim = cosine_similarity(tfidf_mat)
    print("num of read books is ",len(prev_read_books))
    for book_read in prev_read_books:
        index = -1
        try:
            index = books_title_list.index(book_read)         
        except ValueError:
            continue
        print("index is" ,index)
        sim_score = list(enumerate(cosine_sim[index])) 
        
        sim_score = sorted(sim_score, key= lambda x: x[1], reverse=True)[1:6]  
        recommend_index = [i[0] for i in sim_score] 
        rec_books = books_data['Title'].iloc[recommend_index]
        rec_scores = [round(i[1],4) for i in sim_score]
        
        avg_ratings = []
        cover_images = []
        for rec_score, rec_book in zip(rec_scores, rec_books):
            print("books is ",rec_book)
            cover_images.append(books[books["Title"] == rec_book]["Image_URL_M"].values[0])
            
            avg_ratings.append(round(popular_books[popular_books["Title"] == rec_book]["avg_rating"].sort_values(ascending= False).values[0],1))
            
        df = pd.DataFrame(list(zip(rec_books,rec_scores, avg_ratings,cover_images)), columns=['Recommended_title','Similarity_score(0-1)','avg_rating','Image_URL_M'])
        top_recom_books = pd.concat([top_recom_books,df])
    
    print("returned from")
    
    return top_recom_books.sort_values("avg_rating",ascending=False)


def recommend_by_sim_users_mem_based(user_id):
    print("Called recommend_by_sim_users_mem_based")

    """
    Function that recommends books based on the similar users interests through memory based approach (Collaborative filtering).

    Input:
        user_id : int
    
    Returns 
        A dataframe of recommended books and related information.
     
    """
    
    index = np.where(users_pt.index == user_id)[0][0] 
    users_sim_scores = cosine_similarity(users_pt)
    user_sim_scores = users_sim_scores[index]
    similar_users = np.argsort(user_sim_scores)[::-1][1:6]
    print(similar_users)
    read_books = users_pt.columns[users_pt.loc[user_id].values != 0]
    
    top_recom_books = pd.DataFrame(columns = ['Recommended_title','Similarity_score(0-1)','avg_rating','Image_URL_M'])

        
    for sim_user_index , similar_score  in list(zip(similar_users,user_sim_scores[similar_users])):
        sim_user_read_books = users_pt.columns[users_pt.iloc[sim_user_index].values != 0]

        unread_books = [book for book in sim_user_read_books if book not in read_books]
        unread_books = unread_books[:min(len(unread_books),20)]
        cover_images = []
        weighted_scores = []
        avg_ratings = []
        similarity_score = []
        print("number of unread_books ",len(unread_books))
        for book in unread_books:
            rating = ratings_with_name[(ratings_with_name["User-ID"]== users_pt.index[sim_user_index]) & (ratings_with_name['Title'] == book)]["Book-Rating"].values
            if len(rating) > 0 and rating[0] > 0:
                cover_images.append(books[books["Title"] == book]["Image_URL_M"].values[0])
                similarity_score.append(similar_score)    
                avg_ratings.append(round(popular_books[popular_books["Title"] == book]["avg_rating"].sort_values(ascending= False).values[0],2))
                weighted_score = similar_score * float(rating[0])
                weighted_scores.append(weighted_score)
            else:
                weighted_scores.append(0)
        df = pd.DataFrame(list(zip(unread_books,similarity_score, avg_ratings,cover_images)), columns=['Recommended_title','Similarity_score(0-1)','avg_rating','Image_URL_M'])
        top_recom_books = pd.concat([top_recom_books,df])
        print("books in df is",len(df))
    return top_recom_books



def recommend_by_sim_users_model_based(user_id):

    """
    Function that recommends books based on the similar users interests through model based approach (Collaborative filtering).

    Input:
        user_id : int
    
    Returns:  
        A dataframe of recommended books and related information.
    
    """
    print("called recommend_by_sim_user in model_based")
    model_pt = prep.get_model_based_pt()
    
    index = np.where(model_pt.index == user_id)[0][0] 
    users_sim_score = cosine_similarity(model_pt)
    user_sim_scores = users_sim_score[index]

    similar_users = np.argsort(user_sim_scores)[::-1][1:6]
    print(similar_users)
    read_books = model_pt.columns[model_pt.loc[user_id].values >= 1]
    
    top_recom_books = pd.DataFrame(columns = ['Recommended_title','Similarity_score(0-1)','avg_rating','Image_URL_M'])

        
    for sim_user_index , similar_score  in list(zip(similar_users,user_sim_scores[similar_users])):
        sim_user_read_books = model_pt.columns[model_pt.iloc[sim_user_index].values >= 1]

        unread_books = [book for book in sim_user_read_books if book not in read_books]
        unread_books = unread_books[:min(len(unread_books),25)]
        cover_images = []
        weighted_scores = []
        avg_ratings = []
        similarity_score = []
        print("number of unread_books ",len(unread_books))
        
        for book in unread_books:
            avg_ratings.append(round(popular_books[popular_books["Title"] == book]["avg_rating"].sort_values(ascending= False).values[0],1))
            cover_images.append(books[books["Title"] == book]["Image_URL_M"].values[0])
            similarity_score.append(similar_score)    
            weighted_scores.append(0)
        df = pd.DataFrame(list(zip(unread_books,similarity_score,avg_ratings,cover_images)), columns=['Recommended_title','Similarity_score(0-1)','avg_rating','Image_URL_M'])
        top_recom_books = pd.concat([top_recom_books,df])

    print("returned dataframe")
    return top_recom_books


def get_recommendation(user_id,type = 0):

    """
    A function that implements hybrid recommendation system. This is the main function this module.

    Input:
        user_id : int
        type : int 
            {
                0 : memory based approach
                1 : model based approach
            }

    
    Returns:
        A dataframe of recommended books.

    """
    try:
        print("called get_recommendation")
        user_id = int(user_id)
        index = np.where(users_pt.index == user_id)[0][0]
        recom_by_sim_title_past_df =  recommend_from_past_pref_by_name(user_id)
        if int(type) == 0:
            print('called recommend_by_sim_users_mem_based')
            recom_by_sim_users_df = recommend_by_sim_users_mem_based(user_id)   
        else :
            recom_by_sim_users_df = recommend_by_sim_users_model_based(user_id)
        final_df = pd.DataFrame(columns = ['Recommended_title','Similarity_score(0-1)','avg_rating',"Image_URL_M"])
        final_df = pd.concat([recom_by_sim_users_df,recom_by_sim_title_past_df]).drop_duplicates(
            subset =["Recommended_title"],keep = "first").reset_index().sort_values("Similarity_score(0-1)",ascending= False)
        print("returned final_df")
        return final_df
    
    except IndexError:
        print("called top rated")
        return recommend_top_rated_books()



