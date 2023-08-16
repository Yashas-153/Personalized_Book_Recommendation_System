import numpy as np
import pandas as pd

import preprocessing as prep

print("reading all csv files")
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")

print("read all csv files")
books = books.set_axis(["ISBN","Title","Author","Year","Publisher","Image_URL_S","Image_URL_M","Image_URL_L"], axis = "columns")
books.head()

books_data = prep.get_books_data()

books_title_list = books_data['Title'].to_list() #creating list of movies and tv shows
    
ratings_with_name = books.merge(ratings,on = "ISBN").drop(["Author","Year","Publisher","Image_URL_M","Image_URL_L"],axis = 1)
print("done with all data frames")

def recommedn_top_rated_books():
    high_rated = ratings.groupby("ISBN").count().sort_values("Book-Rating",ascending = False).reset_index().drop(["User-ID"],axis = 1)
    high_rated.rename(columns = {'Book-Rating': 'num_of_ratings'},inplace = True)
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"],errors = "coerce")
    avg_ratings = ratings.groupby("ISBN").mean().sort_values("Book-Rating",ascending = True).reset_index().drop(["User-ID"],axis = 1)
    avg_ratings.rename(columns = {'Book-Rating': 'avg_rating'},inplace = True)
    popular_books = books.merge(high_rated,on = "ISBN").merge(avg_ratings, on = "ISBN").drop(["Image_URL_M","Image_URL_L"],axis = 1)
    popular_books["weighted_rating"] = (popular_books["num_of_ratings"] * popular_books["avg_rating"])/sum(popular_books["num_of_ratings"])
    top_100_books = popular_books.sort_values(by= "weighted_rating",ascending = False)[:101].reset_index()
    return top_100_books

def recommend_from_past_pref_by_name(user_id,cosine_sim):
    prev_read_books = ratings_with_name[ratings_with_name["User-ID"] == user_id].groupby("Title").sum().sort_values("Book-Rating",
        ascending = False).reset_index()["Title"]
    top = min(len(prev_read_books),15)
    prev_read_books = prev_read_books[:top]
    top_recom_books = pd.DataFrame(columns = ['Recommended_title', 'Similiarity_score(0-1)','Weighted_score(1-10)'])
    for book_read in prev_read_books:
        index = -1
        try:
            index = books_title_list.index(book_read)         #finds the index of the input title in the programme_list.
        except ValueError:
            continue
        sim_score = list(enumerate(cosine_sim[index])) #creates a list of tuples containing the similarity score and index of the input title and all other programmes in the dataset.
        
        #position 0 is the movie itself, thus exclude
        sim_score = sorted(sim_score, key= lambda x: x[1], reverse=True)[1:6]  #sorts the list of tuples by similarity score in descending order.
        recommend_index = [i[0] for i in sim_score]  #selecting index of recommended movies
        rec_books = books_data['Title'].iloc[recommend_index]
        rec_scores = [round(i[1],4) for i in sim_score]
        
        weighted_ratings = []
        for rec_score, rec_book in zip(rec_scores, rec_books):
            user_rating = ratings_with_name[(ratings_with_name["User-ID"] == user_id) & (ratings_with_name["Title"] == book_read)]["Book-Rating"].values
            if len(user_rating) > 0:
                weighted_ratings.append(rec_score * user_rating[0])
            else:
                weighted_ratings.append(0)
        
        df = pd.DataFrame(list(zip(rec_books,rec_scores, weighted_ratings)), columns=['Recommended_title','Similarity_score(0-1)','Weighted_score(1-10)'])
        top_recom_books = pd.concat([top_recom_books,df])
    
    return top_recom_books.sort_values("Weighted_score(1-10)",ascending=False)

#ans = recommend_from_past_pref_by_name(23902,ratings_with_name)   
#ans

def recommend_by_sim_users(user_id, users_pt, users_sim_score):

    index = np.where(users_pt.index == user_id)[0][0]  
    user_sim_scores = users_sim_score[index]
    #print(index)
    # Sort similar users based on similarity score and get the top similar users
    similar_users = np.argsort(user_sim_scores)[::-1][1:6]
    print(similar_users)
    # Get a list of books that the given user has already read
    read_books = users_pt.columns[users_pt.iloc[index].values != 0]
    #print("Similarity Scores:", user_sim_scores[similar_users])
    
    top_recom_books = pd.DataFrame(columns = ['Recommended_title','Similarity_score(0-1)','Weighted_score(1-10)'])

    recommended_books = []
        
    for sim_user_index , similar_score  in list(zip(similar_users,user_sim_scores[similar_users])):
        # Get a list of books that the similar user has read but the given user hasn't
        sim_user_read_books = users_pt.columns[users_pt.iloc[sim_user_index].values != 0]

        unread_books = [book for book in sim_user_read_books if book not in read_books]
        #print(unread_books)
        weighted_scores = []
        for book in unread_books:
            rating = ratings_with_name[(ratings_with_name["User-ID"]== users_pt.index[sim_user_index]) & (ratings_with_name['Title'] == book)]["Book-Rating"].values
            if len(rating) > 0:
         #       print(similar_score)
                weighted_score = similar_score * rating[0]
                weighted_scores.append(weighted_score)
            else:
                weighted_scores.append(0)
        
        df = pd.DataFrame(list(zip(unread_books,user_sim_scores[similar_users], weighted_scores)), columns=['Recommended_title','Similarity_score(0-1)','Weighted_score(1-10)'])

        top_recom_books = pd.concat([top_recom_books,df])
        
    return top_recom_books


print("started to call all the preprocessing steps")

users_pt = prep.get_user_item_pivot_table()
user_sim_score = prep.get_cosine_sim_of_users()
title_sim_score = prep.get_cosine_sim_of_content()

user_id = int(input("Enter the user_id"))

try :
    index = np.where(users_pt.index == user_id)[0][0]
    recom_by_sim_users_df = recommend_by_sim_users(user_id,users_pt,user_sim_score)
    recom_by_sim_title_past_df =  recommend_from_past_pref_by_name(user_id,title_sim_score)
    
    final_df = pd.DataFrame(columns = ['Recommended_title','Similarity_score(0-1)','Weighted_score(1-10)'])
    final_df = pd.concat([recom_by_sim_users_df,recom_by_sim_title_past_df])
    final_df = final_df.sort_values(by = "Weighted_score(1-10)",ascending = False)

    print(final_df)
except ValueError:
    top_recom_df = recommedn_top_rated_books()
    print(top_recom_df)


