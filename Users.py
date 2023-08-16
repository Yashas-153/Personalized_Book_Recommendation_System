import uuid
import pandas as pd
import Books
class User:
    def __init__(self, age):
        self.user_id = int(uuid.uuid4())  # Generate a unique user ID
        self.age = age
        self.books_rated = {}  # Dictionary to store books and their ratings
        self.activities = {} # Dictionary that stores their acitivity in the app

    def see_and_rate_book(self, book_title,ISBN, rating):
        Books.
        
        
        
# Example usageCopy
user1 = User(age=25)
user2 = User(age=30)

user1.see_and_rate_book("Book1", rating=4)
user1.see_and_rate_book("Book2", rating=5)

user2.see_and_rate_book("Book3", rating=3)
user2.see_and_rate_book("Book1", rating=2)

print(user1.books_rated)
print(user2.books_rated)
