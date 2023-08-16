import uuid

class Book:
    def __init__(self,title,author,year,publisher):
        self.ISBN =  str(uuid.uuid4())
        self.title = title
        self.author = author
        self.year = year
        self.publisher = publisher
    
    def rate_book(self,ISBN,rating):
        