# Step-by-Step Implementation

```python
# Step 1: Import libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Step 2: Get the data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
!unzip book-crossings.zip
```

---

```python
# Step 3: Load the data into DataFrames
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# Import books metadata
df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'}
)

# Import ratings data
df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'}
)
```

---

```python
# Step 4: Data Cleaning
# Filter users with at least 200 ratings
user_counts = df_ratings['user'].value_counts()
active_users = user_counts[user_counts >= 200].index
df_ratings = df_ratings[df_ratings['user'].isin(active_users)]

# Filter books with at least 100 ratings
book_counts = df_ratings['isbn'].value_counts()
popular_books = book_counts[book_counts >= 100].index
df_ratings = df_ratings[df_ratings['isbn'].isin(popular_books)]

# Merge ratings with book titles for clarity
df_merged = pd.merge(df_ratings, df_books, on='isbn')
```

---

```python
# Step 5: Create a user-book matrix
user_book_matrix = df_merged.pivot_table(
    index='isbn',
    columns='user',
    values='rating'
).fillna(0)

# Convert to sparse matrix for efficiency
user_book_sparse = csr_matrix(user_book_matrix.values)
```

---

```python
# Step 6: Train the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_sparse)
```

---

```python
# Step 7: Recommendation function
def get_recommends(book=""):
    # Check if the book exists in the dataset
    if book not in df_merged['title'].values:
        return f"Book '{book}' not found in dataset."

    # Get the ISBN of the input book
    book_isbn = df_books[df_books['title'] == book]['isbn'].iloc[0]
    
    # Find the book's index in the user-book matrix
    book_index = user_book_matrix.index.get_loc(book_isbn)
    
    # Find similar books
    distances, indices = model_knn.kneighbors(
        user_book_sparse[book_index], n_neighbors=6
    )
    
    # Prepare results
    recommended_books = [
        (
            df_books[df_books['isbn'] == user_book_matrix.index[indices.flatten()[i]]]['title'].values[0],
            1 - distances.flatten()[i]  # Convert distance to similarity
        )
        for i in range(1, len(distances.flatten()))
    ]
    
    return [book, recommended_books]
```

---

```python
# Step 8: Test the function
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)
```

---

```python
# Step 9: Validate the function with a test
def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False
    
    recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
    
    for i in range(len(recommended_books)): 
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            test_pass = False
    
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")

# Run the test
test_book_recommendation()
```

---

### Explanation
- **Data Cleaning**: Filters sparse users and books to ensure meaningful recommendations.
- **KNN Model**: Trained on the sparse user-book matrix with cosine similarity as the distance metric.
- **Recommendation Function**: Returns the input book and 5 similar books based on user preferences.

---

This code is complete and ready to run in **Google Colab**. You can copy and paste each block sequentially into a Colab notebook to execute the project end-to-end. 😊
