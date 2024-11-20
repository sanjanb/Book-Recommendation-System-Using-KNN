# Book Recommendation System using Collaborative Filtering

## Overview

This project implements a book recommendation system using collaborative filtering and nearest neighbors. We utilize a dataset containing book ratings and information from the Book-Crossing dataset. The system generates book recommendations based on user ratings, leveraging similarity metrics to find books that other users with similar tastes might enjoy.

---

## Steps and Explanation

### 1. **Downloading and Extracting the Dataset**

```python
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
!unzip -o book-crossings.zip
```

**What:** The first step is downloading the dataset from the internet and extracting the ZIP file that contains the books data. The `-o` flag in the `unzip` command overwrites any existing files without asking for confirmation.

**Why:** The dataset contains CSV files that we need to load into memory to start building the recommendation system.

### 2. **Loading the Data**

```python
df_books = pd.read_csv(books_filename, encoding="ISO-8859-1", sep=";", header=0, names=['isbn', 'title', 'author'], usecols=['isbn', 'title', 'author'], dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})
df_ratings = pd.read_csv(ratings_filename, encoding="ISO-8859-1", sep=";", header=0, names=['user', 'isbn', 'rating'], usecols=['user', 'isbn', 'rating'], dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
```

**What:** We load two CSV files:
- `BX-Books.csv` contains information about books such as ISBN, title, and author.
- `BX-Book-Ratings.csv` contains user ratings for books, where each rating represents how much a user liked a book.

**Why:** These datasets provide the raw information necessary for building the recommendation system, where we will analyze user ratings to find similar users and recommend books accordingly.

---

### 3. **Data Preprocessing**

```python
filtered_users = df_ratings['user'].value_counts()
filtered_users = filtered_users[filtered_users >= 200].index
filtered_books = df_ratings['isbn'].value_counts()
filtered_books = filtered_books[filtered_books >= 100].index
```

**What:** Filtering users and books to reduce the dataset size:
- We retain only users who have rated at least 200 books.
- We retain only books that have been rated at least 100 times.

**Why:** This preprocessing step helps to focus the model on active users and popular books, removing noise from less frequent users and books. This makes the recommendation system more efficient and relevant.

---

### 4. **Creating the Book-User Matrix**

```python
book_user_matrix = df_filtered_books.pivot(index='user', columns='isbn', values='rating').fillna(0)
```

**What:** We create a matrix where each row represents a user, each column represents a book, and the values represent ratings for those books. Missing ratings are filled with 0.

**Why:** This matrix is essential for collaborative filtering since it represents user preferences, which we use to find similarities between users and books.

---

### 5. **Training the Nearest Neighbors Model**

```python
sparse_matrix = csr_matrix(book_user_matrix.values)
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)
```

**What:** We convert the `book_user_matrix` into a sparse matrix to save memory, then train a nearest neighbors model using the cosine similarity metric.

**Why:** Nearest neighbors models are commonly used in recommendation systems because they identify items (or users) that are similar to one another based on their features—in this case, user ratings.

---

### 6. **Generating Recommendations**

```python
def get_recommends(book_title):
    isbn_target_book = df_books.loc[df_books['title'] == book_title, 'isbn'].iloc[0]

    # Get the index of the book in the matrix
    try:
        book_index = book_user_matrix.columns.get_loc(isbn_target_book)
    except KeyError:
        print(f"Book '{book_title}' not found in the book-user matrix.")
        return []

    distances, indices = model.kneighbors(
        book_user_matrix.T.iloc[:, book_index].values.reshape(1, -1), n_neighbors=6
    )

    recommended_books = []
    for i in range(1, len(distances.flatten())):
        isbn = book_user_matrix.index[indices.flatten()[i]]
        recommended_books.append(
            (
                df_books[df_books['isbn'] == isbn]['title'].values[0],
                distances.flatten()[i],
            )
        )

    return recommended_books
```

**What:** The `get_recommends` function takes a book title as input, finds the book in the `df_books` dataset, and retrieves its ISBN. It then finds the nearest neighbors (most similar books) based on user ratings using the `NearestNeighbors` model.

**Why:** The recommendation system works by finding books with similar ratings to the input book, leveraging the nearest neighbors algorithm based on cosine similarity.

---

### 7. **Testing and Using the Model**

```python
recommendations = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(recommendations)
```

**What:** This is the test step where we provide a book title, get recommendations, and print them.

**Why:** We use this step to verify that the recommendation system works correctly and produces relevant book suggestions.

---

## Summary

In this project, we built a recommendation system for books using collaborative filtering. We filtered users and books, created a matrix of user ratings, and used a nearest neighbors algorithm to find similar books based on their ratings. The system generates a list of recommended books for any given book, allowing users to discover similar titles based on others' preferences.

## Running the Code

To run this code:
1. Install the required libraries: `numpy`, `pandas`, `scipy`, `sklearn`, `matplotlib`.
2. Download and extract the dataset using the `wget` and `unzip` commands.
3. Run the provided code to filter users and books, create the book-user matrix, and train the model.
4. Use the `get_recommends` function to get book recommendations for any book title.

--- 

## Conclusion

This recommendation system can be extended and improved by incorporating more advanced techniques, such as matrix factorization, content-based filtering, or hybrid models. You can also scale this approach to handle larger datasets or integrate it into a production system by creating an API for real-time recommendations.
