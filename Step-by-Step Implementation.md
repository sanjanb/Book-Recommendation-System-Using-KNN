# Book Recommendation System using K-Nearest Neighbors

## Overview

This project implements a **Book Recommendation System** using the K-Nearest Neighbors (KNN) algorithm. The system takes a book title as input and provides a list of similar books, based on user ratings from the Book-Crossings dataset.

---

## Features

- **Dataset**: Utilizes the Book-Crossings dataset containing over 1.1 million ratings.
- **Filtering**: Removes sparse data for statistical significance by filtering out users with less than 200 ratings and books with less than 100 ratings.
- **Model**: Implements the KNN algorithm to recommend 5 books based on similarity.
- **Distance Metric**: Calculates closeness of books using cosine similarity.

---

## Dataset

The dataset contains:
1. **Books**: Metadata including ISBN, title, and author.
2. **Ratings**: User ratings of books, scaled from 1 to 10.

The datasets used:
- `BX-Books.csv`
- `BX-Book-Ratings.csv`

---

## Project Files

- **`books_knn.py`**: Python code implementing the book recommendation system.
- **`code-breakdown.md`**: Detailed explanation of each step in the code.
- **`book_recommender.md`**: Project documentation (this file).

---

## Code Explanation

### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
```
- **NumPy**: Handles numerical operations efficiently.
- **Pandas**: Used to manipulate tabular data.
- **SciPy**: Provides `csr_matrix` for sparse matrix representation.
- **scikit-learn**: Implements the K-Nearest Neighbors algorithm.
- **Matplotlib**: Visualizes data (optional).

---

### Step 2: Download and Extract Dataset
```python
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
!unzip book-crossings.zip
```
- Downloads the Book-Crossings dataset.
- Extracts two CSV files:
  - `BX-Books.csv`
  - `BX-Book-Ratings.csv`

---

### Step 3: Load Datasets
```python
import pandas as pd

# Load BX-Books.csv
books_filename = 'BX-Books.csv'
df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'}
)

# Load BX-Book-Ratings.csv
ratings_filename = 'BX-Book-Ratings.csv'
df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'}
)

print(df_books.head())  # Preview books DataFrame
print(df_ratings.head())  # Preview ratings DataFrame
```
- **Books Data (`df_books`)**: Includes `isbn`, `title`, and `author`.
- **Ratings Data (`df_ratings`)**: Includes `user`, `isbn`, and `rating`.

---

### Step 4: Data Preprocessing
1. **Filter Data**:
   - Retain users with >= 200 ratings.
   - Retain books with >= 100 ratings.
2. **Merge Data**:
   - Combine `df_books` and `df_ratings` using the `isbn` column.
3. **Create Pivot Table**:
   - Rows: Users
   - Columns: Books
   - Values: Ratings

```python
# Filter users with at least 200 ratings
filtered_users = df_ratings['user'].value_counts()
filtered_users = filtered_users[filtered_users >= 200].index

# Filter books with at least 100 ratings
filtered_books = df_ratings['isbn'].value_counts()
filtered_books = filtered_books[filtered_books >= 100].index

# Apply filters
df_filtered_users = df_ratings[df_ratings['user'].isin(filtered_users)]
df_filtered_books = df_filtered_users[df_filtered_users['isbn'].isin(filtered_books)]

```

---

### Step 5: Model Training
```python
book_user_sparse = csr_matrix(book_user_matrix)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model.fit(book_user_sparse)
```
- Converts the pivot table into a sparse matrix for efficiency.
- Trains a **KNN model** with:
  - **Cosine Similarity** as the distance metric.
  - **Brute Force Search** to find neighbors.

---

### Step 6: Book Recommendation Function
```python
def get_recommends(book=""):
    book_index = df_books[df_books['title'] == book].index[0]
    distances, indices = model.kneighbors(book_user_sparse[book_index], n_neighbors=6)
    
    recommended_books = []
    for i in range(1, len(indices[0])):
        recommended_books.append([df_books.iloc[indices[0][i]].title, distances[0][i]])
    
    return [book, recommended_books]
```
- Finds neighbors for the input book using the trained KNN model.
- Retrieves titles and distances of the 5 nearest books.
- Returns the input book title and the list of recommendations.

---

### Step 7: Testing
```python
def test_book_recommendation():
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    print(recommends)
```
- Tests the recommendation function using a sample book.

---

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Run the Code in Google Colab**:
   - Open `books_knn.py` in Google Colab.
   - Execute all cells to generate recommendations.

---

## Output Example

```python
get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
```
**Output**:
```plaintext
[
    "Where the Heart Is (Oprah's Book Club (Paperback))",
    [
        ["I'll Be Seeing You", 0.8],
        ['The Weight of Water', 0.77],
        ['The Surgeon', 0.77],
        ['I Know This Much Is True', 0.77]
    ]
]
```
