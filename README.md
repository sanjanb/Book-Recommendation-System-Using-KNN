Below is a detailed explanation of the project, along with a Markdown file format. The Markdown file contains an explanation of the problem, step-by-step approach, and answers to **what**, **how**, and **why** for each part of the coding solution.

---

```markdown
# Book Recommendation System Using K-Nearest Neighbors (KNN)

## Problem Description
The goal of this project is to create a book recommendation algorithm using the **K-Nearest Neighbors (KNN)** algorithm. The algorithm should:
- Take a book title as input.
- Return a list of 5 similar books along with their distances from the input book.

We will use the **Book-Crossings dataset**, which contains:
- 1.1 million ratings of 270,000 books by 90,000 users.

### Key Objectives:
1. Preprocess the data by cleaning and filtering.
2. Use KNN to find books similar to a given book.
3. Ensure the function returns results in the required format.

---

## Step-by-Step Approach

### Step 1: Data Preparation
#### What:
Load and filter the dataset to remove users and books with insufficient ratings. 
#### How:
- Load the `BX-Books` and `BX-Book-Ratings` CSV files into Pandas DataFrames.
- Remove users with fewer than 200 ratings.
- Remove books with fewer than 100 ratings.

#### Why:
Filtering ensures statistical significance and reduces noise in the data, improving recommendation accuracy.

---

### Step 2: Merge the Datasets
#### What:
Combine the books and ratings datasets into one for analysis.
#### How:
- Use Pandas `merge` to join the two datasets on the `isbn` column.
#### Why:
This step combines book metadata (titles and authors) with user ratings, making it easier to associate book titles with their ratings.

---

### Step 3: Create a User-Book Matrix
#### What:
Create a matrix where rows represent books, columns represent users, and values represent ratings.
#### How:
- Use `pivot_table` to transform the data into a user-book matrix.
- Fill missing values with 0 since no rating indicates no interaction.
- Convert the matrix into a sparse matrix for memory efficiency.

#### Why:
The KNN algorithm requires input in the form of a numerical matrix to compute distances between books.

---

### Step 4: Train the KNN Model
#### What:
Fit a KNN model using the user-book matrix.
#### How:
- Use the `NearestNeighbors` class from `sklearn.neighbors`.
- Specify the distance metric as **cosine similarity** and the algorithm as **brute force**.
- Fit the model to the sparse matrix.

#### Why:
The KNN algorithm identifies similar items by calculating distances between them. Cosine similarity is well-suited for recommendation systems as it measures similarity based on orientation rather than magnitude.

---

### Step 5: Define the Recommendation Function
#### What:
Implement a function to find the top 5 similar books for a given book title.
#### How:
1. Check if the book exists in the dataset.
2. Use the trained KNN model to find the nearest neighbors for the input book.
3. Extract the book titles and distances from the results.
4. Return the recommendations in the required format.

#### Why:
The recommendation function serves as the interface between the user and the KNN model, providing personalized recommendations.

---

### Step 6: Test the Function
#### What:
Test the `get_recommends` function to ensure it returns accurate and properly formatted results.
#### How:
- Define a test function with predefined inputs and expected outputs.
- Compare the actual output of `get_recommends` to the expected output.
#### Why:
Testing ensures that the implementation meets the project requirements.

---

## Code Implementation

```python
# Step 1: Import Libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Step 2: Load and Filter Data
df_books = pd.read_csv(
    'BX-Books.csv',
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'}
)

df_ratings = pd.read_csv(
    'BX-Book-Ratings.csv',
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'}
)

# Filter users with less than 200 ratings
user_counts = df_ratings['user'].value_counts()
df_ratings = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 200].index)]

# Filter books with less than 100 ratings
book_counts = df_ratings['isbn'].value_counts()
df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

# Step 3: Merge Datasets
df = pd.merge(df_ratings, df_books, on='isbn')

# Step 4: Create User-Book Matrix
pivot_table = df.pivot_table(index='title', columns='user', values='rating').fillna(0)
sparse_matrix = csr_matrix(pivot_table)

# Step 5: Train the KNN Model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

# Step 6: Define Recommendation Function
def get_recommends(book=""):
    if book not in pivot_table.index:
        return [book, []]
    distances, indices = model.kneighbors(
        pivot_table.loc[book].values.reshape(1, -1), n_neighbors=6
    )
    recommended_books = [
        [pivot_table.index[indices.flatten()[i]], distances.flatten()[i]]
        for i in range(1, len(distances.flatten()))
    ]
    return [book, recommended_books]

# Step 7: Test the Function
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
        print("You passed the challenge! 🎉")
    else:
        print("You haven't passed yet. Keep trying!")

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

test_book_recommendation()
```

---

## What, How, and Why for Each Part

### Loading and Filtering Data
- **What:** Load book and rating data.
- **How:** Use Pandas `read_csv` and filter rows based on rating counts.
- **Why:** Reduce noise and ensure meaningful recommendations.

### Merging Datasets
- **What:** Combine book metadata and ratings.
- **How:** Use `pd.merge` on the `isbn` column.
- **Why:** Ensure titles and ratings are in the same dataset.

### User-Book Matrix
- **What:** Create a numerical representation of user ratings for books.
- **How:** Use `pivot_table` and convert it to a sparse matrix.
- **Why:** Prepare data for KNN, which requires numerical input.

### Training KNN
- **What:** Train a KNN model.
- **How:** Use `NearestNeighbors` with cosine similarity.
- **Why:** Identify similar books based on user preferences.

### Recommendation Function
- **What:** Recommend similar books.
- **How:** Use the trained model to find nearest neighbors.
- **Why:** Provide personalized recommendations.

### Testing
- **What:** Validate the implementation.
- **How:** Compare actual output with expected results.
- **Why:** Ensure the solution meets project requirements.

---

## Conclusion
This project demonstrates how to use **K-Nearest Neighbors (KNN)** to build a recommendation system. The step-by-step approach ensures a clear understanding of data preprocessing, model training, and evaluation. The resulting function provides accurate and meaningful book recommendations.
```
