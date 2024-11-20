# Book Recommendation System Documentation

## Overview

In this project, we will build a **Book Recommendation System** using the **Nearest Neighbors algorithm**. This will allow us to recommend books similar to a given book based on user ratings. The project uses collaborative filtering to make recommendations by analyzing the patterns in the ratings data provided by users.

This document will walk you through the entire process, from the basics of setting up the data to building a recommendation engine and testing it.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Requirements](#setup-and-requirements)
- [Loading the Data](#loading-the-data)
- [Filtering the Data](#filtering-the-data)
- [Creating the Book-User Matrix](#creating-the-book-user-matrix)
- [Building the Nearest Neighbors Model](#building-the-nearest-neighbors-model)
- [Recommendation Function](#recommendation-function)
- [Testing the Model](#testing-the-model)
- [Production and Deployment](#production-and-deployment)
- [Conclusion](#conclusion)

---

## Introduction

The goal of this project is to build a book recommendation system that suggests books based on a book title. The system uses collaborative filtering, a method that makes predictions about a user's interests by collecting preferences from many users. Specifically, we will use the **Nearest Neighbors** algorithm to identify similar books.

### What is Collaborative Filtering?
Collaborative filtering is a method used by recommendation systems to predict the interests of a user by collecting preferences or taste information from many users. It operates under the assumption that if two users agree on one issue, they will agree on others as well. In our case, it means if two users like similar books, they might also enjoy other books that the other user has rated highly.

---

## Setup and Requirements

Before we start, we need to set up our development environment and install the necessary libraries.

### Requirements:
1. **Python 3.x**
2. **Libraries:**
   - `pandas` – For data manipulation and analysis
   - `numpy` – For numerical operations
   - `scikit-learn` – For machine learning models (Nearest Neighbors)
   - `scipy` – For sparse matrix operations
   - `matplotlib` – For plotting (if needed)

To install these libraries, use the following command:

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

### Data:
We are working with two datasets:
1. **BX-Books.csv** – Contains information about books such as ISBN, title, and author.
2. **BX-Book-Ratings.csv** – Contains user ratings for books, where each user has rated multiple books.

---

## Loading the Data

### What is Data Loading?
Data loading is the first step in any data analysis or machine learning project. We need to load the raw data into a format that can be easily manipulated, analyzed, and used for training a machine learning model.

In this step, we load the CSV files containing the book and rating data into **pandas DataFrames**.

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
```

### Why Load Data?
By loading the data into a DataFrame, we can easily inspect, filter, and manipulate the data for further analysis and modeling. DataFrames allow you to work with tabular data in a way that is efficient and intuitive.

---

## Filtering the Data

### What is Data Filtering?
In any recommendation system, not all users or items (books) are equally relevant. Data filtering allows us to narrow down the dataset by removing users who have rated fewer books or books that have fewer ratings. This helps improve the accuracy of our model and reduces computational costs.

In this step, we filter:
- **Users** who have rated at least 200 books.
- **Books** that have been rated by at least 100 users.

```python
# Filter the books and users based on rating counts
filtered_users = df_ratings['user'].value_counts()
filtered_users = filtered_users[filtered_users >= 200].index

filtered_books = df_ratings['isbn'].value_counts()
filtered_books = filtered_books[filtered_books >= 100].index

# Filter the ratings dataframe to only include the selected users and books
df_filtered_users = df_ratings[df_ratings['user'].isin(filtered_users)]
df_filtered_books = df_filtered_users[df_filtered_users['isbn'].isin(filtered_books)]
```

### Why Filter Data?
Filtering ensures that our model is built using a subset of data that is relevant and robust. By focusing on users who are more active and books that have received more ratings, we increase the likelihood that the recommendations made by the model are useful.

---

## Creating the Book-User Matrix

### What is the Book-User Matrix?
The **book-user matrix** is a table where each row represents a user, each column represents a book, and the values in the table are the ratings given by each user to each book. If a user hasn't rated a book, the cell is filled with a zero (or NaN).

```python
# Create the book-user matrix
book_user_matrix = df_filtered_books.pivot(index='user', columns='isbn', values='rating').fillna(0)
```

### Why Create a Book-User Matrix?
This matrix serves as the foundation for our recommendation engine. By using this matrix, we can compare users and books to find similar items and users. It helps to visualize user preferences and ratings in a structured way.

---

## Building the Nearest Neighbors Model

### What is the Nearest Neighbors Algorithm?
The **Nearest Neighbors** algorithm is a machine learning method used to find similar items in a dataset. In our case, we are using it to find similar books based on user ratings. We use the **cosine similarity** metric to calculate how similar the books are.

```python
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Convert the book-user matrix to a sparse matrix format
sparse_matrix = csr_matrix(book_user_matrix.values)

# Initialize and train the Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)
```

### Why Use Nearest Neighbors?
The Nearest Neighbors algorithm is a simple yet effective way to recommend items based on similarity. By using the cosine similarity metric, we can measure the similarity between books even if they have only a few ratings in common.

---

## Recommendation Function

### What is the Recommendation Function?
The recommendation function takes a book title as input, retrieves its ISBN, and finds similar books based on user ratings. It then returns the most similar books along with their similarity scores.

```python
def get_recommends(book_title):
    # Ensure the book exists in the dataset
    if book_title not in df_books['title'].values:
        return f"Error: Book '{book_title}' not found in the dataset."

    # Retrieve ISBN
    isbn_target_book = df_books[df_books['title'] == book_title]['isbn'].values[0]

    # Check if the ISBN exists in the filtered book-user matrix
    if isbn_target_book not in book_user_matrix.columns:
        return f"Error: Book '{book_title}' was removed during filtering."

    # Get the index of the book in the matrix
    book_index = book_user_matrix.columns.get_loc(isbn_target_book)

    # Compute nearest neighbors
    distances, indices = model.kneighbors(
        book_user_matrix.iloc[:, book_index].values.reshape(1, -1), n_neighbors=6
    )

    # Generate recommendations
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        isbn = book_user_matrix.columns[indices.flatten()[i]]
        recommended_books.append(
            (
                df_books[df_books['isbn'] == isbn]['title'].values[0],
                distances.flatten()[i],
            )
        )

    return recommended_books
```

### Why Use the Recommendation Function?
This function automates the process of generating book recommendations based on a book title. It takes care of searching for the book in the dataset, retrieving its information, and calculating similar books efficiently.

---

## Testing the Model

### What is Model Testing?
Model testing involves checking the predictions or recommendations made by the model. In this case, we test the model by requesting recommendations for a specific book title.

```python
# Test the nearest neighbors algorithm
distances, indices = model.kneighbors(book_user_matrix.iloc[0, :].values.reshape(1, -1), n_neighbors=6)
print(distances, indices)

# Get book recommendations for a specific book
recommendations = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(recommendations)
```

### Why Test the Model?


Testing ensures that the model works as expected and produces meaningful recommendations. It's crucial to evaluate whether the nearest neighbors are indeed similar to the input book.

---

## Production and Deployment

### What is Deployment?
Deployment refers to taking a machine learning model that works in a development environment and making it available for use by others, whether through a web interface, an API, or in another production environment.

1. **Creating an API**:
   - Use **Flask** or **FastAPI** to expose the recommendation function through an API.
   - Deploy the application on cloud platforms like **Heroku** or **AWS**.

2. **Scaling**:
   - Ensure that the recommendation system can handle multiple users and large datasets by using optimized data storage and scaling techniques.

### Why Deploy the Model?
Deploying the model allows other users to interact with it, get book recommendations, and use the system in a real-world scenario.

---

## Conclusion

In this project, we have successfully built a Book Recommendation System using collaborative filtering and the Nearest Neighbors algorithm. We've walked through the entire process of loading, filtering, and transforming data into a usable format, training the model, testing its effectiveness, and discussing deployment options. By following these steps, you can build similar recommendation systems for a variety of domains.
