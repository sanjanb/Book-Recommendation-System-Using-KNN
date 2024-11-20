# Book Recommendation System 📖✨

This project builds a **K-Nearest Neighbors (KNN)** based recommendation algorithm using the **Book-Crossings dataset**. It recommends similar books for a given book title, based on user preferences.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Algorithm Description](#algorithm-description)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [File Breakdown](#file-breakdown)
  - [code-breakdown.md](#code-breakdownmd)
  - [recommendation-system.md](#recommendation-systemmd)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project implements a **KNN-based Book Recommendation System**. By analyzing the Book-Crossings dataset, the system suggests books similar to an input book based on user ratings.

---

## Dataset

The **Book-Crossings dataset** includes:
- **1.1 million ratings** on a scale of 1-10.
- Metadata for **270,000 books**.
- Ratings from **90,000 users**.

### Cleaning Criteria:
- **Users**: Include only users with **200+ ratings**.
- **Books**: Include only books with **100+ ratings**.

---

## Algorithm Description

### K-Nearest Neighbors (KNN)
The KNN algorithm measures the similarity between books based on user ratings:
1. **Input**: A user-book rating matrix.
2. **Output**: A list of books similar to the given input book.

### Recommendation Function
The **`get_recommends`** function:
1. Accepts a book title.
2. Identifies 5 similar books using the trained KNN model.
3. Returns:
   - The input book title.
   - A list of 5 recommended books with their similarity scores.

---

## How to Run

### Prerequisites

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/book-recommendation-system.git
   cd book-recommendation-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script**:
   ```bash
   python book-recommender.py
   ```

---

## How It Works

### Steps:

1. **Data Cleaning**:
   - Remove sparse users and books.
   - Create a matrix of user-book ratings.

2. **Model Training**:
   - Train a **K-Nearest Neighbors** model using the user-book matrix.

3. **Recommendations**:
   - Input a book title into the **`get_recommends`** function.
   - Output a list of 5 similar books with similarity scores.

---

## File Breakdown

### [code-breakdown.md](./code-breakdown.md)
- A comprehensive explanation of the code, including:
  - Data cleaning and preparation.
  - Model training and recommendation logic.

### [recommendation-system.md](./recommendation-system.md)
- Insights into the recommendation system's implementation.
- Challenges faced and solutions adopted.

---

## Testing

The project includes a test function to validate the recommendation system:
```python
test_book_recommendation()
```
If all tests pass, the system outputs:
> "You passed the challenge! 🎉🎉🎉🎉🎉"

---

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

Happy recommending! 🚀
```

---

### Additional Files

#### 1. code-breakdown.md

```markdown
# Code Breakdown: Book Recommendation System

---

## Overview

This file provides a detailed explanation of the project code for the Book Recommendation System.

---

## Sections

### 1. Import Libraries

The following libraries are used:
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **SciPy (csr_matrix)**: Efficient matrix operations.
- **scikit-learn (NearestNeighbors)**: Implementation of the KNN algorithm.
- **Matplotlib**: Data visualization.

---

### 2. Data Loading

Two CSV files are loaded:
- **BX-Books.csv**: Book metadata (ISBN, Title, Author).
- **BX-Book-Ratings.csv**: User ratings for books.

---

### 3. Data Cleaning

Steps:
1. Filter users with fewer than 200 ratings.
2. Filter books with fewer than 100 ratings.
3. Create a user-book rating matrix.

---

### 4. Model Training

The **`NearestNeighbors`** model is trained on the sparse user-book rating matrix. The distance metric used is cosine similarity.

---

### 5. Recommendation Function

The **`get_recommends`** function:
1. Takes a book title as input.
2. Finds the 5 nearest neighbors using the trained KNN model.
3. Returns the input book title and 5 recommended books with similarity scores.

---

## Challenges

1. Sparsity in the dataset.
2. Efficient handling of large-scale data.

---

```

#### 2. recommendation-system.md

```markdown
# Recommendation System Insights

---

## Overview

This file explores the recommendation logic and algorithmic insights behind the Book Recommendation System.

---

## Algorithm: K-Nearest Neighbors (KNN)

### Why KNN?

KNN is chosen for its simplicity and effectiveness in computing similarities for sparse datasets.

### Distance Metric

Cosine similarity is used to measure the closeness of books based on user preferences.

---

## Challenges

1. **Data Sparsity**:
   - Many books and users have very few interactions.
   - Solution: Filter users and books with insufficient ratings.

2. **Scalability**:
   - Handling 1.1 million ratings efficiently.
   - Solution: Use sparse matrices for memory-efficient computations.

---

## Improvements

1. Use advanced matrix factorization methods like SVD for better scalability.
2. Include genre or author metadata for hybrid recommendations.
