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

---

## Improvements

1. Use advanced matrix factorization methods like SVD for better scalability.
2. Include genre or author metadata for hybrid recommendations.
