# Book Recommendation System 📚✨

This project builds a **K-Nearest Neighbors (KNN)** based recommendation algorithm for books, leveraging the **Book-Crossings dataset**. The system suggests five books similar to the input book title, using statistical measures of closeness.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Algorithm Description](#algorithm-description)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [File Breakdown](#file-breakdown)
  - [code-breakdown.md](#code-breakdownmd)
  - [book_recommendation.md](#book_recommendationmd)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project uses the **K-Nearest Neighbors** algorithm to recommend books. By analyzing user ratings from the **Book-Crossings dataset**, the algorithm identifies books that share similar user preferences.

---

## Dataset

The **Book-Crossings dataset** includes:
- **1.1 million ratings** on a scale of 1-10.
- **270,000 books** rated by **90,000 users**.

### Cleaning Criteria:
- **Users**: Only include those with **200+ ratings**.
- **Books**: Only include books with **100+ ratings**.

---

## Algorithm Description

### K-Nearest Neighbors (KNN)
The algorithm measures the "distance" between books to identify similar items:
1. **Input**: A matrix of user-book interactions.
2. **Distance Measure**: Euclidean distance or cosine similarity.
3. **Output**: A list of similar books with their respective distances.

### Recommendation Function
The **`get_recommends`** function:
1. Takes a book title as input.
2. Finds 5 similar books using the trained KNN model.
3. Returns the input title and a list of 5 recommendations.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/book-recommendation-system.git
   cd book-recommendation-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```bash
   wget [insert-dataset-url-here]
   ```

4. Run the script:
   ```bash
   python book_recommender.py
   ```

---

## How It Works

1. **Data Preprocessing**:
   - Filter out sparse data.
   - Create a user-book interaction matrix.

2. **Model Training**:
   - Train the KNN model on the preprocessed matrix.

3. **Recommendation**:
   - Input a book title into the **`get_recommends`** function.
   - Get a ranked list of 5 similar books with their distances.

4. **Testing**:
   - Validate the function with test cases.

---

## File Breakdown

### [code-breakdown.md](./code-breakdown.md)
This file explains the project code:
- Code structure
- Detailed walkthrough of functions and logic

### [book_recommendation.md](./book_recommendation.md)
This file provides insights into:
- Dataset handling
- Nearest Neighbors algorithm
- Challenges and solutions

---

### Solution Process

#### Key Steps:
1. **Data Cleaning**:
   - Import the Book-Crossings dataset.
   - Filter users with fewer than 200 ratings and books with fewer than 100 ratings to ensure statistical significance.

2. **Data Preparation**:
   - Create a user-book interaction matrix (rows: users, columns: books).
   - Normalize or scale the data as required for Nearest Neighbors.

3. **Model Implementation**:
   - Use the **`NearestNeighbors`** algorithm from `sklearn.neighbors` to measure closeness.
   - Train the model on the user-book matrix.

4. **Recommendation Function**:
   - Create a function **`get_recommends`** that takes a book title as input.
   - Find 5 similar books using the trained NearestNeighbors model.
   - Return the input title and a list of 5 recommendations with their respective distances.

5. **Testing**:
   - Validate the function against test cases to ensure correctness.


## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

Happy recommending! 🚀
