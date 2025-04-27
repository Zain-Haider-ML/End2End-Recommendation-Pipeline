
# End to End Movie Recommendation System

This project implements a movie recommendation system using hybrid methods combining **Collaborative Filtering** (CF) and **Content-Based Filtering** (CBF). The system is built using the **Surprise** library for CF and **TF-IDF** for CBF. The code also provides multiple recommendation strategies such as **Weighted Hybrid**, **Switching Hybrid**, **Cascade Hybrid**, and **Meta-Level Hybrid**.

The system has been built with a modular architecture including:

- **Model** (`model.py`): Core recommendation algorithms and training functions.
- **Streamlit App** (`app.py`): User interface for recommendations.
- **Flask API** (`api.py`): Exposes the recommendation functionality via an API.
- **Docker**: Containerization for easy deployment and scalability.
- **CI/CD using GitHub Actions**: Continuous Integration/Continuous Deployment pipeline.
- **Azure Deployment**: Hosting the application in a cloud environment.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Running the App](#running-the-app)
- [Deployment](#deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [File Structure](#file-structure)
- [Algorithms](#algorithms)
- [License](#license)

---

## Overview

This project implements a **Movie Recommendation System** that integrates both collaborative filtering and content-based filtering:

- **Collaborative Filtering**: Uses user-item interactions (ratings) to make predictions. The implementation uses Singular Value Decomposition (SVD) from the **Surprise** library.
  
- **Content-Based Filtering**: Uses movie genre features and calculates similarity between movies using **TF-IDF** and **Cosine Similarity**.

### Hybrid Recommendation Methods
The following hybrid approaches are implemented:
1. **Weighted Hybrid**: Combines CF and CBF predictions with a weighted average.
2. **Switching Hybrid**: Switches between CF and CBF based on the user’s interaction history.
3. **Mixed Hybrid**: Combines both CF and CBF results together.
4. **Cascade Hybrid**: First ranks movies with CF, and then re-ranks using CBF scores.
5. **Meta-Level Hybrid**: Combines CF and CBF as features for training a regression model to predict movie ratings.

---

## Requirements

To run the project, you will need the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `surprise`
- `streamlit`
- `flask`

Make sure to install these dependencies in your environment:

```bash
pip install -r requirements.txt
```

Alternatively, the Docker container comes pre-configured with all dependencies.

---

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Install Dependencies:**

   If not using Docker, install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download MovieLens Data:**

   The script will automatically download the MovieLens dataset if it is not present. You can also manually download the dataset from [MovieLens](https://grouplens.org/datasets/movielens/100k/) and place it in the `ml-100k` folder.

---

## Running the App

### Streamlit App

To run the Streamlit app (frontend), use the following command:

```bash
streamlit run app.py
```

### Flask API

To run the Flask API (backend), use the following command:

```bash
python api.py
```

The Flask API will expose the recommendation endpoints to handle requests from the Streamlit app or other clients.

---

## Deployment

### Docker

The project can be containerized using Docker. The `Dockerfile` will create an image to run the app in a container.

1. **Build Docker Image:**

   ```bash
   docker build -t movie-recommendation .
   ```

2. **Run Docker Container:**

   ```bash
   docker run -p 5000:5000 movie-recommendation
   ```

### Azure Deployment

To deploy the application to **Azure**:

1. Set up an Azure App Service or Azure Container Instance.
2. Push your Docker image to **Azure Container Registry** or a public registry.
3. Deploy your app to Azure using the Docker image.

---

## CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration and deployment:

1. **CI Pipeline**: Automatically runs tests, builds the Docker image, and pushes to a registry on code pushes.
2. **CD Pipeline**: Automatically deploys the application to **Azure** whenever new changes are pushed to the main branch.

The `.github/workflows` folder contains the necessary configuration files for the CI/CD pipeline.

---

## File Structure

```
movie-recommendation-system/
├── app.py            # Streamlit application for frontend
├── api.py            # Flask API for backend
├── model.py          # Core recommendation logic
├── Dockerfile        # Docker configuration for deployment
├── requirements.txt  # List of dependencies
├── .github/
│   └── workflows/    # GitHub Actions configuration for CI/CD
├── ml-100k/          # MovieLens dataset
└── README.md         # Project documentation
```

---

## Algorithms

### Collaborative Filtering
Using **SVD (Singular Value Decomposition)**, this approach builds a matrix factorization model from user-item interactions to predict ratings.

### Content-Based Filtering
Using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **Cosine Similarity**, this approach suggests movies based on the similarity of movie genres.

### Hybrid Models
Multiple hybrid models are implemented, combining CF and CBF results:
- **Weighted Hybrid**: CF and CBF predictions are weighted and combined.
- **Switching Hybrid**: Uses CBF for new users and CF for existing users.
- **Mixed Hybrid**: Combines top N results from both CF and CBF.
- **Cascade Hybrid**: Ranks movies first using CF, then re-ranks using CBF.
- **Meta-Level Hybrid**: Uses CF and CBF scores as features for a regression model.

---
