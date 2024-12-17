# Sentimental Analysis
* A Machine Learning project that analyzes IMDB movie reviews to classify sentiment as positive or negative using various tree-based models and TF-IDF vectorization

## Overview

  * **Definition of the tasks / challenge**: The project aims to perform binary sentiment classification on IMDB movie reviews, categorizing them as either positive or negative based on the review text. This involves natural language processing (NLP) techniques and machine learning models to understand and classify the emotional tone of movie reviews.
    
  * **My approach**: The solution employs a comprehensive text preprocessing pipeline including HTML tag removal, contraction expansion, emoji removal, and lemmatization. The processed text is then vectorized using TF-IDF, and multiple tree-based models are compared for classification performance.
    
  * **Summary of the performance achieved**: Based on the Accuracy Score, out models achieved the following performance on test set:
      * XGBoost: Best overall performance with accuracy score of 0.847
      * Random Forest: Close second with accuracy score of 0.842
      * AdaBoost: Strong performance with accuracy score of 0.821
      * Decision Tree and Gradient Boosting:  Baseline model with reasonable performance with accuracy score of 0.714 and 0.796

## Summary of Workdone

### Data

* Data:
  * Type: Text data (movie reviews) with binary sentiment labels
    
  * Size: Near 50,000 movie reviews
    
  * Instances (Train, Test Split): 80% Training (~ 40,000 reviews), 20% testing (~ 10,000 reviews)

#### Preprocessing / Clean up

* HTML tag removal using BeautifulSoup
* Contraction expansion (e.g., "don't" → "do not")
* Emoji removal
* URL removal
* Punctuation removal and text lowercase conversion
* Stopword removal
* Lemmatization using WordNet
* TF-IDF vectorization (10,000 features)

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* **Input**: Text data (movie reviews)
  * Raw text containing HTML tags, contractions, emojis, URLs, and various text artifacts
  * Variable length reviews (ranging from few words to several paragraphs)
    
* **Output**: Binary sentiment classification
  * Postivive: 1
  * Negative: 0

* **Models**:
  * **Decision Tree Classifier**:
    * Chosen for its interpretability and ability to handle non-linear relationships
    * Simple baseline model for comparison

  * **Random Forest**:
    * Ensemble method to reduce overfitting of decision trees
    * Better generalization through bagging
    * Good at handling high-dimensional sparse data from TF-IDF

  * **AdaBoost**:
    * Focuses on misclassified samples through boosting
    * Helps in handling difficult-to-classify reviews
    * Reduces bias in the model

  * **XGBoost**:
    * Efficient handling of sparse matrices
    * Better regularization to prevent overfitting
      
* **Model Parameters**:
  * **TF-IDF Vectorizer**:
    * max_features: 10,000
    * Default n-gram range: (1,1)
    * Default tokenization pattern

  * **Common Parameters Across Models**:
    * random_state: 42 (for reproducibility)
    * n_estimators: 100 (for ensemble models)
    * Default learning rates and tree depths
      
### Training

* **Hardware and Software Environment**:
  * The code runs efficiently on standard CPU hardware
  * No GPU acceleration required

* **Training Decision**:
  * Used default hyperparameters initially
  * No extensive hyperparameter tuning performed
  * No early stopping implemented

### Performance Comparison

* **XGboost**: 0.847 accuracy score
* **Random Forest**: 0.842 accuracy score
* **Ada Boost**: 0.821 accuracy score
* **Gradient Boosting**: 0.796 accuracy score
* **Decision Tree**: 0.714 accuracy score

### Future Work

* Implement deep learning models (LSTM, BERT)
* Experiment with different text preprocessing techniques
* Add cross-validation for more robust model evaluation

## How to reproduce results

* **Dataset requirements**:
  * IMDB-Dataset.csv: Contains 50,000 movie reviews with sentiment labels

    * Column 'review': Text of the movie review
    * Column 'sentiment': Binary sentiment label (positive/negative)

* **Result Reproduction**:
  * **1. Data Loading**:
    * Load dataset using pandas
    * Initial data exploration through checking for missing values, view data distribution
      
  * **2. Data Preprocessing**:
    * Install NLTK packages
    * Text cleaning steps

  * **3. Feature Processing**:
    * TF-IDF Vectorization:
      * Set max_features to 10,000
      * Fit vectorizer on training data

    * Train-Test Split: 80/20

  * **4. Model Training and Evaluation**:
    * Train 5 models:
      * XGboost
      * Random Forest
      * Ada Boost
      * Gradient Boosting
      * Decision Tree
    
### Software Setup
* **Required package**:
  * Pandas, NumPy, scikit-learn, nltk, beautifulsoup4, contractions, xgboost, seaborn, matplotlib
  * ```sh
    !pip install contractions
    !pip install pandas
    !pip install numpy
    !pip install scikit-learn
    !pip install xgboost
    !pip install matplotlib
    !pip install seaborn
    ```
    
### Dataset
* Download the IMBD data:
  ```sh
  !gdown 1v36q7Efz0mprjAv4g6TkQM2YlDKdqOuy
  ```
* Loading dataset:
  ```sh
  import pandas as pd
  df = pd.read_csv('./IMDB-Dataset.csv')
  ```
  






