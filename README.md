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

* 1. HTML tag removal using BeautifulSoup
* 2. Contraction expansion (e.g., "don't" → "do not")
* 3. Emoji removal
* 4. URL removal
* 5. Punctuation removal and text lowercase conversion
* 6. Stopword removal
* 7. Lemmatization using WordNet
* 8. TF-IDF vectorization (10,000 features)

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
      
### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

## Citations

* Provide any references.







