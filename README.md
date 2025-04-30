# ecommerce-text-classification-using-naive-bayes
This project demonstrates text classification using Naive Bayes models, specifically Gaussian Naive Bayes, Bernoulli Naive Bayes, and Multinomial Naive Bayes. The project explores different feature extraction techniques, including CountVectorizer and TF-IDF (Term Frequency-Inverse Document Frequency), to evaluate model performance.

### Project Overview
The objective of this project is to classify text data into categories using three Naive Bayes models. The data is preprocessed using two different methods of feature extraction: CountVectorizer and TF-IDF. The accuracy of each model is evaluated and compared to understand how well each method performs on the given dataset.

### Models Used
Gaussian Naive Bayes (GaussianNB)

Assumes that the features follow a normal (Gaussian) distribution.

Bernoulli Naive Bayes (BernoulliNB)

Suitable for binary/boolean features, ideal for classification tasks where features are binary or represent the presence/absence of a term.

Multinomial Naive Bayes (MultinomialNB)

Best suited for discrete count data, such as the number of times a word appears in a document.

### Feature Extraction Techniques
CountVectorizer: Converts a collection of text documents into a matrix of token counts. It represents text data numerically by counting the frequency of each word across all documents.

TF-IDF: Reflects how important a word is to a document in a collection. It reduces the impact of commonly occurring words and highlights more meaningful terms based on their frequency across documents

### Project Workflow
Data Preprocessing:
The data is cleaned and preprocessed by handling missing values, text cleaning, and feature scaling.

Feature Extraction:
Both CountVectorizer and TF-IDF are used to convert the text data into a numeric format suitable for machine learning.

Model Training:
Three Naive Bayes models (Gaussian, Bernoulli, and Multinomial) are trained on the processed data using Cross-Validation to assess their accuracy.

Model Evaluation:
The models' performance is evaluated based on accuracy, and comparisons are made between the results of CountVectorizer and TF-IDF.

### Requirements
To run this project, you need to have Python 3.x installed along with the following libraries:

scikit-learn

pandas

numpy

matplotlib

seaborn

nltk (for natural language processing tasks)

### Conclusion
In this project, Multinomial Naive Bayes and Gaussian Naive Bayes performed well with both TF-IDF and CountVectorizer features, achieving an accuracy of 87% when using TF-IDF. The Bernoulli Naive Bayes model showed moderate performance with an accuracy of 78% due to the nature of the data not being binary enough for optimal results.

When using CountVectorizer, the models performed similarly, but TF-IDF provided a more nuanced representation of the data, particularly for the Multinomial Naive Bayes model. This indicates that TF-IDF outperforms CountVectorizer in capturing the important features for classification tasks, especially when dealing with word frequency and relevance across documents.
