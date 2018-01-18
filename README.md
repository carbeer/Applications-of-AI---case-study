# Applications of AI - Case Study

#### Team members
Amal, Carolin, Chris, Leon, Yue

#### Usage
Open the shell in the main directory and execute the script using  ```python '.\Prediction process.py'```  

#### Short description
This is a python script to classify reviews into positive or negative sentiments using the **Bag Of Words model**.
##### Training
Firstly, we iteratively processed the raw data from the reviews
1. Stemming *(EnglishStemmer, nltk snowball package)*
2. Stopword removal *(nltk corpus package)*
3. Lemmatization *(WordNetLemmatizer, nltk corpus package)*

After cleaning the data, we created a dictionary of words that occur in the documents and created a document-term matrix *(CountVectorizer, sklearn package)*.
Finally the model is trained using the multinominal Naive Bayes classifier *(MultinomialNB, sklearn package)*.
##### Predictions
The input data will undergo the same preprocessing and then, the trained model can make predictions using the Naive Bayes classifier.
The output will be written into a csv file: **'predictions.csv'**

#### Hard facts
- The **data split** is as follows: 70% of the training data is used to train the model, 30% are used for testing purposes. 
- Our tests achieved an **accuracy of 87%**.
