#Email Spam Detector Using Naive Bayes Algorithm

# Importing the libraries
import numpy as np
import pandas as pd
import re
import nltk

#Reading dataset
dataset = pd.read_csv('emails.csv')

#Checking for duplicate data and deleting them
dataset.drop_duplicates(inplace = True)

#Checking for missing data
dataset.isnull().sum()

#Cleaning Text (Removing punctuation, reducing words down to base form)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
dataset_size = dataset.shape[0]
#email = re.sub(r'[-()\"#/@;:<>{}=~|.?,]', ' ', dataset.iloc[0][2471])
#email = re.sub('[^a-zA-Z]', ' ', email)
for i in range(0 , dataset_size):
    try:
       email = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
       email = email.lower()
       email = email.split()
       ps = PorterStemmer()
       email = [ps.stem(word) for word in email if not word in set(stopwords.words('english'))]
       email = ' '.join(email)
       corpus.append(email)
    except:
        print ('Exception Occurred at' + str(i))
    
#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating and Running the naive bayes algorithm 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB().fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)