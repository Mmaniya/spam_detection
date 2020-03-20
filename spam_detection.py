from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,accuracy_score

KNN = KNeighborsClassifier(n_neighbors=1)

def train_classifier(clf, X_train, y_train):
    clf.fit(X_train, y_train)

def predict_labels(clf, features):
    return(clf.predict(features))

def tend_to_predict():
  for a in range(0,5):
    print(objects[a])
    train_classifier(clf[a], X_train, y_train)
    y_pred = predict_labels(clf[a],X_test)
    stopset = set(stopwords.words("english"))
    vectorizer = CountVectorizer(stop_words=stopset,binary=True)
    vectorizer = CountVectorizer()

df = pd.read_csv('./spam.csv', encoding='latin-1')
print ('show what kind of data we are dealing with')
print (df.head())
data_train, data_test, labels_train, labels_test = train_test_split(df.v2,df.v1, test_size=0.2, random_state=0)

def GetVocabulary(data): 
    vocab_set = set([])
    for document in data:
        words = document.split()
        for word in words:
            vocab_set.add(word) 
    return list(vocab_set)

vocab_list = GetVocabulary(data_train)
print ('Number of all the unique words : ' + str(len(vocab_list)))

def Document2Vector(vocab_list, data):
  word_vector = np.zeros(len(vocab_list))
  words = data.split()
  for word in words:
    if word in vocab_list:
      word_vector[vocab_list.index(word)] += 1
  return word_vector

print (data_train[1:2,])
ans = Document2Vector(vocab_list,"the the the")
print (data_train.values[2])

train_matrix = []
for document in data_train.values:
    word_vector = Document2Vector(vocab_list, document)
    train_matrix.append(word_vector)

print (len(train_matrix))



def NaiveBayes_train(train_matrix,labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    
    spam_vector_count = np.ones(num_words);
    ham_vector_count = np.ones(num_words)  
    spam_total_count = num_words;
    ham_total_count = num_words                  
    
    spam_count = 0
    ham_count = 0
    for i in range(num_docs):
        if i % 500 == 0:
            print ('Training Proress:' + str(i))
            
        if labels_train[i] == 'spam':
            ham_vector_count += train_matrix[i]
            ham_total_count += sum(train_matrix[i])
            ham_count += 1
        else:
            spam_vector_count += train_matrix[i]
            spam_total_count += sum(train_matrix[i])
            spam_count += 1
    
    print (ham_count)
    print (spam_count)
    
    p_spam_vector = np.log(ham_vector_count/ham_total_count)
    p_ham_vector = np.log(spam_vector_count/spam_total_count)
    return p_spam_vector, np.log(spam_count/num_docs), p_ham_vector, np.log(ham_count/num_docs)

    
p_spam_vector, p_spam, p_ham_vector, p_ham = NaiveBayes_train(train_matrix, labels_train.values)

    
def Predict(test_word_vector,p_spam_vector, p_spam, p_ham_vector, p_ham):
    spam = sum(test_word_vector * p_spam_vector) + p_spam
    ham = sum(test_word_vector * p_ham_vector) + p_ham
    if spam > ham:
        return 'spam'
    else:
        return 'ham'

predictions = []
i = 0
for document in data_test.values:
    if i % 200 == 0:
        print ('Testing Progress:' + str(i))
    i += 1    
    test_word_vector = Document2Vector(vocab_list, document)
    ans = Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)
    predictions.append(ans)

print (len(predictions))


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

print (accuracy_score(labels_test, predictions))

def predict():
  input_msg = input("Enter Your Message:")
  print(Predict(Document2Vector(vocab_list,input_msg), p_spam_vector, p_spam, p_ham_vector, p_ham))



def test():
  predict()
  exit_msg()

def exit_msg():
  user_answer = input("Are You Sure Want To Continue(true/false): ").lower().strip()
  if user_answer == "true":
    test()
  elif user_answer == "false":
    exit()
  else:
      print("Error: Answer must be true or false")

test()
