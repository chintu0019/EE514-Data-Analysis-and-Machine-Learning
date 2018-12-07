import pandas as pd
import os
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from nltk.corpus import stopwords

#from the nltk corpus of stopwords
stop_words = set(stopwords.words('english'))

spam_subject = []
spam_text = []

ham_subject = []
ham_text = []

#iterates through each file in enron/spam directory
for filename in os.listdir('../dataset/preprocessed/enron1/spam/'):
    
    #takes out newlines in the file, each line is put into the list 'lines'
    lines = [line.rstrip('\n') for line in open('../dataset/preprocessed/enron1/spam/'+filename)]
    
    #lines[0] gets the first line, line[0][9:] 
    #gets the substring of line[0] starting from index 9.
    #lower() makes it all lowercase
    tempsubject = lines[0][9:].lower()
    
    #if the word contains only letters or numbers or both, keep it. Else, toss.
    cleaned = [e for e in tempsubject.split(' ') if e.isalnum()]
    
    #if word is not useless/stopword e.g. 'is', 'the', etc. then keep it, else toss. 
    #Also toss blanks.
    cleaned = [word for word in cleaned if word not in stop_words and word!='']
    
    #concatenates all elements of the 'cleaned' list with spaces in between the words.
    #i.e. makes a sentence.
    cleaned = ' '.join(cleaned)
    
    if len(cleaned)>0:
        spam_subject.append(cleaned)
    else:
        spam_subject.append(' ')#doing this just to have equal length vectors
    
    temp_spam_text = ' '#placeholder for a single message content
    
    #so far we have just dealt with the first line - the subject line.
    #now we move onto the rest - the message/content/body.
    for line in lines[1:]:
        temptext = line.lower()
        cleaned = [e for e in temptext.split(' ') if e.isalnum()]
        cleaned = [word for word in cleaned if word not in stop_words and word!='']
        cleaned = ' '.join(cleaned)
        if len(cleaned)>0:
            #put the whole message together by appending each cleaned line to the previous ones
            temp_spam_text+=cleaned + ' '
    
    #finally put the cleaned entire message with no blank spaces at the ends into
    #the spam_text list. This is the final version of the cleaned message for this file.
    spam_text.append(temp_spam_text.strip())
            
for filename in os.listdir('enron/ham/'):
    lines = [line.rstrip('\n') for line in open('enron/ham/'+filename)]
    tempsubject = lines[0][9:].lower()
    cleaned = [e for e in tempsubject.split(' ') if e.isalnum()]
    cleaned = [word for word in cleaned if word not in stop_words and word!='']
    cleaned = ' '.join(cleaned)
    if len(cleaned)>0:
        ham_subject.append(cleaned)
    else:
        ham_subject.append(' ')
    
    temp_ham_text = ' '
    for line in lines[1:]:
        temptext = line.lower()
        cleaned = [e for e in temptext.split(' ') if e.isalnum()]
        cleaned = [word for word in cleaned if word not in stop_words and word!='']
        cleaned = ' '.join(cleaned)
        if len(cleaned)>0:
            temp_ham_text+=cleaned+' '
    ham_text.append(temp_ham_text.strip())

print(len(ham_subject))
print(len(ham_text))
print(len(spam_subject))
print(len(spam_text))

#I thought this was more useful instead of separate subject/message
#thought of this a bit late so I kept the previous code anyway
#basically put the subject and message together into the ham_subj_text list.
ham_subj_text = []
for i in range(len(ham_subject)):
    ham_subj_text.append(ham_subject[i]+' '+ham_text[i])
    
spam_subj_text = []
for i in range(len(spam_subject)):
    spam_subj_text.append(spam_subject[i]+' '+spam_text[i])

#create a new pandas dataframe with columns 'subject', 'text', 'subj_text', 'class'
#class is the vector which contains the label for each message i.e. ham/spam
#class is encoded as 1 for ham, 0 for spam
data = pd.DataFrame({'subject':ham_subject+spam_subject,
                      'text':ham_text+spam_text,
                      'subj_text':ham_subj_text+spam_subj_text,
                      'class':[1]*len(ham_subject)+[0]*len(spam_subject)})
print(len(data))
data.head()

#epic feature engineering with countvectorizer - simply put, it counts the # of whatever word
#was in your message, and puts it in the appropriate unique column.
#I believe it's one column per word if I recall correctly. 
#VERY sparse matrix. 
count_vectorizer = CountVectorizer()

#creates this matrix that I was writing about above
count_vectorizer.fit(data['subj_text'].values)
counts = count_vectorizer.transform(data['subj_text'].values)

#start making a model
clf = LogisticRegression()

#3-fold cross-validation of the model, 5-fold is more often used but if 3-fold performs well,
#then your model is golden.
#clf,counts,data['class'] is just the model,data/matrix,class-labels for each row in the matrix
#cv=3 is the number of folds in k-fold cross-validation (cross-validation = cv)
scores = cross_val_score(clf, counts, data['class'], cv=3)

#print out the average of the 3 values from the 3-fold cross-validation
print('Accuracy:  ', np.mean(scores))

#can make precision/recall/f1/etc. with different scoring
precisions = cross_val_score(clf, counts, data['class'], cv=3, scoring='precision_weighted')
print('Precision: ', np.mean(precisions))
recalls = cross_val_score(clf, counts, data['class'], cv=3, scoring='recall_weighted')
print('Recall:    ', np.mean(recalls))
f1s = cross_val_score(clf, counts, data['class'], cv=3, scoring='f1_weighted')
print('F1:        ', np.mean(f1s))


#here's an example spam message in my gmail account
#how do I know this is spam? 
#My name wasn't mentioned, the email is from a chinese website which I had never visited
#nothing specific about it
#etc. etc. 

line = '''At the moment our team is looking for a manager in your area. We are
looking for somebody who is ready to learn and start immediately. After
reviewing your CV, we came to the conclusion that you are an ideal
candidate for this job position.

Our company is engaged in providing services in the area of health
insurance. During our work, we have helped thousands of people around the
world and we earned an irrefutable reputation. Now you have the opportunity
to become a part of our friendly team.

Position requirements:
- You must be a US citizen.
- Your age must be more than 21 years.
- You must have a stable internet connection.
- You must be willing to learn and improve.

Position responsibilities:
- Keeping your projects documentation and filling of reports.
- Providing quality service to clients of the company.
- Perform the tasks on time.
- Close cooperation with other managers and our experts.

Your salary will be 3000 $ per month plus 500 $ every week. Also, you will
have the personal bonuses. If you are ready to start working, respond to
this email. We will give you a trial period after which you can decide this
job is right for you or not. Hope to hear from you soon.

Best regards,
Orli Irwin.'''

#gotta clean it the same way I did with the training examples for it to work properly
temptext = line.lower()
cleaned = [e for e in temptext.split(' ') if e.isalnum()]
cleaned = [word for word in cleaned if word not in stop_words and word!='']
cleaned = ' '.join(cleaned)
transformed = count_vectorizer.transform([cleaned])

#make the model, train the model, make a prediction.
clf = LogisticRegression()
clf.fit(counts,data['class'])

#probabilities for choosing a class. first in the array is 0's prob, next is 1's prob.
#picks the one with the highest prob. 
print(clf.predict_proba(transformed))

#spits out the highest prob prediction.
print(clf.predict(transformed))