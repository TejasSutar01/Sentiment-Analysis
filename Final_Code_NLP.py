# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

df= pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\EXCELR ASSIGMENTS\\NLP PROJECT\\FINAL_REVIEWS\\lifebouy_datesorted_final.csv",na_values="-",encoding='latin1')
df.head
df.isnull().sum()#754
df.dropna(inplace=True)
df.isnull().sum()
df.head#752
df=df.drop(['CutomerName'], axis=1)
df.head
#Storing the reviews in sepearte Dataframe
sep_rev=df['Reviews']
#Lowering the words & Removing numbers from it
sep_rev=re.sub("[^A-Za-z" "]+"," ",str(sep_rev)).lower()
sep_rev=re.sub("[0-9" "]+"," ",str(sep_rev)) 
#Extracting the words
cm_words = sep_rev.split(" ")

####Importing stopwords
with open("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\EXCELR ASSIGMENTS\COMPLETED\TEXT MINING\AMAZON\stop.txt","r") as sw:
        stopwords = sw.read()
        stopwords=stopwords.split("\n")        
        cm_words=[w for w in cm_words if w not in stopwords]
        sep_rev = " ".join(cm_words)       

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in df['Reviews']:
    pol_score = sia.polarity_scores(line)
    pol_score['Reviews'] = line
    results.append(pol_score)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
def compound_score(text):
    comp=sentiment_analyzer_scores(text)
    return comp['compound']

df['compound_score']=df['Reviews'].apply(lambda x:compound_score(x))

    
#{'neg': 0.0, 'neu': 0.308, 'pos': 0.692, 'compound': 0.9256, 
#'Reviews': 'Wow its amazing and its smells very nice like a perfume.'}    

#extracted_data1= pd.DataFrame.from_records(results)
df.head
df['Sentiment'] = 0
df.loc[df['compound_score'] > 0.2, 'Sentiment'] = 1
df.loc[df['compound_score'] < -0.2, 'Sentiment'] = -1
df.head()

#calculating sentiment polarity and Subjectivity
from textblob import TextBlob        
polarity=[]
for i in df['Reviews'].values:
    try:
        analysis=TextBlob(i)
        polarity.append(analysis.sentiment.polarity)        
    except:
        polarity.append(0)        
df['Polarity']=polarity
df[df.Polarity>0].head(10)     
df.head  
df=df.drop(["Unnamed: 0"],axis=1)  
df.to_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\EXCELR ASSIGMENTS\\NLP PROJECT\\MY_REVIEWS\\san_rev.csv")

import datetime
df['Date'] = pd.to_datetime(df['Date'],errors='coerce')
df['Month'] = df['Date'].dt.month_name()
df.head 
import seaborn as sns
sns.barplot(x= 'Month', y = 'Polarity', data = df)
df.set_index(['Date'],inplace=True)
plt.figure(figsize=(10,10))
plt.plot(df.index, df['Polarity'])
plt.xlabel("Date")
plt.ylabel("Polarity")
sns.distplot(polarity)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(df['Reviews'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts, df['Sentiment'], test_size=0.3, random_state=1)
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
pred_nb= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, pred_nb))
#MultinomialNB Accuracy: 0.7256637168141593


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['Reviews'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_tf, df['Sentiment'], test_size=0.3, random_state=123)
tclf = MultinomialNB().fit(X_train, y_train)
pred_nb_tf= tclf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, pred_nb_tf))
#MultinomialNB Accuracy: 0.6814159292035398

####Random forest
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(text_tf, df['Sentiment'], test_size=0.3, random_state=0)
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)
pred_RFC = text_classifier.predict(X_test)
print("RandomForestClassifier Accuracy:",metrics.accuracy_score(y_test, pred_RFC ))
#RandomForestClassifier Accuracy: 0.8008849557522124
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,pred_RFC))
print(classification_report(y_test,pred_RFC))
print(accuracy_score(y_test, pred_RFC))

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(text_tf, df['Sentiment'], test_size=0.3, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = text_classifier.predict(X_test)
print("lr Accuracy:",metrics.accuracy_score(y_test, lr_pred))
#logistic reg Accuracy: 0.7964601769911505


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
#
data= pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\EXCELR ASSIGMENTS\\NLP PROJECT\\MY_REVIEWS\\san_rev.csv")
stk = pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\EXCELR ASSIGMENTS\\NLP PROJECT\\STOCK MARKET DATA\\Hul_all.csv")
#Taking 3 months data
stk_3= stk.iloc[82:,]
stk_3.isnull().sum() ## no null values

sorted_data = data.sort_values(by=['Date'])   # 754 total
sorted_data.isnull().sum()
sorted_data = sorted_data.dropna() ## removing na values and null values 752 total

stk_3.columns
stk_3['ClosePrice'] = MinMaxScaler().fit_transform(stk_3['Close'].values.reshape(-1,1))


stk_3['Date'] = pd.to_datetime(stk_3['Date'])

###Creating new dataframe with polarity values to compare with stock price
rev_days = pd.DataFrame(columns= ['polarity'])
rev_days[['polarity']] = df[['Polarity']].astype(float)
rev_days.index = pd.to_datetime(df['Date'])
rev_days= rev_days.resample('D').mean().ffill()

rev_days['Date'] = rev_days.index
rev_days.reset_index(inplace= True,drop=True)
########Merging stock price & reviews columns
final_rev = pd.merge(stk_3,rev_days,on='Date')
final_rev['dates'] = pd.to_datetime(final_rev['Date']).dt.date

######### plots
final_rev.columns
######################## Open Price vs Close Price
x =final_rev['dates']

y1 =final_rev.loc[:,['ClosePrice']]
y2 = final_rev['polarity']
#stk=stk.loc[:, ['Date','Close']] 

# Plot with differently-colored markers.
plt.plot(x, y1, 'b-', label='Close price')
plt.plot(x, y2, 'g-', label='polarity')
plt.legend(loc='upper left')
plt.xlabel('dates')
plt.ylabel('Polarity')

# Plot Line1 (Left Y Axis)
fig, ax1 = plt.subplots(1,1,figsize=(16,9))
ax1.plot(x, y1, color='tab:blue')
# Plot Line2 (Right Y Axis)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(x, y2, color='tab:red')

# Decorations
# ax1 (left Y axis)
ax1.set_xlabel('date', fontsize=20)
ax1.tick_params(axis='x', rotation=40, labelsize=12)
ax1.set_ylabel('ClosePrice', color='tab:red', fontsize=20)
ax1.tick_params(axis='y', rotation=40, labelcolor='tab:red' )
ax1.grid(alpha=.4)
# ax2 (right Y axis)
ax2.set_ylabel("polarity", color='tab:blue', fontsize=20)
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_xticks(np.arange(0, len(x), 60))
ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
ax2.set_title("closing price vs polarity : Plotting in Secondary Y Axis", fontsize=22)
fig.tight_layout()
plt.show()

## single plot
x= final_rev['Date']
y= final_rev['polarity']
plt.figure(figsize=(30,10))
plt.plot(x,y ,color='blue')
plt.title('date vs polarity',fontsize=28)
plt.xlabel('stock Date',fontsize=28)
plt.ylabel('polarity',fontsize=28)
plt.xticks(rotation=40)
plt.grid(linewidth=1)
plt.show()








