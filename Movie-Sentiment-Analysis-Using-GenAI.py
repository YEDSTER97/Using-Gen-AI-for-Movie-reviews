# a simple tool to help you decide if a movie is worth watching based on text reviews

# Sample Dataset
reviews = ['This movie was fantastic! A must-watch.',
           'I didn\'t enjoy this movie at all.',
           'Amazing storyline and great acting!',
           'The plot was dull and predictable.',
           'Loved the cinematography! Highly recommended.']

labels = ['positive', 'negative', 'positive', 'negative', 'positive']


# 1. using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Vectorize the Text Data
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size = 0.3, random_state = 42)

# Loading the model
model = MultinomialNB()
model.fit(x_train,y_train)

# Prediction & Accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy Score: ", accuracy)

# Infering from the result
if accuracy > 0.8: 
    print("Must Watch! Good Vibes!! Book the tickets!!!")
else: print("Needs More Work")


# 2. Using TextBlob
from textblob import TextBlob

# Converting list to string
lns = " ".join(reviews)
blob = TextBlob(lns)

pol = blob.sentiment.polarity
sub = blob.sentiment.subjectivity

if pol > 0.8:
    print("Overall This movie gives good vibes :)")
else:
    print("Nah! not worth it")