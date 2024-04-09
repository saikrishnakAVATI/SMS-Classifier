import streamlit as st
import pickle

#lets load the saved vectorizer and naive model
tfidf = pickle.load(open('vectorizer.pk1','rb'))
model = pickle.load(open('model.pk1','rb'))

#transform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
   text = text.lower() #converting to lowercase
   text = nltk.word_tokenize(text) #Tokenize
   text = [word for word in text if word.isalnum()] #removing special characters and retaining alphanumeric words

   text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]#Removing stopwords and punctuations

   text = [ps.stem(word) for word in text] #applying stemming
   return " ".join(text)

#Streamlit code
st.title("SMS Classifier")
input_sms = st.text_area("Enter message")

if st.button('Predict'):
    #preprocess
    transformed_sms = transform_text(input_sms)
    #vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display
    if result == 1:
      st.header("Spam")
    else:
      st.header("Not Spam")

    