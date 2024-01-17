import pickle
import streamlit as st
from PIL import Image


st.title("Spam or Ham Classifier")

input= st.text_area("Enter message: ","")

#loading model and feature extractor pkl files
feature_extraction=pickle.load(open("spam_feature_extractor.pkl","rb"))
model=pickle.load(open("spam_classifier.pkl","rb"))



def spam_classifier(text):
    text_vectorized=feature_extraction.transform([text])
    prediction=model.predict(text_vectorized)
    return "spam" if prediction[0]==1 else "ham"

#verify button
if st.button("verify"):
    if input:
        result=spam_classifier(input)
        st.write(f"The text is classified as: {result}")

        if result=="spam":
            st.image(Image.open("spam.jpg"),caption="Spam",use_column_width=True)
        
        else:
            st.image(Image.open("ham.png"),caption="Ham",use_column_width=True)