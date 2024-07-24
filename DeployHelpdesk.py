##https://medium.com/analytics-vidhya/deploy-your-first-end-to-end-ml-model-using-streamlit-51cc486e84d7

import streamlit as st
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict
from gensim import corpora
from gensim import models,similarities
from gensim.parsing.preprocessing import remove_stopwords

model = pickle.load(open('lsimodel.pkl','rb'))

#Function to cleanse document
def process_document(document):

    #Remove stopwords, convert to lower case and remove "?" character
    cleaned_document = remove_stopwords(document.lower()).replace("?","")  
    return cleaned_document.split()

#Function to get recommended FAQ
def get_recommendedfaq(question):
    #Read the input CSV into a Pandas dataframe
    helpdesk_data = pd.read_csv("helpdesk_dataset.csv")

    #Extract the Question column 
    documents = helpdesk_data["Question"]
    
    #Create a document vector
    doc_vectors=[process_document(document) for document in documents]

    #Create the dictionary
    dictionary = corpora.Dictionary(doc_vectors)
    
    #Create a corpus
    corpus = [dictionary.doc2bow(doc_vector) for doc_vector in doc_vectors] 
    
    #Pre Process the Question 
    question_corpus = dictionary.doc2bow(process_document(question))
    
    #Create an LSI Representation
    vec_lsi = model[question_corpus] 
    
    #Create a similarity Index
    index = similarities.MatrixSimilarity(model[corpus])

    #Find similarity of the question with existing documents
    sims = index[vec_lsi] 

    #Find the corresponding FAQ Link
    #sort an array in reverse order and get indexes
    matches=np.argsort(sims)[::-1] 

    return (helpdesk_data.iloc[matches[0]]["LinkToAnswer"])


def main():
    #st.title("FAQ Recommender")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> FAQ Recommender </h2>
    </div>
    <p />
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    Question = st.text_input("Question","Type Here")

    if st.button("Recommend"):
        output = get_recommendedfaq(Question)
        st.success('Recommended FAQ : {}'.format(output))

if __name__=='__main__':
    main()