import random 
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model  # type: ignore # Updated import statement

lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r"C:\Users\Kalash Srivastava\python\Python Practice\intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')  # Ensure the filename matches your saved model

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lowercasing
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)  # Return outside the loop

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)  # Sort results
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents_json = intents_json['intents']
    for i in list_of_intents_json:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            return result  # Move return outside the loop

print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    if ints:
        res = get_response(ints, intents)
        print(res)
    else:
        print("I'm not sure how to respond to that.")
