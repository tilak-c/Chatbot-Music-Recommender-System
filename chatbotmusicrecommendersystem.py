import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
import webbrowser
import requests 
import time
from pygame import mixer
import billboard 


from googlesearch import search
from nltk.stem import WordNetLemmatizer
from covid import Covid
import nltk
import pandas as pd

# Load EmoLex lexicon
nltk.download('nrc_lexicon')
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Load EmoLex lexicon
nltk.download('opinion_lexicon')
from nltk.corpus import opinion_lexicon

# Load AFINN lexicon
from afinn import Afinn
import pandas as pd
import random

lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

data_file=open('intents.json').read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag'])) 
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes) 

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]
    
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training)  
X_train=list(training[:,0])
y_train=list(training[:,1])  

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
weights=model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)    
model.save('mymodel.h5',weights)

from keras.models import load_model
model = load_model('mymodel.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


#Predict
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    
    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def get_response(return_list,intents_json):
    
    if len(return_list)==0:
        tag='noanswer'
    else:    
        tag=return_list[0]['intent']
    if tag=='datetime':        
        print(time.strftime("%A"))
        print (time.strftime("%d %B %Y"))
        print (time.strftime("%H:%M:%S"))

    if tag=='google':
        query=input('Enter query...')
        chrome_path = r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
        for url in search(query, tld="co.in", num=1, stop = 1, pause = 2):
            webbrowser.open("https://google.com/search?q=%s" % query)
    if tag=='weather':
        api_key='987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = input("Enter city name : ")
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url) 
        x=response.json()
        print('Present temp.: ',round(x['main']['temp']-273,2),'celcius ')
        print('Feels Like:: ',round(x['main']['feels_like']-273,2),'celcius ')
        print(x['weather'][0]['main'])
        
    if tag=='news':
        main_url = " http://newsapi.org/v2/top-headlines?country=in&apiKey=bc88c2e1ddd440d1be2cb0788d027ae2"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = [] 
          
        for ar in article: 
            results.append([ar["title"],ar["url"]]) 
          
        for i in range(10): 
            print(i + 1, results[i][0])
            print(results[i][1],'\n')
            
    
    if tag=='song':
        chart=billboard.ChartData('hot-100')
        print('The top 10 songs at the moment are:')
        for i in range(10):
            song=chart[i]
            print(song.title,'- ',song.artist)
    if tag=='timer':        
        mixer.init()
        x=input('Minutes to timer..')
        time.sleep(float(x)*60)
        mixer.music.load('Handbell-ringing-sound-effect.mp3')
        mixer.music.play()
    if tag=="covid19":
        covid = Covid()
        x=input('Enter the country name').lower()
        if x=="world":
            covid=Covid(source="worldometers")
            confirmed = covid.get_total_confirmed_cases()
            print('Confirmed :', end =" ")
            print(confirmed)

            active = covid.get_total_active_cases()
            print("Active:", end =" ")
            print(active)

            recovered = covid.get_total_recovered()
            print('Recovered:', end =" ")
            print(recovered)

            deaths = covid.get_total_deaths()
            print('Deaths:', end =" ")
            print(deaths)
        
        else:
            country = covid.get_status_by_country_name(x)

            data ={
                    key:country[key]
                    for key in country.keys() and {'confirmed', 
                                                 'active',
                                                 'deaths',
                                                 'recovered'}
                }
  
        print(data)
#     if tag=='covid19':
        
#         covid19=COVID19Py.COVID19(data_source='csbs')
#         country=input('Enter Location...')
        
#         if country.lower()=='world':
#             latest_world=covid19.getLatest()
#             print('Confirmed:',latest_world['confirmed'],' Deaths:',latest_world['deaths'])
        
#         else:
                   
#             latest=covid19.getLocations()
            
#             latest_conf=[]
#             latest_deaths=[]
#             for i in range(len(latest)):
                
#                 if latest[i]['country'].lower()== country.lower():
#                     latest_conf.append(latest[i]['latest']['confirmed'])
#                     latest_deaths.append(latest[i]['latest']['deaths'])
#             latest_conf=np.array(latest_conf)
#             latest_deaths=np.array(latest_deaths)
#             print('Confirmed: ',np.sum(latest_conf),'Deaths: ',np.sum(latest_deaths))
    

    list_of_intents= intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result

def response(text):
    return_list=predict_class(text,model)
    response=get_response(return_list,intents)
    return response
texts = []
print('You are now chatting with the chatbot \n')
while(1):
    x=input()
    texts.append(x)
    print(response(x))
    if x.lower() in ['bye','goodbye','bye bye','see you']:  
        break


afn = Afinn()

# Define a function to extract emotions from a list of texts
def extract_emotions(texts):
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Tokenize the texts and concatenate into one list
    tokens = []
    for text in texts:
        tokens += word_tokenize(text)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]

    # Initialize emotion dictionary with all dimensions
    emotions = {'anger': 0, 'anticipation': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}

    # Get the sentiment score for the texts
    for text in texts:
        sentiment_scores = sid.polarity_scores(text)
        for key in emotions.keys():
            if key in sentiment_scores:
                emotions[key] += sentiment_scores[key]

    # Check for words in EmoLex lexicon
    for token in tokens:
        if token in opinion_lexicon.negative():
            emotions['anger'] += 1
            emotions['disgust'] += 1
            emotions['sadness'] += 1
            emotions['fear'] += 1
        elif token in opinion_lexicon.positive():
            emotions['anticipation'] += 1
            emotions['joy'] += 1
            emotions['trust'] += 1
            emotions['surprise'] += 1
        # Check for words in AFINN lexicon
        elif afn.score(token) > 0:
            emotions['anticipation'] += afn.score(token)
            emotions['joy'] += afn.score(token)
            emotions['trust'] += afn.score(token)
            emotions['surprise'] += afn.score(token)
        elif afn.score(token) < 0:
            emotions['anger'] += abs(afn.score(token))
            emotions['disgust'] += abs(afn.score(token))
            emotions['sadness'] += abs(afn.score(token))
            emotions['fear'] += abs(afn.score(token))
        # Check for words in EmoLex lexicon
        synsets = wn.synsets(token)
        if synsets:
            synset = synsets[0]
            for lemma in synset.lemmas():
                if lemma.name() in emotions.keys():
                    emotions[lemma.name()] += 1

    # Normalize emotion scores
    total_score = sum(emotions.values())

    return emotions

# Input texts to analyze
#texts = ["I'm so happy today!", "I'm really angry about this situation", "
#texts = ["I'm I'm really angry about this situation", "I'm feeling quite anxious before the big presentation", "The news of his death was so sad", "I feel very calm and sad today", "This roller coaster is making me feel so scared"]


emotions = extract_emotions(texts)
print(texts)
print(emotions)
#my_dict = {'a': 10, 'b': 5, 'c': 20}
max_key = max(emotions, key=emotions.get)
print(max_key)
val = input("Do you want to listen to a song? Yes/No ")

if (val == ('yes' or 'Yes')):
    # Load Excel sheet into a DataFrame
    df = pd.read_excel('Songs.xlsx', sheet_name='Sheet1')

    # Select a specific column by column name
    selected_column = df[max_key]

    # Select a random index from the column
    random_index = random.randint(0, len(selected_column)-1)

    # Select the cell at the random index
    random_cell = selected_column[random_index]

    # Print the randomly selected cell
    print(random_cell)

    import webbrowser

    # Open a specific YouTube channel page
    username = random_cell
    url = f'https://www.youtube.com/results?search_query={username}'
    webbrowser.open(url)
    
elif (val==('No' or 'no')):
    print("Alright !, Good Bye !")

else:
    print("Sorry that is invalid input")