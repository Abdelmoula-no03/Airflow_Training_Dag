import numpy as np
import pandas as pd
import re
import nltk
import os
import pickle
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.remove('not')
lemmatizer = WordNetLemmatizer()
repertoire_actuel = os.path.dirname(os.path.abspath(__file__))


def data_preprocessing(tweet):
    # data cleaning
    tweet = re.sub(re.compile('<.*?>'), '', tweet) #removing html tags
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet) #taking only words

    # lowercase
    tweet = tweet.lower()

    # tokenization
    tokens = nltk.word_tokenize(tweet)

    # stop_words removal
    tweet = [word for word in tokens if word not in stop_words] #removing stop words

    # lemmatization
    tweet = [lemmatizer.lemmatize(word) for word in tweet]

    # join words in preprocessed tweet
    tweet = ' '.join(tweet)

    return tweet

def extract_clean():
    # Concaténez le nom du fichier pour obtenir le chemin absolu complet
    chemin_fichier = os.path.join(repertoire_actuel, 'data.csv')

    # Lire le fichier CSV
    data = pd.read_csv(chemin_fichier,encoding='latin-1')
    sub_dataset_size = 0.5  # 80% 

    # group the data by the categorical variable
    groups = data.groupby('label')

   # create an empty DataFrame to store the sub-dataset
    sub_df = pd.DataFrame()

   # for each group, randomly select a subset and append it to the sub-dataset
    for _, group in groups:
        sub_group = group.sample(frac=sub_dataset_size)
        sub_df = sub_df.append(sub_group)

   # faire une copie de la dataset ainsi créée
    df = sub_df.copy()

   # application de la fonction de nettoyage sur les lignes de la dataset
    
    df['preprocessed_review'] = df['review'].apply(lambda tweet: data_preprocessing(tweet))

    return df


def train_test_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score


    #on définit les données d'entrainement et de test
    y = data['label'].values
    data.drop(['label'],axis =1,inplace = True)
    X_train, X_test, y_train, y_test = train_test_split(data, y,test_size=0.2, stratify=y)
        
    # load the saved vectorizer
    chemin_fichier1 = os.path.join(repertoire_actuel, 'vectorizer.pkl')
    with open(chemin_fichier1,'rb') as f:
        vectorizer = pickle.load(f)

    #transform data with vectorizer
    X_train_tweet = vectorizer.transform(X_train['preprocessed_review'])
    X_test_tweet = vectorizer.transform(X_test['preprocessed_review'])    

    # load the saved model 
    chemin_fichier = os.path.join(repertoire_actuel, 'my_model.pkl')
    with open(chemin_fichier,'rb') as file:
        model = pickle.load(file)

    model.warm_start= True
    # calcul de l'ancien score    
    y_pred = model.predict(X_test_tweet)
    ancien_acc = round(accuracy_score(y_test,y_pred), 4)    

    model.warm_start= True    

    # retrain the model 
    model.fit(X_train_tweet,y_train)
    y_pred = model.predict(X_test_tweet)
    new_acc = round(accuracy_score(y_test,y_pred), 4)

    #save the updated model
    with open(chemin_fichier,'wb') as file:
         pickle.dump(model,file)

    return new_acc, ancien_acc
    



def get_data_test():
    # Concaténez le nom du fichier pour obtenir le chemin absolu complet
    chemin_fichier = os.path.join(repertoire_actuel, 'data.csv')

    # Lire le fichier CSV
    data = pd.read_csv(chemin_fichier,encoding='latin-1')
    return data