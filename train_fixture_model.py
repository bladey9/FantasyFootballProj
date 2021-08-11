#!/usr/bin/env python
# coding: utf-8

# ### Train data for Gameweek n
# 
# - To predict a gameweek n, train data up to n-1 for the Fixture Model
# - Random Forests are used as our Machine Learning Algorithm

# In[3]:


import numpy as np
import pandas as pd
pd.options.display.max_columns = None
import requests
np.set_printoptions(suppress=True)
from numpy import loadtxt
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib as plt
from matplotlib import pyplot as plt


# In[8]:


class trainrf():
    
    def __init__(self,training_dataset,gameweek_id):
        #training_dataset = historcal data used for training
        #gameweek_id = to see which data to predict
        
        self.training_dataset = training_dataset
        self.gameweek_chosen = gameweek_id
        self.fixture_prediction()
        self.gameweek_static = self.gameweek.copy()
        self.gameweek = self.gameweek.copy()
        self.predictions_r = {}
        self.train()
        
    def train(self):
        self.X,self.y = self.prepare_data(self.dataset)
        self.X_test, self.y_test = self.prepare_data(self.gameweek)
        self.randomforest()
    
    def fixture_prediction(self):
        #Method to see which matches are in gameweek described
        #@params None
        #@return None
        
        partial_dataset = self.training_dataset.copy()
    
        #Prepare gameweek data to see which matches are being played in gameweek n
        url = 'https://fantasy.premierleague.com/api/fixtures?event='+ str(self.gameweek_chosen)
        r = requests.get(url)
        json_new = r.json()
        fixtures_df = pd.DataFrame(json_new)
        #Retrieve home and away team ids for the first and last fixture of gameweek n 
        home_team = fixtures_df.iloc[0]["team_h"]
        away_team = fixtures_df.iloc[0]["team_a"]
        home_team_last = int(fixtures_df.iloc[-1:]["team_h"])
        away_team_last = int(fixtures_df.iloc[-1:]["team_a"])
        #Create a new dataset - up to the first fixture of the gameweek n chosen. 
        #Does not include the gameweek n in dataset as this is what is being predicted
        for index,row in partial_dataset.iterrows():
            teamh = partial_dataset["Team A"][index]
            teama = partial_dataset["Team B"][index]
            if int(teamh) == int(home_team) and int(teama) == int(away_team):
                #index represents the fixture row which is the first fixture of gameweek n. 
                self.dataset = partial_dataset[:index]
                first = index
            if teamh == home_team_last and teama == away_team_last:
                #get the index of the dataset corresponding to the last fixture of gameweek n
                last = index

        #Using the first and last fixtures of gameweek n, we can grab a snipped of the dataset_training file 
        self.gameweek = partial_dataset[first:last+1]
        
    def prepare_data(self, dataset):
        #Method to load dataframe into appropraite format
        #@params None
        #@return X, Y = Our X data and Y label data
        dataset["labels"]  = dataset["result"]
        del dataset["Team A"]
        del dataset["Team B"]
        del dataset["Fixture"]
        del dataset["result"]

        dataset.to_csv(r'datasetML.csv',header = False, index = False)
        dataset = loadtxt(r'datasetML.csv', delimiter=',')

        X = dataset[:,0:25].astype(float)
        Y = dataset[:,25]
        return X,Y
    

    def scale(self,X):
        #Method to scale down data
        #@params X
        #@return X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float64))

        return X_scaled


    def randomforest(self, iterations = 15, leaf_iter = 12):
        #Method to train a random forest on ideal parameters with scaled and non-scaled data
        #@params iterations = for number of trees, leaf_iter = iterations for number of leaf noes
        #@return None
        y_train = self.y
        for run in range(0,2):
            if run == 1:
                X_train = self.scale(self.X)
                self.X_test = self.scale(self.X_test)
            else:
                X_train = self.X
            overall_r = []
            for e in range(7,iterations+1):

                for j in range(1,leaf_iter):
                    leafnodes = 24*(2*j)
                    classifier = RandomForestClassifier(n_estimators = 10 * e, max_leaf_nodes = leafnodes, criterion = 'entropy', random_state = 42)
                    classifier.fit(X_train, y_train)

                    y_pred = classifier.predict(self.X_test)
                    
                    #add each prediction
                    r_pred = []
                    for i in y_pred.tolist():
                        r_pred.append(i)
                    overall_r.append(r_pred)    

            #run this only the first time so the dictionaries are initialised
            if run == 0:
                for i,pred_list in enumerate(overall_r):
                    for j,each in enumerate(pred_list):
                        self.predictions_r[j] = {}

            for i,pred_list in enumerate(overall_r):
                for j,each in enumerate(pred_list):
                    self.predictions_r[j][each] = self.predictions_r[j].get(each,0)+1

    def post_prep_rf(self):
        #Method to prepare the results with the most predicted label for each match chosen
        #@params None
        #@return gameweek predictions, accuracies

        #grab the best prediction
        prediction = []
        second_pred = []
        import operator
        for key,values in self.predictions_r.items():
            prediction.append(max(values.items(), key=operator.itemgetter(1))[0])
            del values[(max(values.items(), key=operator.itemgetter(1))[0])]

        i = 0
        for key,values in self.predictions_r.items():
            try:
                second_pred.append(max(values.items(), key=operator.itemgetter(1))[0])
            except ValueError:
                second_pred.append(prediction[i])
            i += 1

        gameweek_forPM_df = pd.DataFrame()
        result_score = {1.0:[3,0], 2.0:[2,1], 3.0:[1,0], 4.0:[2,2], 5.0:[1,1], 6.0:[0,0], 7.0:[0,1], 8.0:[1,2], 9.0:[0,3]}
        gameweek_forPM_df["team H"] = self.gameweek_static["Team A"]
        gameweek_forPM_df["team A"] = self.gameweek_static["Team B"]
        i = 0
        for index, row in gameweek_forPM_df.iterrows():
            gameweek_forPM_df.at[index,"team_h_score"] = result_score[prediction[i]][0]
            gameweek_forPM_df.at[index,"team_a_score"] = result_score[prediction[i]][1]
            i+=1
        
        accuracies =[]
        full_acc= 0
        second_acc = 0
        no_hit = 0
        for i,value in enumerate(prediction):
            if value == self.y_test[i]:
                full_acc += 1
            elif second_pred[i] == self.y_test[i]:
                second_acc += 1
            else:
                no_hit+=1
        accuracies.append([full_acc/len(prediction),second_acc/len(prediction),no_hit/len(prediction)])
                
        #return the gameweek for Player Model, and the accuracies of labels from the Fixture prediction matches.
        return gameweek_forPM_df,accuracies

