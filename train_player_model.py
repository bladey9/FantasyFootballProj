#!/usr/bin/env python
# coding: utf-8

# ### Train Player Model
# 
# - Train Player model with neural network
# - Predict projected player points for new gameweek

# In[1]:


import numpy as np
import pandas as pd
pd.options.display.max_rows = None
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
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# In[23]:


class predict_player():

    def __init__(self, dataset,gameweek):
        self.dataset = dataset
        self.gameweek = gameweek
        self.train()
        
        
    def train(self):
        self.neural_network()
        self.prepare_predictions()
        
    def prepare_data(self,dataset):
        #Method to prepare dataset in appropriate manner for training
        #@params dataset
        #@return X = data,Y = labels
        dataset.to_csv(r'playerMDL.csv',header = False, index = False)
        dataset = loadtxt(r'playerMDL.csv', delimiter=',')
        X = dataset[:,0:17].astype(float)
        Y = dataset[:,17]

        return X,Y

    def scale(self,X):
        #Method to scale X data
        #@params X
        #@return X_scaled
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float64))

        return X_scaled

    def encode(self,Y):
        #Method to one-hot encode vectors
        #@params Y Labels
        #@return Y_labels encoded
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)

        return dummy_y

    def match_pred_with_features(self,prediction_df):
        #Method to match predictions with features by directly changing in-game features to figures
        #@params prediction dataframe
        #@return prediction dataframe

        #Change in-game features of a player to match the scoreline, e.g if a scoreline is 4-4,
        #increase players chances of goals
        for index,row in prediction_df.iterrows():
            team_h_score = prediction_df["team_h_score"][index]
            team_a_score = prediction_df["team_a_score"][index]
            goalsp90 = prediction_df["goals_per_90"][index]
            assistsp90 = prediction_df["assists_per_90"][index]
            home = prediction_df["was_home"][index]
            if home == 1:
                prediction_df.at[index,"goals_conceded_per_90"] = team_a_score
            if home == 1 and team_a_score > 0:
                prediction_df.at[index,"clean_sheets_per_90"] = 0.25
            if home == 1 and team_h_score == 0:
                prediction_df.at[index,"goals_per_90"] = 0.25
                prediction_df.at[index,"assists_per_90"] = 0.25
            if home == 1 and team_a_score == 0:
                prediction_df.at[index,"clean_sheets_per_90"] = 0.75
            if home == 1 and team_h_score > 1:
                prediction_df.at[index,"goals_per_90"] = (goalsp90 * 2)
                prediction_df.at[index,"assists_per_90"] = (assistsp90 * 2)
            if home == 0:
                prediction_df.at[index,"goals_conceded_per_90"] = team_h_score
            if home == 0 and team_h_score > 0:
                prediction_df.at[index,"clean_sheets_per_90"] = 0.25
            if home == 0 and team_h_score == 0:
                prediction_df.at[index,"clean_sheets_per_90"] = 0.75
            if home == 0 and team_a_score == 0:
                prediction_df.at[index,"goals_per_90"] = 0.25
                prediction_df.at[index,"assists_per_90"] = 0.25
            if home == 0 and team_a_score > 1:
                prediction_df.at[index,"goals_per_90"] = (goalsp90 * 2)
                prediction_df.at[index,"assists_per_90"] = (assistsp90 * 2)

        return prediction_df

    def neural_network(self):
        #Method to train neural network on historical player data
        #@params None
        #@return None
        
        self.gameweek = self.match_pred_with_features(self.gameweek)
        X_train, Y_train = self.prepare_data(self.dataset)
        pred_df = self.gameweek.copy()
        
        del pred_df["team"]
        del pred_df["web_name"]
        pred_df.to_csv(r'playerMDL.csv',header = False, index = False)
        pred_df = loadtxt(r'playerMDL.csv', delimiter=',')

        X_test = pred_df[:,:].astype(float)
        X = self.scale(X_train)
        X_test = self.scale(X_test)
        
        Y = self.encode(Y_train)

        main_model = Sequential()
        main_model.add(Dense(50, input_dim=17, activation='relu'))
        main_model.add(Dense(50, input_dim=50, activation='relu'))
        main_model.add(Dense(50, input_dim=50, activation='relu'))
        main_model.add(Dense(25, input_dim=25, activation='relu'))
        main_model.add(Dense(13, activation='softmax'))

        main_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping_monitor = EarlyStopping(patience=10)
        main_model.fit(X, Y, epochs=200, validation_split=0.2)

        self.predictions = main_model.predict(X_test)
        
    def prepare_predictions(self):
        #Method to prepare predictions for hill climbing algorithm
        #@params None
        #@return None
        
        #place the projected points alongside each player
        labels = []
        for each in self.predictions:
            ind = np.argmax(each)
            labels.append(ind)
        player_predictions = pd.DataFrame()
        i = 0
        player_predictions.insert(0,"element_type", self.gameweek["element_type"])
        player_predictions.insert(1,"team", self.gameweek["team"])
        player_predictions.insert(2,"web_name", self.gameweek["web_name"])
        player_predictions["predicted_points"] = 0

        for index,row in player_predictions.iterrows():
            player_predictions.at[index,"predicted_points"] = labels[i]
            i+=1
            
        self.player_predictions = player_predictions

        
        

