#!/usr/bin/env python
# coding: utf-8

# ### Get all available players
# 
# - single out players which are not likely to play that gameweek
# - remove anyone with a chance of playing below 50%
# - remove anyone with predicted points less than 2

# In[11]:


import numpy as np
import pandas as pd
import random as rd
import random
from random import randint
from random import sample
import matplotlib.pyplot as plt
import requests
pd.options.display.max_rows = None
pd.options.display.max_columns = None
import itertools


# In[12]:


class get_able_players():

    def __init__(self, player_predictions):
        self.player_predictions = player_predictions
        self.retrieve_players()
        self.add_cost_chances()
        self.select_able_players()
        self.organise_tuples()

    def retrieve_players(self):
        #Method to retrieve the live data of player status
        #@params None
        #@return None
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        r = requests.get(url)
        json = r.json()
        stats_df = pd.DataFrame(json['element_stats'])
        self.elements_df = pd.DataFrame(json['elements'])

    def add_cost_chances(self):
        #Method to add the cost and chances of playing to the dataframe for each player
        #@params None
        #@return None
        player_predictions = self.player_predictions
        for index,row in player_predictions.iterrows():
            name = player_predictions["web_name"][index]
            for _index,_row in self.elements_df.iterrows():
                second_name = self.elements_df["web_name"][_index]
                if name == second_name:
                    player_predictions.at[index, "now_cost"] = self.elements_df["now_cost"][_index]
                    player_predictions.at[index, "chances"] = self.elements_df["chance_of_playing_next_round"][_index]
                    player_predictions.at[index, "exp_points"] = self.elements_df["ep_next"][_index]
        self.player_predictions = player_predictions

    def select_able_players(self):
        #Method to remove players that have little chance of playing or are predicted to have less than 2 points
        #@params None
        #@return None
        player_predictions = self.player_predictions
        player_predictions = player_predictions[player_predictions["chances"]!= 0.0]
        player_predictions = player_predictions[player_predictions["chances"]!= 25.0]
        player_predictions = player_predictions[player_predictions["chances"]!= 50.0]
        for index,row in player_predictions.iterrows():
            player_predictions.at[index, "exp_points"] = float(player_predictions.at[index, "exp_points"])
        player_predictions = player_predictions[player_predictions["exp_points"] > 2.0]
        player_predictions["now_cost"] = (player_predictions["now_cost"] / 10)
        self.player_predictions = player_predictions

    def organise_tuples(self):
        #Method to organise tuples for hill climber algorithm, change from dataframe into tuple
        #@params None
        #@return None
        players = []
        player_predictions = self.player_predictions
        for index,row in player_predictions.iterrows():
            name = player_predictions["web_name"][index]
            element = player_predictions["element_type"][index]
            team = player_predictions["team"][index]
            predicted_points = player_predictions["predicted_points"][index]
            now_cost = player_predictions["now_cost"][index]
            players.append([name,element,team,predicted_points,now_cost])
        self.players = players


# In[ ]:




