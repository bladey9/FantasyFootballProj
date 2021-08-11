#!/usr/bin/env python
# coding: utf-8

# ### Call the whole model
# 
# - gameweek int decides when the squad is built

# In[10]:


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


# In[11]:


from classes.train_fixture_model import *
from classes.train_player_model import *
from classes.get_player_dataset import *
from classes.able_players import * 
from classes.generate_team import *
from classes.generate_transfer import *


# In[4]:


gameweek = 32
#read the training data
training_data = pd.read_csv(r'training_data.csv')
#train and predict on one gameweek, gameweek - 1 is what data will be used to train on
gameweek_fixture_pred = trainrf(training_data,i)
#get gameweek fixtures
gameweek_fixture_pred,acc = gameweek_fixture_pred.post_prep_rf()
#make player dataset with fixture prediction
players = player(gameweek_fixture_pred,i)
#calc player poinnts
player_points = predict_player(players.player_results_df, players.dataset)
pl_pr_po = player_points.player_predictions
#get able players
players = get_able_players(pl_pr_po)
#generate a team
squad = generate_team(players.players)

#show full squad
squad.full_squad

#for transfer model, call the following function
#new_squad = generate_transfer(squad.top_choice, players.players)
#new_squad.full_squad


# In[ ]:




