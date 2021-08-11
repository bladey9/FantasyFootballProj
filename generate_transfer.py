#!/usr/bin/env python
# coding: utf-8

# ### Hill Climber for transfer model
# 
# - get the squad and make one change to the squad for the next gameweek
# - return the best squad

# In[2]:


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


# In[2]:


class generate_transfer():
    
    def __init__(self, top_choice,able_players):
        self.top_choice = top_choice
        self.players = able_players
        self.initiate_transfer()
        
    def init_hc_transfer(self,top_choice):
        #Method to retrieve the actual squad and amend the squad into the four position vectors
        #@params top_choice = current squad
        #@return gkp,defs,mids,atks = squad vectors

        gkp = top_choice[0]
        defs = top_choice[1]
        mids = top_choice[2]
        atks = top_choice[3]

        return gkp, defs, mids, atks

    def evaluate_fitness(self,max_threshold, players, gkp, defs, mids, atks):
        #Method to evaluate the fintness of a give squad
        #@params max_threshold = budget, players = available players, squad vectors
        #@return fitness = squad total projected points, team weight = team total cost, team_points = indv points
        
        team_weight= []
        team_points = []
        fitness = 0

        goalkeeper, defender, midfielder, attacker = [], [], [], []
        for i in players:
            if i[1] == 1:
                goalkeeper.append(i)
            elif i[1] == 2:
                defender.append(i)
            elif i[1] == 3:
                midfielder.append(i)
            elif i[1] == 4:
                attacker.append(i)

        #calculate weight of squad (total cost)
        team_weight.append([goalkeeper[i][4] for i in range(len(goalkeeper)) if gkp[i] == 1])
        team_weight.append([defender[i][4] for i in range(len(defender)) if defs[i] == 1])
        team_weight.append([midfielder[i][4] for i in range(len(midfielder)) if mids[i] == 1]) 
        team_weight.append([attacker[i][4] for i in range(len(attacker)) if atks[i] == 1])
        team_weight = list(itertools.chain(*team_weight))
        
        #calculate total team points predicted 
        team_points.append([goalkeeper[i][3] for i in range(len(goalkeeper)) if gkp[i] == 1])
        team_points.append([defender[i][3] for i in range(len(defender)) if defs[i] == 1])
        team_points.append([midfielder[i][3] for i in range(len(midfielder)) if mids[i] == 1]) 
        team_points.append([attacker[i][3] for i in range(len(attacker)) if atks[i] == 1])
        team_points = list(itertools.chain(*team_points))

        #calculate how many players are from each team to see if gone over
        team_code = []
        team_code.append([goalkeeper[i][2] for i in range(len(goalkeeper)) if gkp[i] == 1])
        team_code.append([defender[i][2] for i in range(len(defender)) if defs[i] == 1])
        team_code.append([midfielder[i][2] for i in range(len(midfielder)) if mids[i] == 1]) 
        team_code.append([attacker[i][2] for i in range(len(attacker)) if atks[i] == 1])
        team_code = list(itertools.chain(*team_code))
        overkill_team = False
        for team in range(1,21):
            i = 0
            for team_ in team_code:
                if team == team_:
                    i+= 1
                    #overkill = too many players from one team
                    if i > 3:
                        overkill_team = True
        #If team weight is too high or overkill is true, fitness of the squad is 0
        if sum(team_weight) > max_threshold or overkill_team is True:
            fitness = 0
        else:
            fitness = sum(team_points)
        return fitness, team_weight, team_points


    def mutate_transfer(self,mutation_rate, players, gkp, defs, mids, atks):
        #Method to change one member of the squad given the one transfer rule
        #@params mutation_rate = the likelihood of change, players = avail players, gkp,def,mids,atks = squad vectors
        #@return squad vectors, rpi,api = remove player index and added player index
        
        #rpi= removed player index, #api = added player index
        rpi, api = "no transfer found", "no transfer found"
        goalkeeper, defender, midfielder, attacker = [], [], [], []
        for i in players:
            if i[1] == 1:
                goalkeeper.append(i)
            elif i[1] == 2:
                defender.append(i)
            elif i[1] == 3:
                midfielder.append(i)
            elif i[1] == 4:
                attacker.append(i)

        r = rd.random()
        #see which list we will randomly change of all the four positionns
        if r <= 0.1:
            position = gkp
            lp = goalkeeper
        elif r <= 0.45:
            position = defs
            lp = defender
        elif r <= 0.8:
            position = mids
            lp = midfielder
        elif r <= 1:
            position = atks
            lp = attacker

        x = [i for i in range(len(position))]
        random_number = random.sample(x,1)
        present_numbers = [i for i in range(len(position)) if position[i] ==1]
        #if player is already in the squad, dont do anything, if not carry on
        if random_number[0] not in present_numbers:
            r = [i for i in range(0,len(present_numbers))]
            #choose which player to remove randomly
            remove_index = random.sample(r,1)
            for player in range(len(position)):
                if player == present_numbers[remove_index[0]]:
                    position[player] = 0
                    #remove the player and append removed player
                    rpi = lp[player][0]
                if player == random_number[0]:
                    position[player] = 1
                    #add the player and append added player
                    api = lp[player][0]

        return gkp,defs,mids,atks, rpi,api

    def hillclimber(self,gkp, defs, mids, atks, players, max_threshold, generations, mutation_rate):
        #Method to call hillclimber on the squad
        #@params squad vectors, max_threshold = budget, generations, mutation_rate
        #@return new squad, removed and added player index
        fitness = []
        different = False
        i = 0
        print("running")
        while (different == False and i < generations):
                #if (i % 50000) == 0:
                    #print(i/50000, "% completed")
                
                #make a copy of the squad
                gkp1 = gkp.copy()
                defs1 = defs.copy()
                mids1 = mids.copy()
                atks1 = atks.copy()
                
                #mutate copied squad, calc fitness of both new and old squads
                gkp1, defs1, mids1, atks1, rpi,api = self.mutate_transfer(mutation_rate, players, gkp1, defs1, mids1, atks1)
                g0_fitness, team_weight, team_points = self.evaluate_fitness(max_threshold, players, gkp, defs, mids, atks)
                g1_fitness, team_weight, team_points = self.evaluate_fitness(max_threshold, players, gkp1, defs1, mids1, atks1)
                
                #if new squad has better fitness, this becomes the main squad
                if g1_fitness > g0_fitness:
                    print("fitness levels new vs old", g1_fitness, g0_fitness)
                    gkp = gkp1
                    mids = mids1
                    defs = defs1
                    atks = atks1
                    different = True

                fitness.append(g0_fitness)
                i+=1
        #if no transfer found, rpi and api = 0
        if different == False:
            rpi = 0
            api = 0
        else:
            print("old")
        return gkp, defs, mids, atks, fitness, rpi, api
    
    def graph(self,fitness,mutation_rate, players, gkp, defs, mids, atks):
        #Method to portray the hill climber in a graph
        #@params fitness = squad fitness, mutation_rate, players = available players, squad vectors, generations
        #@return total team points and total team weight
        team_weight= []
        team_points = []

        goalkeeper, defender, midfielder, attacker = [], [], [], []
        for i in players:
            if i[1] == 1:
                goalkeeper.append(i)
            elif i[1] == 2:
                defender.append(i)
            elif i[1] == 3:
                midfielder.append(i)
            elif i[1] == 4:
                attacker.append(i)

        team_weight.append([goalkeeper[i][4] for i in range(len(goalkeeper)) if gkp[i] == 1])
        team_weight.append([defender[i][4] for i in range(len(defender)) if defs[i] == 1])
        team_weight.append([midfielder[i][4] for i in range(len(midfielder)) if mids[i] == 1]) 
        team_weight.append([attacker[i][4] for i in range(len(attacker)) if atks[i] == 1])
        team_weight = list(itertools.chain(*team_weight))

        team_points.append([goalkeeper[i][3] for i in range(len(goalkeeper)) if gkp[i] == 1])
        team_points.append([defender[i][3] for i in range(len(defender)) if defs[i] == 1])
        team_points.append([midfielder[i][3] for i in range(len(midfielder)) if mids[i] == 1]) 
        team_points.append([attacker[i][3] for i in range(len(attacker)) if atks[i] == 1])
        team_points = list(itertools.chain(*team_points))

        return (sum(team_points)), (sum(team_weight))
    
    def project_squad(self,players, top_choice, rpi, api):
        #Method to get the best squad vector and change it from the labels back into the real projected points
        #@params players = available players, top_choice = top squad vectors, api,rpi = added and removed player indexes
        #@return full_squad in dataframe, and squad in list form
        goalks, defs, mids, atts = [], [], [], []
        for i in players:
            if i[1] == 1:
                goalks.append(i)
            elif i[1] == 2:
                defs.append(i)
            elif i[1] == 3:
                mids.append(i)
            elif i[1] == 4:
                atts.append(i)

        squad = []
        for i in range(len(top_choice[0])):
            if top_choice[0][i] == 1:
                squad.append(goalks[i])
        for i in range(len(top_choice[1])):
            if top_choice[1][i] == 1:
                squad.append(defs[i])
        for i in range(len(top_choice[2])):
            if top_choice[2][i] == 1:
                squad.append(mids[i])
        for i in range(len(top_choice[3])):
            if top_choice[3][i] == 1:
                squad.append(atts[i])

        #return projected points from labels to the actual projected points
        full_squad = pd.DataFrame(squad, columns= ["Player Name", "Element Type", "Team", "Predicted_Points", "Cost"])
        full_squad.insert(loc = 3, column = "Xg_Points", value = "0")
        for index,row in full_squad.iterrows():
            po = full_squad["Predicted_Points"][index]
            if po == 1:
                full_squad.at[index,"Xg_Points"] = "-1"
            elif po == 2:
                full_squad.at[index,"Xg_Points"]  =  "0"
            elif po == 3:
                full_squad.at[index,"Xg_Points"]  = "1"
            elif po == 4:
                full_squad.at[index,"Xg_Points"]  = "2"
            elif po == 5:
                full_squad.at[index,"Xg_Points"]  = "3"
            elif po == 6:
                full_squad.at[index,"Xg_Points"] = "4"
            elif po == 7:
                full_squad.at[index,"Xg_Points"]  = "5"
            elif po == 8:
                full_squad.at[index,"Xg_Points"]  = "6"
            elif po == 9:
                full_squad.at[index,"Xg_Points"] = "7"
            elif po == 10:
                full_squad.at[index,"Xg_Points"]  = "8 - 9" 
            elif po == 11:
                full_squad.at[index,"Xg_Points"]  = "10 - 12" 
            elif po == 12:
                full_squad.at[index,"Xg_Points"]  = "12 - 14"
            elif po == 13:
                full_squad.at[index,"Xg_Points"]  = "15+"
        del full_squad["Predicted_Points"]

        return full_squad, squad
    
    
    def initiate_transfer(self):
        #Method to innitiate the hill climber transfer model
        #@params None
        #@return None    
        
        #init the vectors with the given squad
        gkp, defs, mids, atks = self.init_hc_transfer(self.top_choice)

        max_threshold = 100
        generations = 10000
        #mutation rates chosen
        mutation_rates =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        iterations = 1

        fit_group = []
        totalp = [0]
        final_choices = []
        removed_pi = 0
        added_pi = 0
        for i in range(iterations):
            for mutation_rate in mutation_rates:
                
                    final_choices = []
                    gkp1 = gkp
                    mids1 = mids
                    defs1 = defs
                    atks1 = atks
                    fitness_history = []
                    #call the hill climber on the squad
                    gkp1, defs1, mids1, atks1, fitness, rpi, api = self.hillclimber(gkp1, defs1, mids1, atks1, self.players, max_threshold, generations, mutation_rate)
                    fit_group.append(fitness)
                    final_points, final_cost = self.graph(fitness,mutation_rate,self.players, gkp1, defs1, mids1, atks1)
                    final_choices.append([gkp1, defs1, mids1, atks1])
                    #if fitness is the best fitness achieved, make the new squad the current squad
                    if final_points > max(totalp):
                        top_transfer_team  = [gkp1,defs1,mids1,atks1]
                        top_cost = final_cost
                        top_points = final_points
                        removed_pi = rpi
                        added_pi = api
                    totalp.append(final_points)
        #print("Total final points:", top_points)
        #print("Total final cost:", top_cost)
        full_squad, squad = self.project_squad(self.players,top_transfer_team, removed_pi, added_pi)
        self.full_squad = full_squad
        self.squad = squad
        self.removed_pi = removed_pi
        self.added_pi = added_pi
        #if we did find a good transfer, call the change() method to initiate the change
        if removed_pi != 0:
            self.change()      
    
    def change(self):
        #Method to initiate the change of squads
        #@params None
        #@return None  
        for player in self.players:
            if self.added_pi == player[0]:
                player_to_be = player
        new_squad = self.full_squad.copy()
        #make a dataframe for the new squad
        for index,row in new_squad.iterrows():
            name = new_squad["Player Name"][index]
            if name == self.removed_pi:
                new_squad.at[index, "Player Name"] = player_to_be[0]
                new_squad.at[index, "Element Type"] = player_to_be[1]
                new_squad.at[index, "Team"] = player_to_be[2]
                new_squad.at[index, "Xg_Points"] = player_to_be[3]
                new_squad.at[index, "Cost"] = player_to_be[4]
        print("Removed Player into Squad: ", self.removed_pi, " Added Player into Squad: ", self_added_pi)
                
        self.full_squad = new_squad

