#!/usr/bin/env python
# coding: utf-8

# ### Hill Climber algorithm to generate a team
# 
# - takes in the list of available players for the gameweek with their predicted scores from the player model
# - constructs a 15-man squad adhering to budget connstraints and team constraints set by FPL

# In[ ]:


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


# In[78]:


class generate_team():
    
    def __init__(self, players):
        self.players = players
        self.initiate()
    
    def init_player_lists(self,players):
        #Method to initialise four vectors of 0, where each defines the position and is the length of the amount of
        #players in that position. Randomly assign 1's to an index in each list to initiate a squad
        #@params players = the list of available players
        #@return goalkeeper, defender, midfielder, attacker lists
        
        #make the lists from the actual players
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
        
        #initialise each vector to have a random player, goalkeepers 2 players, def 5, mid 5, atk 3.
        goalkeepers = [0 for i in range(len(goalks))]
        x = [i for i in range(len(goalks))]
        random_numbers = random.sample(x, 2)
        for random_number in random_numbers:
            goalkeepers[random_number] = 1

        defenders = [0 for i in range(len(defs))]
        x = [i for i in range(len(defs))]
        random_numbers = random.sample(x, 5)
        for random_number in random_numbers:
            defenders[random_number] = 1

        midfielders = [0 for i in range(len(mids))]
        x = [i for i in range(len(mids))]
        random_numbers = random.sample(x, 5)
        for random_number in random_numbers:
            midfielders[random_number] = 1

        attackers = [0 for i in range(len(atts))]
        x = [i for i in range(len(atts))]
        random_numbers = random.sample(x, 3)
        for random_number in random_numbers:
            attackers[random_number] = 1

        return goalkeepers, defenders, midfielders ,attackers
    
    def init_hc(self,max_threshold, players):
        #Method to call initialise squad vectors. If the constraints are not met, repeat the initialisation
        #@params max_threshold = the budget constraint, players = the available players
        #@return goalkeeper, defender, midfielder, attacker lists
        
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

        total_weight = max_threshold +1
        total_points = 51
        #total_weight cannot go over budget constraint
        while (total_weight > 100 or total_points < 50):
            gkp, defs, mids, atks = self.init_player_lists(players)
            g_weight = (sum([goalkeeper[i][4] for i in range(len(goalkeeper)) if gkp[i] == 1]))
            d_weight = (sum([defender[i][4] for i in range(len(defender)) if defs[i] == 1]))
            m_weight = (sum([midfielder[i][4] for i in range(len(midfielder)) if mids[i] == 1]))
            a_weight = (sum([attacker[i][4] for i in range(len(attacker)) if atks[i] == 1]))
            total_weight = g_weight + d_weight + m_weight + a_weight 

            team_points = []
            team_points.append([goalkeeper[i][3] for i in range(len(goalkeeper)) if gkp[i] == 1])
            team_points.append([defender[i][3] for i in range(len(defender)) if defs[i] == 1])
            team_points.append([midfielder[i][3] for i in range(len(midfielder)) if mids[i] == 1]) 
            team_points.append([attacker[i][3] for i in range(len(attacker)) if atks[i] == 1])
            team_points = list(itertools.chain(*team_points))
            total_points = sum(team_points)
        return gkp, defs, mids, atks

    def evaluate_fitness(self,max_threshold, players, gkp, defs, mids, atks):
        #Method to evaluate fitness by looking at the weight of the squad. Similarly, to see if any team has
        #more than 3 players in the squad. This is not allowed in FPL rules
        #@params max-threshold = budget, players = available players, gkp,defs,mids,atks = squad player vectors
        #@return fitness = how many projected points, team_weight = cost of squad, team_points = indiv player points
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
                    if i > 3:
                        #overkill = too many players from one team
                        overkill_team = True
        
        #If team weight is too high or overkill is true, fitness of the squad is 0
        if sum(team_weight) > max_threshold or overkill_team is True:
            fitness = 0
        else:
            fitness = sum(team_points)
        return fitness, team_weight, team_points
    
    def mutate(self,mutation_rate, gkp, defs, mids, atks):
        #Method to mutate the squad in a certain way to find new better squads
        #@params mutation_rate = the likelihood of change, gkp,def,mids,atks = squad vectors
        #@return gkp,defs,mids,atks = new mutated squad
    
        a = rd.random()
        b = rd.random()
        c = rd.random()
        d = rd.random()
        
        #For each position, get a random number, swap out a current player and change a position in the vector
        #from zero to one to add a new player

        if a <= mutation_rate:
            x = [i for i in range(len(gkp))]
            random_number = random.sample(x, 1)
            present_numbers = [i for i in range(len(gkp))if gkp[i] == 1]
            #if player is already in the squad, dont do anything, if not carry on
            if random_number[0] not in present_numbers:
                #choose which player to remove randomly out of 2 players
                r = [i for i in range(0,2)]
                #remove the player
                remove_index = random.sample(r,1)
                for player in range(len(gkp)):
                    if player == present_numbers[remove_index[0]]:
                        gkp[player] = 0
                    if player == random_number[0]:
                        gkp[player] = 1

        if b <= mutation_rate:
            x = [i for i in range(len(defs))]
            random_number = random.sample(x, 1)
            present_numbers = [i for i in range(len(defs))if defs[i] == 1]
            #if player is already in the squad, dont do anything, if not carry on
            if random_number[0] not in present_numbers:
                #choose which player to remove randomly out of 5 players
                r = [i for i in range(0,5)]
                #remove the player
                remove_index = random.sample(r,1)
                for player in range(len(defs)):
                    if player == present_numbers[remove_index[0]]:
                        defs[player] = 0
                    if player == random_number[0]:
                        defs[player] = 1

        if c <= mutation_rate:
            x = [i for i in range(len(mids))]
            random_number = random.sample(x, 1)
            present_numbers = [i for i in range(len(mids))if mids[i] == 1]
            #if player is already in the squad, dont do anything, if not carry on
            if random_number[0] not in present_numbers:
                r = [i for i in range(0,5)]
                #choose which player to remove randomly out of 5 players
                remove_index = random.sample(r,1)
                #remove the player
                for player in range(len(mids)):
                    if player == present_numbers[remove_index[0]]:
                        mids[player] = 0
                    if player == random_number[0]:
                        mids[player] =1

        if d <= mutation_rate:
            x = [i for i in range(len(atks))]
            random_number = random.sample(x, 1)
            present_numbers = [i for i in range(len(atks))if atks[i] == 1]
            #if player is already in the squad, dont do anything, if not carry on
            if random_number[0] not in present_numbers:
                #choose which player to remove randomly out of 3 players
                r = [i for i in range(0,3)]
                #remove the player
                remove_index = random.sample(r,1)
                for player in range(len(atks)):
                    if player == present_numbers[remove_index[0]]:
                        atks[player] = 0
                    if player == random_number[0]:
                        atks[player] = 1

        return gkp, defs, mids, atks

    def hillclimber(self,gkp, defs, mids, atks, players, max_threshold, generations, mutation_rate):
        #Method to call hill climber algorithm with each individual method, used to rretrieve new better squads
        #@params gkp,defs,mids,atks = player vectors, max_threshold = budget, generations, mutation_rate
        #@return gkp,defs,mids,atks = player vectors, fitness = list of squad projected points over time
    
        fitness = []
        for i in range(generations):
            #if (i % 50000) == 0:
                #print(i/50000, "% completed")
            
            #make a copy of the exisiting squad
            gkp1 = gkp.copy()
            defs1 = defs.copy()
            mids1 = mids.copy()
            atks1 = atks.copy()

            #mutate copied squad, calc fitness of both new and old squads
            gkp1, defs1, mids1, atks1 = self.mutate(mutation_rate, gkp1, defs1, mids1, atks1)
            g0_fitness, team_weight, team_points = self.evaluate_fitness(max_threshold, players, gkp, defs, mids, atks)
            g1_fitness, team_weight, team_points = self.evaluate_fitness(max_threshold, players, gkp1, defs1, mids1, atks1)

            #if new squad has better fitness, this becomes the main squad
            if g1_fitness > g0_fitness:
                gkp = gkp1
                mids = mids1
                defs = defs1
                atks = atks1

            fitness.append(g0_fitness)

        return gkp, defs, mids, atks, fitness
    
    def graph(self,fitness,mutation_rate, players, gkp, defs, mids, atks,generations):
        #Method to call a visual representation of the graph
        #@params fitness = squad fitness, mutation_rate, players = available players, squad vectors, generations
        #@return total team points and total team weight
        
        #print("Mutation rate = ", mutation_rate)
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

        #print("Total final points:", sum(team_points))
        #print("Total final cost:", sum(team_weight))

        #plt.plot(range(generations), fitness, label = 'Fitness')
        #plt.legend()
        #plt.title('Fitness level of hill Climber over 500 generations')
        #plt.xlabel('Generations')
        #plt.ylabel('Fitness')
        #plt.show()

        return (sum(team_points)), (sum(team_weight))
    
    def project_squad(self,players, top_choice):
        #Method to get the best squad vector and change it from the labels back into the real projected points
        #@params players = available players, top_choice = top squad vectors
        #@return None

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

        #amend dataframe to change labels into actual predicted FPL poinnts
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
            elif po >= 14:
                full_squad.at[index,"Xg_Points"]  = str(po+2)
        del full_squad["Predicted_Points"]


        return full_squad, squad
    
    def initiate(self):
        #Method to initiate the hillcimber 
        #@params None
        #@return None
        
        max_threshold = 100
        generations = 10000
        #mutation rates chosen
        mutation_rates =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        iterations = 3

        fit_group = []
        totalp = [0]
        totalc = [0] 
        final_choices = []
        #make a new squad
        gkp, defs, mids, atks = self.init_hc(max_threshold, self.players)
        for i in range(iterations):
            for mutation_rate in mutation_rates:
                
                final_choices = []
                #make a copy of the init hill climber so each iteration has the same initial starting squad
                gkp1 = gkp
                mids1 = mids
                defs1 = defs
                atks1 = atks
                fitness_history = []
                #call the hillclimber on the squad
                gkp1, defs1, mids1, atks1, fitness = self.hillclimber(gkp1, defs1, mids1, atks1, self.players, max_threshold, generations, mutation_rate)
                fit_group.append(fitness)
                final_points, final_cost = self.graph(fitness,mutation_rate, self.players, gkp1, defs1, mids1, atks1,generations)
                final_choices.append([gkp1, defs1, mids1, atks1])
                #if squad is optimal, append to optimal squad
                if final_points >= max(totalp):
                    if final_points == max(totalp) and final_cost < max(totalc):
                        continue
                    else:
                        top_choice  = [gkp1,defs1,mids1,atks1]
                        top_cost = final_cost
                        top_points = final_points
                totalp.append(final_points) 
                totalc.append(final_cost)
        full_squad, squad = self.project_squad(self.players,top_choice)
        self.full_squad = full_squad
        self.squad = squad
        #top choice is the vector representation of the best squad, used for transfer model
        self.top_choice = top_choice
        print("Squad Selection is Completed")


# In[ ]:




