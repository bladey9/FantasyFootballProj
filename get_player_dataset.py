#!/usr/bin/env python
# coding: utf-8

# ### Create player training dataset for player model
# 
# - Get all able players, loop through their history, append game features of those games
# - Recieve the predicted scoreline to make dataset ready for prediction 

# In[1]:


import numpy as np
import pandas as pd
pd.options.display.max_rows = None
import requests
pd.set_option('display.max_columns', None)


# In[91]:


class player():
    
    def __init__(self,data,gameweek_index = 39):
        #get gameweek index to see which gameweek should be used for prediction and up until what gameweek for training
        self.results = data
        self.gameweek_index = gameweek_index -1
        self.train()
        
    def train(self):
        self.get_able_players()
        self.make_player_dataset()
        self.organise()
        self.receive_scoreline()
        self.label_data()
    
    def get_able_players(self):
        # Get the players which are avaialble for selection
        # @param none 
        # @return none

        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        r = requests.get(url)
        json = r.json()
        stats_df = pd.DataFrame(json['element_stats'])

        elements_df = pd.DataFrame(json['elements'])
        del elements_df["chance_of_playing_next_round"]
        del elements_df["chance_of_playing_this_round"]
        del elements_df["cost_change_event"]
        del elements_df["cost_change_event_fall"]
        del elements_df["cost_change_start"]
        del elements_df["cost_change_start_fall"]
        del elements_df["dreamteam_count"]
        del elements_df["transfers_in"]
        del elements_df["transfers_in_event"]
        del elements_df["transfers_out"]
        del elements_df["transfers_out_event"]
        del elements_df["photo"]
        del elements_df["in_dreamteam"]
        del elements_df["news"]
        del elements_df["form"]
        del elements_df["code"]
        del elements_df["first_name"]
        del elements_df["news_added"]
        del elements_df["special"]
        del elements_df["corners_and_indirect_freekicks_order"]
        del elements_df["corners_and_indirect_freekicks_text"]
        del elements_df["direct_freekicks_order"]
        del elements_df["direct_freekicks_text"]
        del elements_df["penalties_order"]
        del elements_df["penalties_text"]
        del elements_df["ict_index_rank_type"]
        del elements_df["ict_index_rank"]
        del elements_df["creativity_rank_type"]
        del elements_df["creativity_rank"]
        del elements_df["influence_rank"]
        del elements_df["influence_rank_type"]
        del elements_df["threat_rank"]
        del elements_df["threat_rank_type"]
        del elements_df["value_season"]


        all_players_df = elements_df
        self.all_players_df = all_players_df[all_players_df.status != "u"]
    
    def make_player_dataset(self):
        #Method to make the player training dataset - loop through each players history annd append features
        #@params None
        #@return None
        dataset = self.all_players_df
        dataset["goals_per_90"] = 0.0
        dataset["assists_per_90"] = 0.0
        dataset["clean_sheets_per_90"] = 0.0
        dataset["goals_conceded_per_90"] = 0.0
        dataset["points_pg"] = 0.0
        dataset["minutes_last3"] = 0.0
        dataset["bonus_last5"] = 0.0
        train_df = pd.DataFrame()
        dataset = dataset[dataset["minutes"] != 0]
        for index,row in dataset.iterrows():

            #grab the player information of each player in the dataset
            player_id = dataset["id"][index]
            url = 'https://fantasy.premierleague.com/api/element-summary/' + str(int(player_id)) + "/"
            r = requests.get(url)
            json = r.json()
            player_history_df = pd.DataFrame(json["history"])

            del player_history_df["fixture"]
            del player_history_df["opponent_team"]
            del player_history_df["kickoff_time"]
            del player_history_df["round"]
            del player_history_df["own_goals"]
            del player_history_df["penalties_saved"]
            del player_history_df["penalties_missed"]
            del player_history_df["yellow_cards"]
            del player_history_df["red_cards"]
            del player_history_df["saves"]
            del player_history_df["value"]
            del player_history_df["bps"]
            del player_history_df["transfers_balance"]
            del player_history_df["selected"]
            del player_history_df["transfers_in"]
            del player_history_df["transfers_out"]
            del player_history_df["element"]
            player_history_df["points_gained"] = 0
            player_history_df.insert(loc=0, column='element_type', value=0)

            goals, assists, clean_sheets, minutes, points, goals_conceded= 0,0,0,0,0,0
            point_form = []
            split = self.gameweek_index
            for _index,_row in player_history_df[:split].iterrows():
                goals_x = player_history_df["goals_scored"][_index]
                goals += goals_x
                assists_x = player_history_df["assists"][_index]
                assists += assists_x
                clean_sheets_x = player_history_df["clean_sheets"][_index]
                clean_sheets += clean_sheets_x
                goals_conceded_x= player_history_df["goals_conceded"][_index]
                goals_conceded += goals_conceded_x
                minutes_x = player_history_df["minutes"][_index]
                minutes += minutes_x
                points_x = player_history_df["total_points"][_index]
                points += points_x
                point_form.append(points_x)

                player_history_df.at[_index,"points_pg"] = sum(point_form) / len(point_form)
                player_history_df.at[_index,"element_type"] = dataset["element_type"][index]
                player_history_df.at[_index,"points_gained"] = player_history_df["total_points"][_index]

            #loop through again given we have the full point form
            matches_index={}
            i = 0
            for _index, _row in player_history_df[:split].iterrows():
                if i >= 5:
                    player_history_df.at[_index, "pointslast1"] = point_form[i-1]
                    player_history_df.at[_index, "pointslast3"] = sum(point_form[i-3:i])
                    player_history_df.at[_index, "pointslast5"] = sum(point_form[i-5:i])
                i += 1

            del player_history_df["total_points"]

            #add the last recent form to the prediction model, to have the last match, last 3 and last 5 form
            dataset.at[index, "pointslast1"] = point_form[-1]
            dataset.at[index, "pointslast3"] = sum(point_form[-3:])
            dataset.at[index, "pointslast5"] = sum(point_form[-5:])
            train_df = train_df.append(player_history_df,ignore_index= True)

            #if no minutes played in a match, remove record

            if minutes > 0:
                dataset.at[index,"goals_per_90"] = (goals / minutes) * 90
                dataset.at[index,"assists_per_90"] = (assists / minutes) * 90
                dataset.at[index,"clean_sheets_per_90"] = (clean_sheets / minutes) * 90
                dataset.at[index,"goals_conceded_per_90"] = (goals_conceded / minutes) * 90
                dataset.at[index,"points_pg"] = (points/len(player_history_df))
            else:
                dataset = dataset.drop(index)


            #Calculate predicted minutes and bonus by averaging out the number of minutes in the last 5

            last_3_mins = 0
            last_5_bonus = 0
            for i,r in player_history_df[-3:split].iterrows():
                mins = player_history_df["minutes"][i]
                last_3_mins += mins
            for i,r in player_history_df[-5:split].iterrows():
                bonus = player_history_df["bonus"][i]
                last_5_bonus += bonus
            dataset.at[index, "minutes_last3"] = (last_3_mins / 3)
            dataset.at[index, "bonus_last5"] = (last_5_bonus / 5)

            #Calculate influence, creativity and threat per game by dividig by total No of Games

            influence = float(dataset["influence"][index])
            creativity = float(dataset["creativity"][index])
            threat = float(dataset["threat"][index])

            dataset.at[index, "influence"] = influence / len(player_history_df)
            dataset.at[index, "creativity"] = creativity / len(player_history_df)
            dataset.at[index, "threat"] = threat / len(player_history_df)

            player_history_df= player_history_df[player_history_df['minutes'] != 0]

        del dataset["ep_next"]
        del dataset["ep_this"]
        del dataset["event_points"]
        del dataset["id"]
        del dataset["second_name"]
        del dataset["points_per_game"]
        del dataset["selected_by_percent"]
        del dataset["squad_number"]
        del dataset["status"]
        del dataset["total_points"]
        del dataset["goals_scored"]
        del dataset["assists"]
        del dataset["clean_sheets"]
        del dataset["goals_conceded"]
        del dataset["own_goals"]
        del dataset["penalties_saved"]
        del dataset["penalties_missed"]
        del dataset["yellow_cards"]
        del dataset["red_cards"]
        del dataset["bonus"]
        del dataset["bps"]
        del dataset["saves"]
        del dataset["value_form"]
        del dataset["ict_index"]
        del dataset["minutes"]
        del dataset["now_cost"]
        del train_df["ict_index"]

        #change was_home column from boolean value to int value
        train_df["was_home"] = train_df["was_home"]*1

        #remove any columns with NaN values
        train_df = train_df.dropna()
        
        self.all_players_df = dataset
        self.train_df = train_df


    def organise(self):
        #Method to organise dataframes so they are the same for training and predicting
        #@params None
        #@return None
        self.all_players_df = self.all_players_df[["element_type", 
                           "team", 
                           "web_name", 
                           "minutes_last3", 
                           "goals_per_90",
                           "assists_per_90",
                           "clean_sheets_per_90",
                           "goals_conceded_per_90",
                           "bonus_last5",
                          "influence",
                          "creativity",
                          "threat",
                          "points_pg",
                          "pointslast1",
                          "pointslast3",
                          "pointslast5"]]

        self.train_df = self.train_df[['element_type',
                             'was_home',
                             'team_h_score',
                             'team_a_score',
                             'minutes',
                             'goals_scored',
                             'assists',
                             'clean_sheets',
                             'goals_conceded',
                             'bonus',
                             'influence',
                             'creativity',
                             'threat',
                             'points_pg',
                             'pointslast1',
                             'pointslast3',
                             'pointslast5',
                             'points_gained']]


    def receive_scoreline(self):
        #Method to append scoreline from Fixture Model to data
        #@params None
        #@return None
        dataset = self.all_players_df
        results = self.results
        teams = []
        dataset.insert(loc=1, column='team_a_score', value=0)
        dataset.insert(loc=1, column='team_h_score', value=0)
        dataset.insert(loc=1, column='was_home', value=0)
        for index,row in dataset.iterrows():
            team_code = dataset["team"][index]
            for _index, _row in results.iterrows():
                teamh = results["team H"][_index]
                teama = results["team A"][_index]
                if team_code == teamh:
                    dataset.at[index, "was_home"] = 1
                    dataset.at[index, "team_h_score"] = results["team_h_score"][_index]
                    dataset.at[index, "team_a_score"] = results["team_a_score"][_index]
                elif team_code == teama:
                    dataset.at[index, "was_home"] = 0
                    dataset.at[index, "team_h_score"] = results["team_h_score"][_index]
                    dataset.at[index, "team_a_score"] = results["team_a_score"][_index]
        for index,row in results.iterrows():
                teamh = results["team H"][index]
                teama = results["team A"][index]
                teams.append(teamh)
                teams.append(teama)
        missed =[]
        for i in range(1,21):
            if i not in teams:
                missed.append(i)
        #drop missed fixtures which didnt have a scoreline
        for each in missed:
            dataset= dataset[dataset.team != each]
        self.dataset = dataset
        self.dataset = self.dataset.dropna()
        
    def label_data(self):
    
        #1 = 0 points
        #2 = 1 points
        #3 = 2 points
        #4 = 3 points
        #5 = 4 points
        #6 = 5 points
        #7 = 6 points
        #8 = 7 points
        #9 = 8-9 points
        #10 = 10-11 points
        #11 = 12-14 points
        #12 = 15+ points

        player_results_df = self.train_df.copy()
        player_results_df["label"] = 0
        for index, row in player_results_df.iterrows():
            points = player_results_df["points_gained"][index]
            if points < 0:
                player_results_df.at[index,"label"] = 1
            elif points == 0:
                player_results_df.at[index,"label"] = 2
            elif points == 1:
                player_results_df.at[index,"label"] = 3
            elif points == 2:
                player_results_df.at[index,"label"] = 4
            elif points == 3:
                player_results_df.at[index,"label"] = 5
            elif points == 4:
                player_results_df.at[index,"label"] = 6
            elif points == 5:
                player_results_df.at[index,"label"] = 7
            elif points == 6:
                player_results_df.at[index,"label"] = 8
            elif points == 7:
                player_results_df.at[index,"label"] = 9 
            elif points == 8 or points == 9:
                player_results_df.at[index,"label"] = 10 
            elif points == 10 or points == 11:
                player_results_df.at[index,"label"] = 11 
            elif points == 12 or points == 13 or points == 14:
                player_results_df.at[index,"label"] = 12
            elif points > 12:
                player_results_df.at[index,"label"] = 13

        del player_results_df["points_gained"]
        self.player_results_df = player_results_df


# In[ ]:




