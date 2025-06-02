import json 
import numpy as np

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score

def process_icc_dataset(dataset):
    preference_data = []
    matchday_dict = {}
    unique_matches = []
    team_dict = {0: 'South Africa', 1: 'Australia', 2: 'India', 3: 'New Zealand', 4: 'England', 5: 'Sri Lanka', 6: 'Bangladesh', 7: 'West Indies', 8: 'Pakistan'}
    for idx, team_matches in enumerate(dataset['Matches']):
        for mat in team_matches:
            matchday = ' '.join(mat.split(',')[0].split(' ')[1:]) + mat.split(',')[1]
            if matchday in matchday_dict.keys():
                continue
            unique_matches.append(mat)
            match_status = mat.split(' ')[0]
            opponent = None         
            if match_status == 'D':
                opponent = mat.split('vs')[-1].split('Match drawn')[0].strip()
            elif mat.replace(team_dict[idx], '_').count('_') == 2:
                opponent = mat.replace(team_dict[idx], '_').split('vs')[-1].split('_')[0].strip()
            else:
                if len(mat.split('vs')[-1].split('won')[0].strip().split(' ')) == 2:
                    opponent = mat.split('vs')[-1].split('won')[0].strip().split(' ')[0].strip()
                else:
                    opponent = ' '.join(mat.split('vs')[-1].split('won')[0].strip().split(' ')[:2])
            matchday_dict[matchday] = (team_dict[idx], opponent, match_status)
    
    for key, value in matchday_dict.items():
        preference_data.append(value)
        
    gold_rankings = dataset['Team'].tolist()
    team_identifier = {}

    for idx, team in enumerate(gold_rankings):
        team_identifier[team] = idx
            
    return preference_data, team_identifier, gold_rankings

def process_nfl_dataset(dataset, season):
    df = dataset[dataset['season'] == season]
    preference_data = []
    matchday_dict = {}
    unique_matches = []

    team_win_count = {}
    teams = set(df['home_team']).union(df['away_team'])

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        game_id = row['game_id']

        if home_score > away_score:
            result = 'W'
            winner, loser = home, away
        elif home_score < away_score:
            result = 'L'
            winner, loser = away, home
        else:
            result = 'D'
            winner = loser = None

        unique_matches.append(game_id)
        
        if result == 'D':
            matchday_dict[game_id] = (home, away, result)
            preference_data.append((home, away, 'D'))
        else:
            matchday_dict[game_id] = (winner, loser, 'W')
            preference_data.append((winner, loser, 'W'))
            team_win_count[winner] = team_win_count.get(winner, 0) + 1
            team_win_count[loser] = team_win_count.get(loser, 0)

    gold_rankings = sorted(team_win_count.keys(), key=lambda t: team_win_count[t], reverse=True)
    team_dict = {idx: team for idx, team in enumerate(sorted(teams))}
    team_identifier = {team: idx for idx, team in team_dict.items()}

    return preference_data, team_identifier, gold_rankings

def evaluate_ranking(gold_list, predicted_list):
    results = {}

    kendall = kendalltau(gold_list, predicted_list)
    spearman = spearmanr(gold_list, predicted_list)

    results["kendalltau-tau"] = kendall.statistic
    results["kendalltau-pvalue"] = kendall.pvalue
    results["spearman-rho"] = spearman.statistic 
    results["spearman-pvalue"] = spearman.pvalue

    # Top-k Accuracy
    for k in [3, 5]:
        gold_top_k = set(gold_list[:k])
        pred_top_k = set(predicted_list[:k])
        topk_accuracy = len(gold_top_k & pred_top_k) / k
        results[f"top_{k}_accuracy"] = topk_accuracy
        
    true_relevance = np.asarray([gold_list])
    scores = np.asarray([predicted_list])
    ndcg = ndcg_score(true_relevance, scores)
    results["ndcg"] = ndcg

    return results