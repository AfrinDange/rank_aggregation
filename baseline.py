import numpy as np
import trueskill 

from collections import defaultdict
from glicko2 import Player as GlickoPlayer
from elo_rating import Elo
from scipy.optimize import minimize
from collections import defaultdict

def rank_using_trueskill(preference_data):
    '''
        preference_data: [('South Africa', 'Pakistan', 'W'),
                    ('South Africa', 'West Indies', 'W'),
                    ('South Africa', 'West Indies', 'D'),
                    ('South Africa', 'New Zealand', 'L'),
    '''
    ratings = defaultdict(trueskill.Rating)
    for p1, p2, outcome in preference_data:
        if outcome == 'W':
            ratings[p1], ratings[p2] = trueskill.rate_1vs1(ratings[p1], ratings[p2])
        elif outcome == 'L':
            ratings[p2], ratings[p1] = trueskill.rate_1vs1(ratings[p2], ratings[p1])
        else:
            ratings[p1], ratings[p2] = trueskill.rate_1vs1(ratings[p1], ratings[p2], drawn=True)
            
    trueskill_rankings = []
    for team, rating in sorted(ratings.items(), key=lambda x: x[1].mu, reverse=True):
        trueskill_rankings.append(team)
    
    return trueskill_rankings

def rank_using_elo(preference_data, k=0.15):
    e = Elo()
    for p1, p2, outcome in preference_data:
        if outcome == 'W':
            e.add_match(p1, p2, 1.0, k=k)     
        elif outcome == 'L':
            e.add_match(p1, p2, 0.0, k=k)   
        else:
            e.add_match(p1, p2, 0.5, k=k)    

    sorted_teams = sorted(e.ratings().items(), key=lambda x: x[1], reverse=True)
    return [team for team, _ in sorted_teams]


def rank_using_glicko(preference_data):
    players = defaultdict(GlickoPlayer)
    for p1, p2, outcome in preference_data:
        if outcome == 'W':
            players[p1].update_player([players[p2].rating], [players[p2].rd], [1])
            players[p2].update_player([players[p1].rating], [players[p1].rd], [0])
        elif outcome == 'L':
            players[p2].update_player([players[p1].rating], [players[p1].rd], [1])
            players[p1].update_player([players[p2].rating], [players[p2].rd], [0])
        else:
            players[p1].update_player([players[p2].rating], [players[p2].rd], [0.5])
            players[p2].update_player([players[p1].rating], [players[p1].rd], [0.5])
    return [team for team, _ in sorted(players.items(), key=lambda x: x[1].rating, reverse=True)]

def rank_using_bradley_terry(preference_data):
    teams = list(set([x[0] for x in preference_data] + [x[1] for x in preference_data]))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    win_matrix = np.zeros((n, n))
    for p1, p2, outcome in preference_data:
        i, j = team_to_idx[p1], team_to_idx[p2]
        if outcome == 'W':
            win_matrix[i, j] += 1
        elif outcome == 'L':
            win_matrix[j, i] += 1
        else:
            win_matrix[i, j] += 0.5
            win_matrix[j, i] += 0.5

    def neg_log_likelihood(params):
        likelihood = 0.0
        for i in range(n):
            for j in range(n):
                if i != j and (win_matrix[i, j] + win_matrix[j, i] > 0):
                    pi = np.exp(params[i])
                    pj = np.exp(params[j])
                    pij = pi / (pi + pj)
                    likelihood += win_matrix[i, j] * np.log(pij)
        return -likelihood

    init_params = np.zeros(n)
    res = minimize(neg_log_likelihood, init_params, method='L-BFGS-B')
    strengths = res.x
    ranked_teams = [team for team, _ in sorted(zip(teams, strengths), key=lambda x: x[1], reverse=True)]
    return ranked_teams
    