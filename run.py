import argparse
import json
import numpy as np
import os
import pandas as pd
import random


from collections import defaultdict
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login 

from baseline import rank_using_trueskill, rank_using_elo, rank_using_glicko, rank_using_bradley_terry
from aggregation_mechanisms import rank_using_direct_prompt, rank_using_self_consistency, rank_using_implicit_cot, rank_using_explicit_cot
from utils import process_icc_dataset, process_nfl_dataset, evaluate_ranking

parser = argparse.ArgumentParser(description="Run Rank Aggregation")
parser.add_argument("--method", type=str, choices=["direct_prompt", "trueskill", "elo", "glicko", "bradley_terry", "self_consistency", "implicit_chain_of_thought", "explicit_chain_of_thought"])
parser.add_argument("--csv_path", type=str, default="results.csv")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--shuffle", action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # set up 
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    login(token)

    random.seed(42)
    
    if args.dataset == "icc-2023-2025":
        dataset_name = "icc-test-championship-rankings-2023-2025-cycle"
        dataset = load_dataset("konan-kun/" + dataset_name)
        dataset = dataset['train'].to_pandas()

        preference_data, team_identifier, gold_rankings = process_icc_dataset(dataset)
    elif args.dataset.startswith("nfl"):
        season = int(args.dataset.split("-")[-1])
        dataset = pd.read_csv("./data/nfl_mahomes_era_games.csv")
        names = json.load(open("./data/names.json", "r"))
        dataset["home_team"] = dataset["home_team"].map(names)
        dataset["away_team"] = dataset["away_team"].map(names)
    
        preference_data, team_identifier, _ = process_nfl_dataset(dataset, season)
        # names = json.load(open("./data/names.json", "r"))
        # gold_rankings = [names[id] for id in json.load(open("./data/ranks.json", "r"))[str(season)]]
        gold_rankings = json.load(open("./data/ranks.json", "r"))[str(season)]
        
    # perform rank aggregation 
    if args.method == "trueskill":
        predicted_rankings = rank_using_trueskill(preference_data)
    if args.method == "elo":
        predicted_rankings = rank_using_elo(preference_data)
    if args.method == "glicko":
        predicted_rankings = rank_using_glicko(preference_data)
    if args.method == "bradley_terry":
        predicted_rankings = rank_using_bradley_terry(preference_data)
    elif args.method == "direct_prompt":
        llm_preference_data = [f"{tup[0]} vs {tup[1]}, {tup[int(tup[2] == 'L')] + ' won' if tup[2] != 'D' else 'Match Drawn'}" for tup in preference_data]
        if args.shuffle:
            random.shuffle(llm_preference_data)
        formatted_preference_data = '\n'.join(llm_preference_data)
        predicted_rankings = rank_using_direct_prompt(formatted_preference_data)
    elif args.method == "self_consistency":
        llm_preference_data = [f"{tup[0]} vs {tup[1]}, {tup[int(tup[2] == 'L')] + ' won' if tup[2] != 'D' else 'Match Drawn'}" for tup in preference_data]
        if args.shuffle:
            random.shuffle(llm_preference_data)
        formatted_preference_data = '\n'.join(llm_preference_data)
        predicted_rankings = rank_using_self_consistency(formatted_preference_data, num_samples=10)
    elif args.method == "implicit_chain_of_thought":
        llm_preference_data = [f"{tup[0]} vs {tup[1]}, {tup[int(tup[2] == 'L')] + ' won' if tup[2] != 'D' else 'Match Drawn'}" for tup in preference_data]       
        if args.shuffle:
            random.shuffle(llm_preference_data)
        formatted_preference_data = '\n'.join(llm_preference_data)
        predicted_rankings = rank_using_implicit_cot(formatted_preference_data)
    elif args.method == "explicit_chain_of_thought":
        llm_preference_data = [f"{tup[0]} vs {tup[1]}, {tup[int(tup[2] == 'L')] + ' won' if tup[2] != 'D' else 'Match Drawn'}" for tup in preference_data]       
        if args.shuffle:
            random.shuffle(llm_preference_data)
        formatted_preference_data = '\n'.join(llm_preference_data)
        predicted_rankings = rank_using_explicit_cot(formatted_preference_data)
        
    gold_list = [team_identifier[team] for team in gold_rankings]
    predicted_list = [team_identifier[team] for team in predicted_rankings]

    results = evaluate_ranking(gold_list, predicted_list)
    
    results["gold_ranking"] = gold_rankings
    results["predicted_ranking"] = predicted_rankings

    metrics = list(results.keys())
    values = list(results.values())
    index = pd.MultiIndex.from_tuples([(args.dataset, metric) for metric in metrics], names=["dataset", "metric"])
    df = pd.DataFrame({args.method: values}, index=index)

    try:
        existing_df = pd.read_csv(f"{args.dataset}_{args.csv_path}", header=[0], index_col=[0, 1])
        existing_df[args.method] = df[args.method]
        existing_df.to_csv(f"{args.dataset}_{args.csv_path}")
    except FileNotFoundError:
        df.to_csv(f"{args.dataset}_{args.csv_path}")
    