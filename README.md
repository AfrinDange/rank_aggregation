## To run the code:

Create environment
`conda env create -f environment.yml`

Run experiments using
dataset can be icc-2023-2025, nfl-2018, nfl-2019, nfl-2020, nfl-2020, nfl-2022, nfl-2023
method can be elo, glicko, trueskill, bradley_terry, direct_prompt, self-consistency, implicit_chain_of_thought, explicit_chain_of_thought
`python .\run.py --dataset <dataset> --method <method>`


# About Data
Rankings scraped from https://www.nfl.com/standings/league/<year>/REG 
Match Outcomes data - https://data.scorenetwork.org/football/nfl-game-outcomes.html
ICC match outcome and rankings scraped from ICC website.
