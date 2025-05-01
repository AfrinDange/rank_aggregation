import json
import os 

from collections import Counter
from dotenv import load_dotenv
from google import genai 
from google.genai import types

def rank_using_direct_prompt(formatted_preference_data):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=google_api_key)
    model = "gemini-2.0-flash"

    system_inst = '''You are provided a set of match results between teams. 
Each result includes the winner and loser of a match, or a draw. Your task is to compute the final rankings for the teams based on their overall performance in these matches. Consider the following metrics for ranking:
- Number of wins
- Number of losses
- Number of draws

Rank the teams in descending order of their overall skill and performance. The team with the highest number of wins, followed by fewer losses and draws, will be ranked first. In case of ties, rank based on other performance metrics like match consistency and the margin of victories.

The results are formatted as follows:
- "team1 vs team2 result" where result can be "team1 won", "team2 won", or "match drawn".

Respond only with the final ranking of teams as a JSON list, sorted in descending order of performance:
Example:
["Team A", "Team B", "Team C"]
'''

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_preference_data)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json",
        system_instruction=[types.Part.from_text(text=system_inst)],
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        response_text += chunk.text

    try:
        rankings = json.loads(response_text)
        if not isinstance(rankings, list):
            raise ValueError("Expected JSON list of strings")
        return [str(team) for team in rankings]
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response from model: {e}")
    
def rank_using_self_consistency(formatted_preference_data, num_samples=10):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=google_api_key)
    model = "gemini-2.0-flash"

    system_inst = '''You are provided a set of match results between teams. 
Each result includes the winner and loser of a match, or a draw. Your task is to compute the final rankings for the teams based on their overall performance in these matches. Consider the following metrics for ranking:
- Number of wins
- Number of losses
- Number of draws

Rank the teams in descending order of their overall skill and performance. The team with the highest number of wins, followed by fewer losses and draws, will be ranked first. In case of ties, rank based on other performance metrics like match consistency and the margin of victories.

The results are formatted as follows:
- "team1 vs team2 result" where result can be "team1 won", "team2 won", or "match drawn".

Respond only with the final ranking of teams as a JSON list, sorted in descending order of performance:
Example:
["Team A", "Team B", "Team C"]
'''

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_preference_data)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json",
        system_instruction=[types.Part.from_text(text=system_inst)],
    )
    
    all_rankings = []
    
    for _ in range(num_samples):
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            response_text += chunk.text

        try:
            rankings = json.loads(response_text)
            if not isinstance(rankings, list):
                raise ValueError("Expected JSON list of strings")
            all_rankings.append([str(team) for team in rankings])
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from model: {e}")
    
    print(all_rankings)
    final_ranking = []
    num_positions = len(all_rankings[0])
    
    for pos in range(num_positions):
        teams_in_position = [ranking[pos] for ranking in all_rankings if ranking[pos] not in final_ranking]
        
        position_counter = Counter(teams_in_position)
        most_common_team = position_counter.most_common(1)[0][0]
        final_ranking.append(most_common_team)
        
    return final_ranking

def rank_using_cot(formatted_preference_data):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=google_api_key)
    model = "gemini-2.0-flash"

#     system_inst = '''
# You are an expert sports analyst. You are given a list of match results between cricket teams. Each result is in the form:

# "Team A vs Team B result", where the result can be "Team A won", "Team B won", or "match drawn".

# Your task is to:
# 1. Tally the number of wins, losses, and draws for each team.
# 2. Use this information to reason about each team's overall performance.
# 3. Rank the teams in descending order based on their performance (more wins, fewer losses, draws considered in tie-breaking).
# 4. Present your reasoning clearly step-by-step (Chain of Thought).
# 5. Finally, return **only** the final ranking as a JSON list in the following format:
# {
#   "ranking": ["Team A", "Team B", "Team C"]
# }

# Only output valid JSON.
# '''
# '''
# Instruction:
# Analyze these cricket match results and create team rankings using step-by-step reasoning. Follow this process:

# 1. Chain-of-Thought Analysis:
#    - Calculate and list each team's performance metrics in a markdown table format:
#      | Team | Wins | Losses | Draws | Points | Net Wins (Wins - Losses) |
#    - Example: For "South Africa vs India, South Africa won", South Africa gains 1 win, India gets 1 loss.

# 2. RankPrompt Comparison:
#    - Generate 3 candidate rankings using different methods:
#      1. Pure win-count
#      2. Net wins (Wins - Losses)
#      3. Head-to-head performance
#    - Compare these candidates step-by-step using criteria:
#      a) Which method best reflects actual match outcomes?
#      b) Which has fewest contradictions with head-to-head results?
#      c) Which follows tournament rules most accurately?

# 3. Tiebreaker Protocol:
#    - For teams with equal points:
#      1. Compare direct match outcomes
#      2. Calculate average victory margin (use <RUN_RATE> if available)
#      3. Consider consistency (streak of wins)

# 4. Verification Check:
#    - Create a verification table:
#      | Team Pair | Actual Winner | Ranking Position | Consistency Check |
#    - Validate if higher-ranked teams beat lower-ranked ones in direct matches.

# Output Format:
# - Final JSON list must align with verification check results
#     Example:
#     ["South Africa", "India", "New Zealand"]

# Advanced Rules:
# - If conflict between metrics exists, use this priority: 
#   1. Head-to-head > 2. Net wins > 3. Win consistency > 4. Random selection
# - For drawn matches, award 0.5 points to each team
# - Penalize teams with forfeits by subtracting 2 points per forfeit

# Match Results (Input Example):
# South Africa vs India, South Africa won
# Australia vs Sri Lanka, Sri Lanka won
# India vs New Zealand, Match Drawn
#     '''
    system_inst='''
You are given a list of cricket match results. Use step-by-step reasoning to analyze and rank the teams based on their performances. Follow the exact process below to ensure deterministic output.

1. Chain-of-Thought Analysis  
Create a markdown table of each team’s match outcomes:

| Team           | Wins | Losses | Draws | Forfeits | Points | Net Wins |
|----------------|------|--------|-------|----------|--------|----------|
| Example Team A | 2    | 1      | 1     | 0        | 2.5    | 1        |

Scoring Rules:
- Win = 1 point  
- Draw = 0.5 points  
- Loss = 0 points  
- Forfeit = –2 points  
- Net Wins = Wins – Losses  
- If a team is not mentioned in any match, include them with zeroes in all columns.

2. Generate 3 Candidate Rankings

| Method                  | Ranking                          |
|-------------------------|----------------------------------|
| 1. Win Count            | ["Team A", "Team B", "Team C"]   |
| 2. Net Wins             | ["Team A", "Team C", "Team B"]   |
| 3. Head-to-Head         | ["Team B", "Team A", "Team C"]   |

3. Compare Ranking Candidates

Use these criteria in this exact order to select the best ranking:
1. Head-to-head accuracy: Fewer contradictions with actual match outcomes.  
2. Net wins: More consistent win–loss advantage.  
3. Win consistency: Longest streaks or fewest alternations between win/loss.  
4. Lexical tie-breaker: If tied, sort tied teams alphabetically.

4. Tiebreaker Protocol

If two or more teams have equal points or are otherwise tied, break ties in this order:
1. Direct match outcomes  
2. Average margin of victory (use <RUN_RATE> if provided)  
3. Win/loss streak consistency  
4. Alphabetical order of team names (as final fallback)

5. Verification Table

Create a validation table:

| Team Pair        | Actual Winner | Higher Ranked Team | Consistency Check |
|------------------|----------------|---------------------|-------------------|
| Team A vs Team B | Team A         | Team A              | ✅                |

Ensure no lower-ranked team beats a higher-ranked team, unless justified by rules above.

6. Final Output

Output ONLY the final ranking in JSON format: ["South Africa", "India", "New Zealand"]

Input Example:
South Africa vs India, South Africa won  
Australia vs Sri Lanka, Sri Lanka won  
India vs New Zealand, Match Drawn

Notes:
- Always include every team mentioned.
- Do NOT randomly select between tied teams — use deterministic tie-breaking as described.
- This is a deterministic task. All steps must lead to a reproducible, verifiable result.
'''
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_preference_data)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json",
        system_instruction=[types.Part.from_text(text=system_inst)],
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        response_text += chunk.text
    try:
        parsed = json.loads(response_text)
        print(parsed)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON object with a 'ranking' field")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON output: {e}\n\nRaw response:\n{response_text}")
