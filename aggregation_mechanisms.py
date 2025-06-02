import json
import os 
import time 

from collections import Counter
from dotenv import load_dotenv
from google import genai 
from google.genai import types
from tqdm import tqdm 

def rank_using_direct_prompt(formatted_preference_data):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=google_api_key)
    model = "gemini-2.0-flash"

    system_inst = '''
You are given pairwise match results between teams, where each match can result in a win, loss, or a draw. Generate the final team rankings based on overall wins, losses, draws and number of matches played by a team.
Respond with the ranked list of only team names in descending order of skill in JSON format. The list should contain all teams.
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
        temperature=0.0,
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

    system_inst = '''
You are given pairwise match results between teams, where each match can result in a win, loss, or a draw. Generate the final team rankings based on overall wins, losses, draws and number of matches played by a team.
Respond with the ranked list of only team names in descending order of skill in JSON format. The list should contain all teams.
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
        temperature=0.5,
        response_mime_type="application/json",
        system_instruction=[types.Part.from_text(text=system_inst)],
    )
    
    all_rankings = []
    
    for _ in tqdm(range(num_samples), leave=False):
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            response_text += chunk.text
        time.sleep(20)
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

def rank_using_explicit_cot(formatted_preference_data):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=google_api_key)
    model = "gemini-2.0-flash"

    system_inst='''  
You are a ranking analyst using the Elo rating system. Compute rankings as follows:  

Elo Algorithm  
1. Initialize: All teams start with 1500 Elo points.  
2. Expected Score: For Team X (rating \(R_X\)) vs Team Y (\(R_Y\)):  
   \[
   E_X = \frac{1}{1 + 10^{(R_Y - R_X)/400}}, \quad E_Y = 1 - E_X
   \]  
3. Update Ratings after each match (K=32):  
   - Win: \(R_{\text{winner}} = R_{\text{winner}} + K \times (1 - E_{\text{winner}})\)  
   - Loss: \(R_{\text{loser}} = R_{\text{loser}} + K \times (0 - E_{\text{loser}})\)  
   - Draw: Both teams get \(R = R + K \times (0.5 - E)\)  

Process  
1. Process matches sequentially.  
2. Track rating changes using the formulas above.  
3. Sort teams by final Elo (highest first).  

Example  
Input:  
Team A vs Team B: B won  
Team A vs Team C: A won  
Team A vs Team D: Match Drawn

Output:  
Respond with the ranked list of only team names in descending order of skill in JSON format. The list should contain all teams.
["Team B", "Team A", "Team C"]
'''  
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_preference_data)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.0,
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

def rank_using_implicit_cot(formatted_preference_data):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=google_api_key)
    model = "gemini-2.0-flash"

    system_inst='''
1. Create a markdown table of each team’s match outcomes:

| Team        | Wins | Losses | Draws | Win Percentage | Points | Net Wins |
|-------------|------|--------|-------|----------------|--------|----------|
| Example     | 2    | 1      | 1     | 66.67%         | 2.5    | 1        |

**Scoring Rules:**
- Win = 1 point
- Draw = 0.5 points
- Loss = 0 points
- Win Percentage = (Wins / (Wins + Losses + Draws)) * 100
- Net Wins = Wins - Losses

If a team is not mentioned in any match, include them with zeroes in all columns.

2. Generate 3 Candidate Rankings:
| Method              | Ranking                     |
|---------------------|-----------------------------|
| 1. Win Percentage   | ["Team A", "Team B", "Team C"] |
| 2. Points           | ["Team A", "Team C", "Team B"] |
| 3. Head-to-Head     | ["Team B", "Team A", "Team C"] |

3. Compare Ranking Candidates:
- Head-to-head accuracy: Fewer contradictions with match outcomes.
- Win Percentage: Higher win percentage indicates better overall performance.
- Points: Total points accumulated.
- Win Consistency: Longest win/loss streak or least alternation.
- Lexical Tie-breaker: Alphabetical order if still tied.

4. Tiebreaker Protocol:
1. Direct match outcomes
2. Average margin of victory (e.g., if available)
3. Win/loss streak consistency
4. Alphabetical order

5. Verification Table:
| Team Pair      | Actual Winner | Higher Ranked Team | Consistency |
|----------------|---------------|--------------------|-------------|
| Team A vs Team B | Team A        | Team A             | ✓           |

6. Final Output (JSON):
Respond with the ranked list of only team names in descending order of skill in JSON format. The list should contain all teams.
["Team A", "Team B", "Team C"]

7. Input Example:
Team A vs Team B, Team A won  
Team B vs Team C, Team C won  
Team A vs Team C, Match Drawn
'''  
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_preference_data)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.0,
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
