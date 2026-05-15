import os
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import multiprocessing as mp 
from concurrent.futures import ProcessPoolExecutor

from main_env import CinchMainEnv, SUITS, RANKS
from manage_checkpoints import find_all_checkpoint_dirs, find_all_checkpoints
from get_agent import get_agent, select_action

CHECKPOINT_DICT = {}
MODEL_DICT = {}
RATINGS = {}

DEBUG = False
INITIAL_ELO = 1000
NUM_TESTS = 1_000
NUM_WORKERS = mp.cpu_count() - 1

def get_new_elo(player_elo, opponent_elo, result, k=16):
    """
    Calculate the new Elo rating for a player after a match.

    Parameters:
    - player_elo: Current Elo rating of the player
    - opponent_elo: Current Elo rating of the opponent
    - result: Some aggregate value between 0 and 1 representing the outcome of the match (1 for a win, 0.5 for a draw, 0 for a loss)
    - k: The K-factor, which determines how much the ratings change (default is 16)

    Returns:
    - new_player_elo: The new Elo rating for the player
    """
    expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    new_player_elo = player_elo + k * (result - expected_score)
    return new_player_elo

def run_matchup(args):
    """
    Run a matchup between two models for a specified number of games.
    
    Parameters:
     - args: A tuple containing (model_name_1, path_1, model_name_2, path_2, num_games)
         - model_name_1: Name of the first model
         - path_1: Checkpoint path for the first model
         - model_name_2: Name of the second model
         - path_2: Checkpoint path for the second model
         - num_games: Number of games to play in the matchup
         
    Returns:
     - A dictionary containing the results of the matchup, including wins, losses, ties, and bet points for each model.
    """
    model_name_1, path_1, model_name_2, path_2, num_games = args 
    env = CinchMainEnv(debug=False) 
    model_1 = get_agent(env, path_1)
    model_2 = get_agent(env, path_2)
    wins = 0 
    losses = 0 
    ties = 0 
    total_bet_points = [0, 0] 
    bet_1 = [] 
    bet_2 = [] 
    for game_idx in range(num_games): 
        obs = env.reset() # Alternate seating to reduce bias 
        swap_seats = (game_idx % 2 == 1) 
        while not env.done: 
            if swap_seats: 
                team0_model = model_2 
                team1_model = model_1 
                team0_path = path_2 
                team1_path = path_1 
            else: 
                team0_model = model_1 
                team1_model = model_2 
                team0_path = path_1 
                team1_path = path_2 
            if env.current_player % 2 == 0: 
                action = select_action(team0_model, team0_path, obs, env) 
            else: 
                action = select_action(team1_model, team1_path, obs, env) 
            obs, rewards, done, _ = env.step(action) 

        team0_score = env.bet_points[0] 
        team1_score = env.bet_points[1] 
        # Undo seat swapping for stats 
        if swap_seats: 
            team0_score, team1_score = (team1_score, team0_score) 
        bet_1.append(team0_score) 
        bet_2.append(team1_score) 
        total_bet_points[0] += team0_score 
        total_bet_points[1] += team1_score 
        if team0_score > team1_score: 
            wins += 1 
        elif team0_score < team1_score: 
            losses += 1 
        else: 
            ties += 1 

    print(f"Matchup: {model_name_1} vs {model_name_2} - Wins: {wins}, Losses: {losses}, Ties: {ties}, Average Bet Points {model_name_1}: {np.mean(bet_1):.3f}, Average Bet Points {model_name_2}: {np.mean(bet_2):.3f}")
    return {
        "model_1": model_name_1, 
        "model_2": model_name_2, 
        "wins": wins, 
        "losses": losses, 
        "ties": ties, 
        "bet_1": bet_1, 
        "bet_2": bet_2, 
        "total_bet_points": total_bet_points
    }


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    env = CinchMainEnv(debug=DEBUG)

    # First load all the models in a dictionary.
    checkpoint_dirs = find_all_checkpoint_dirs(".")
    for checkpoint_dir in checkpoint_dirs:
        checkpoints = find_all_checkpoints(checkpoint_dir)
        for checkpoint in checkpoints:
            model_name = os.path.dirname(checkpoint) + "_" + os.path.basename(checkpoint).split('.')[0]
            # Remove any non-alphanumeric characters except underscores from the model name.
            model_name = ''.join(c for c in model_name if c.isalnum() or c == '_')
            CHECKPOINT_DICT[model_name] = checkpoint
            get_agent(env, checkpoint)
            RATINGS[model_name] = INITIAL_ELO
            print('=' * 20, f"Successfully loaded model {model_name}", '=' * 20)

    # Create a game schedule.
    model_names = list(CHECKPOINT_DICT.keys()) 
    game_schedule = [] 
    for i in range(len(model_names)): 
        for j in range(i + 1, len(model_names)): 
            model_1_name = model_names[i] 
            model_2_name = model_names[j] 
            game_schedule.append((model_1_name, CHECKPOINT_DICT[model_1_name], model_2_name, CHECKPOINT_DICT[model_2_name], NUM_TESTS)) 

    # Shuffle the game schedule to randomize the order of matchups.
    np.random.shuffle(game_schedule)
    print(f"Total matchups: {len(game_schedule)}")

    # Make a results directory if it doesn't exist.
    os.makedirs("results", exist_ok=True)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor: 
        results = list( 
            tqdm(
                executor.map(run_matchup, game_schedule), 
                total=len(game_schedule) 
            ) 
        )

    # Process results and update Elo ratings.
    for result in tqdm(results, desc="Processing Results"):
        model_1 = result["model_1"]
        model_2 = result["model_2"]
        wins = result["wins"]
        losses = result["losses"]
        ties = result["ties"]
        total_games = (wins + losses + ties) 
        
        # Elo result 
        score = (wins + 0.5 * ties) / total_games 
        new_rating_1 = get_new_elo(RATINGS[model_1], RATINGS[model_2], score) 
        new_rating_2 = get_new_elo(RATINGS[model_2], RATINGS[model_1], 1 - score) 
        print(f"\nMatchup: {model_1} vs {model_2}")
        print(f"Result: {wins} wins, {losses} losses, {ties} ties (Winrate: {score:.3f})")
        print( f"{model_1}: " f"{RATINGS[model_1]:.1f}" f" -> " f"{new_rating_1:.1f}" ) 
        print( f"{model_2}: " f"{RATINGS[model_2]:.1f}" f" -> " f"{new_rating_2:.1f}" )
        print(f"Average Bet Points {model_1}: {np.mean(result['bet_1']):.3f}")
        print(f"Average Bet Points {model_2}: {np.mean(result['bet_2']):.3f}")
        print("=" * 100)
        RATINGS[model_1] = new_rating_1 
        RATINGS[model_2] = new_rating_2 

        # Save results 
        result_file = os.path.join("results", f"{model_1}_vs_{model_2}.txt") 
        with open(result_file, "w") as f: 
            f.write( f"Wins: {wins}\n" ) 
            f.write( f"Losses: {losses}\n" ) 
            f.write( f"Ties: {ties}\n" ) 
            f.write( f"Winrate: " f"{wins / total_games:.3f}\n" ) 
            f.write( f"Average Bet Points " f"{model_1}: " f"{np.mean(result['bet_1']):.3f}\n" ) 
            f.write( f"Average Bet Points " f"{model_2}: " f"{np.mean(result['bet_2']):.3f}\n" ) 

        csv_file = os.path.join("results", f"{model_1}_vs_{model_2}.csv") 
        with open(csv_file, "w") as f:
            f.write("Game Index,Bet Points Model 1,Bet Points Model 2\n") 
            for i in range(total_games): 
                f.write( f"{i+1},{result['bet_1'][i]},{result['bet_2'][i]}\n" )

    # Final rankings
    print("\nFinal Elo Rankings\n") 
    sorted_ratings = sorted(RATINGS.items(), key=lambda x: x[1], reverse=True) 
    for name, rating in sorted_ratings: 
        print( f"{name} " f"{rating:.1f}" )