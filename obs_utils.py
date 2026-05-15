import torch
import numpy as np

from main_env import SUITS, RANKS
from time_it import timeit

def build_mlp_obs(obs):
    """
    Convert the observation dictionary to a tensor.
    The observation dictionary contains the following keys:
     - "hand": a binary vector of length 52 indicating which cards are in the player's hand
     - "trick": a normalized integer vector of length 52 indicating which cards have been played in the current trick
     - "cards_played": a float vector (normalized by 1/24) of length 52 indicating which cards have been played in the game
     - "trump": a normalized integer from 0 to 3 indicating the trump suit (0=hearts, 1=diamonds, 2=clubs, 3=spades)
     - "player": a normalized integer from 0 to 3 indicating the player's position (0=first player, 1=second player, 2=third player, 3=fourth player)
     - "void": a flattened binary vector of length 16 indicating which suits the various players are void in based on the cards that have been played (4 suits x 4 players)
     - "trick_pos": a normalized integer from 0 to 3 indicating the player's position in the current trick (0=first to play, 1=second to play, etc.)
     - "lead_suit": a binary vector of length 5 indicating the lead suit of the current trick (one-hot encoding with an extra index for no lead suit)
     - "trick_winner": A binary vector of length 3 indicating the winner of the current trick (0=opposing team, 1=current team, 2=neither).
     - "trick_value": A normalized float indicating the value of the current trick based on the cards played so far. (Indicates whether the trick is currently a junk trick or a high value trick)
     - "can_win_mask": A binary vector of length 52 indicating which legal cards in the player's hand can currently win the trick based on the cards that have been played so far.
     - "trumps_at_start": a normalized integer list of length 4 indicating how many trump cards each player had before drawing from the deck
     - "count_from_deck": a normalized integer list of length 4 indicating how many cards each player has drawn from the deck
     - "count_from_dead_wood": a normalized integer list of length 4 indicating how many cards each player has drawn from the dead wood
     - "count_in_widow": an normalized integer indicating how many cards are currently in the widow (partially important in case certain point trump cards are missing from the game)
    """
    lead = obs["lead_suit"]
    lead_vec = np.zeros(5)
    if lead != -1:
        lead_vec[lead] = 1
    else:
        lead_vec[4] = 1

    trick_winner = obs["trick_winner"]
    trick_winner_vec = np.zeros(3)
    if trick_winner != -1:
        trick_winner_vec[trick_winner] = 1
    else:
        trick_winner_vec[2] = 1

    # print("Len of hand vector:", len(obs["hand"]))
    # print("Len of trick vector:", len(obs["trick"]))
    # print("Len of cards_played vector:", len(obs["cards_played"]))
    # print("Len of trump vector:", len(np.eye(4)[obs["trump"]]))
    # print("Len of player vector:", len(np.eye(4)[obs["player"]]))
    # print("Len of void vector:", len(obs["void"]))
    # print("Len of trick_pos vector:", len(np.array([obs["trick_pos"]])))
    # print("Len of lead_suit vector:", len(lead_vec))
    # print("Len of trumps_at_start vector:", len(np.array(obs["trumps_at_start"])))
    # print("Len of count_from_deck vector:", len(np.array(obs["count_from_deck"])))
    # print("Len of count_from_dead_wood vector:", len(np.array(obs["count_from_dead_wood"])))
    # print("Len of count_in_widow vector:", len(np.array([obs["count_in_widow"]])))

    final_tensor = torch.tensor(np.concatenate([
        obs["hand"],
        obs["trick"] / 3.0,
        obs["cards_played"] / 24.0, # Each of the 4 players has 6 cards.
        np.eye(4)[obs["trump"]] / 3.0,
        np.eye(4)[obs["player"]] / 3.0,
        obs["void"],
        np.array([obs["trick_pos"]]) / 3.0,
        lead_vec,
        trick_winner_vec,
        np.array([obs["trick_value"]]),
        np.array(obs["can_win_mask"]),
        np.array(obs["trumps_at_start"]) / 6.0,
        np.array(obs["count_from_deck"]) / 6.0,
        np.array(obs["count_from_dead_wood"]) / 6.0,
        np.array([obs["count_in_widow"]]) / 5.0 # There can be at most 5 cards in the widow. 52 - 11 drawn (13 of trump in game before) - 36 from initial drawing = 5
    ]), dtype=torch.float32)
    return final_tensor


@timeit
def build_transformer_obs(obs):
    """
    Converts the raw observation from the environment into a structured format suitable for input to the transformer model.
    
    The output is a dictionary with two keys:
    - "cards": A tensor of shape (52, feature_dim) containing features for each card in the deck.
        Each card's features include:
        - Suit (0-3)
        - Rank (0-12)
        - Location (0=unknown, 1=in hand, 2=in current trick, 3=already played)
        - Played by (0=played by me, 1=played by player on left, 2=played by partner, 3=played by player on right, 4=not played)
        - Played order (0 if not played, otherwise the order in which it was played)
        - Legal (1 if the card is legal to play, 0 otherwise)
    - "global": A tensor containing global features about the current game state.
        Each feature is encoded as an integer:
        - Trump suit (0-3)
        - Trick position (0-3)
        - Lead suit (0-3, with 4 representing no lead suit)
        - Trick winner (0=opposing team, 1=own team, 2=none yet)
        - Current player (0-3)
        - Trumps at least (4 values representing the minimum number of trumps each player has, normalized by 6)
        - Count from deadwood (4 values representing the number of cards each player has played from deadwood, normalized by 6)
        - Count in widow (number of cards in the widow, normalized by 5)
        - Void (4 values representing whether each player is void in each suit for each player, 0 or 1). 
          A player could be void in a suit, but other players might not know that yet, so this is based on the current player's knowledge of voids.
    """
    cards = []
    player = obs["player"]

    for suit_i, suit in enumerate(SUITS):
        for rank_i, rank in enumerate(RANKS):
            idx = suit_i * 13 + rank_i

            # -------------------------
            # Location encoding
            # 0 = unknown
            # 1 = in hand
            # 2 = in current trick
            # 3 = already played
            # -------------------------
            location = 0
            played_by = 4  # default to not played
            if obs["hand"][idx] == 1:
                location = 1
            elif obs["trick"][idx] > 0:
                location = 2
            elif obs["cards_played"][idx] > 0:
                location = 3
                played_by = int(obs["cards_played_by"][idx] - player) % 4  # 0=played by me, 1=played by player on left, 2=played by partner, 3=played by player on right               

            played_order = obs["cards_played"][idx]

            legal = idx in obs["legal_actions"]

            cards.append([
                suit_i,
                rank_i,
                location,
                played_by,
                played_order,
                int(legal)
            ])

    cards = torch.tensor(cards, dtype=torch.long)

    # Adjust trumps_at_least and count_from_deadwood to be in player order starting from current player
    trumps_at_least = obs["trumps_at_least"][player:] + obs["trumps_at_least"][:player]
    count_from_deadwood = obs["count_from_dead_wood"][player:] + obs["count_from_dead_wood"][:player]
    void = list(obs["void"][player*4:]) + list(obs["void"][:player*4])

    global_features_list = [
        obs["trump"],
        obs["trick_pos"],
        obs["lead_suit"] if obs["lead_suit"] != -1 else 4,
        obs["trick_winner"] if obs["trick_winner"] != -1 else 2,
        player,
    ] + list(trumps_at_least) + list(count_from_deadwood) + [obs["count_in_widow"]] + list(void)
    global_features = torch.tensor(global_features_list, dtype=torch.long)

    return {
        "cards": cards,
        "global": global_features
    }