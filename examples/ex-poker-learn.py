from maspy import *
from maspy.learning import *
from random import shuffle
from collections import Counter
from itertools import combinations

HAND_RANKINGS = {
    "High Card": 1,
    "One Pair": 2,
    "Two Pair": 3,
    "Three of a Kind": 4,
    "Straight": 5,
    "Flush": 6,
    "Full House": 7,
    "Four of a Kind": 8,
    "Straight Flush": 9,
    "Royal Flush": 10,
}

def evaluate_hand(hand):
    ranks = [card[0] for card in hand]
    suits = [card[1] for card in hand]
    rank_counts = Counter(ranks).values()
    
    is_flush = len(set(suits)) == 1
    is_straight = len(set(ranks)) == 5 and max(ranks) - min(ranks) == 4

    if is_flush and is_straight:
        return HAND_RANKINGS["Straight Flush"]
    if 4 in rank_counts:
        return HAND_RANKINGS["Four of a Kind"]
    if 3 in rank_counts and 2 in rank_counts:
        return HAND_RANKINGS["Full House"]
    if is_flush:
        return HAND_RANKINGS["Flush"]
    if is_straight:
        return HAND_RANKINGS["Straight"]
    if 3 in rank_counts:
        return HAND_RANKINGS["Three of a Kind"]
    if list(rank_counts).count(2) == 2:
        return HAND_RANKINGS["Two Pair"]
    if 2 in rank_counts:
        return HAND_RANKINGS["One Pair"]
    return HAND_RANKINGS["High Card"]

class Poker(Environment):
    def __init__(self):
        super().__init__()
        ranks = [2,3,4,5,6,7,8,9,10,11,12,13,14]
        suits = ['S','H','D','C']
        deck = [(r,s) for r in ranks for s in suits]
        all_hands = combinations(deck, 5)
        self.deck = deck.copy()
        shuffle(self.deck)
        self.create(Percept("Hand", (deck, 5), combination))
        
    @action(cartesian, (2,2,2,2,2), None)
    def discard_draw(self, discards):
        for i in range(len(discards)):
            if discards[i] == 1:
                self.get(Percept("Hand",(Any,Any)))
            self.deck.remove(discards[i])
        
if __name__ == "__main__":
    pk = Poker()