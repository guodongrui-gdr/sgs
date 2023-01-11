# items.py
import card
import player
from typing import List


class Items:
    def __init__(self, player_num):
        self.GetCardHeap: card.GetCardHeap = card.GetCardHeap()
        self.LeftCardHeap: card.LeftCardHeap = card.LeftCardHeap()
        self.IdentityCardHeap = card.IdnetityCardHeap()
        self.TmpCard: List[card.Card] = []
        self.PlayerList: List[player.Player] = []

        # 初始化
        self.GetCardHeap.init_card_heap()
        self.IdentityCardHeap.init_card_heap(player_num)
