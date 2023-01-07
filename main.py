# main.py

import random

import Commander
from Card import Get_Card_Heap, Identity_Card_Heap, Left_Card_Heap
from Process import *

if __name__ == '__main__':
    player_num = 2  # 游戏人数
    print('游戏开始')
    Items = items.Items()
    commanders = [Commander.Caocao for _ in range(player_num)]
    Items.PlayerList = [Player.player(c) for c in commanders]
    # 初始化
    Items.GetCardHeap = Get_Card_Heap()
    Items.GetCardHeap.init_card_heap()  # 初始化摸牌堆
    identity_card_heap = Identity_Card_Heap()
    identity_card_heap.init_card_heap(player_num)  # 初始化身份牌堆
    Items.LeftCardHeap = Left_Card_Heap()

    # 抽取身份
    for i in range(player_num):
        Items.PlayerList[i].identity = identity_card_heap.get_identity()

    # 确定座次
    player_list_cache = np.zeros(player_num, Player.player)
    player_list_copy = Items.PlayerList.copy()
    for player in player_list_copy:
        if player.identity == '主公':
            player.idx = 1
            player.max_HP += 1
            player.current_HP += 1
            player.max_HandCards += 1
            player_list_cache[0] = player
            del Items.PlayerList[Items.PlayerList.index(player)]
        else:
            while True:
                idx = random.randint(1, player_num - 1)
                if player_list_cache[idx] == 0:
                    player.idx = idx + 1
                    player_list_cache[idx] = player
                    del Items.PlayerList[Items.PlayerList.index(player)]
                    break
    Items.PlayerList = list(player_list_cache.copy())
    for i in range(len(Items.PlayerList)):
        if i == 0:
            Items.PlayerList[i].next = Items.PlayerList[i + 1]
            Items.PlayerList[i].pre = Items.PlayerList[-1]
        elif i == len(Items.PlayerList) - 1:
            Items.PlayerList[i].next = Items.PlayerList[0]
            Items.PlayerList[i].pre = Items.PlayerList[i - 1]
        else:
            Items.PlayerList[i].next = Items.PlayerList[i + 1]
            Items.PlayerList[i].pre = Items.PlayerList[i - 1]
    del player_list_cache, player_list_copy

    for player in Items.PlayerList:
        for start_card in Items.GetCardHeap.get_card(4, Items.LeftCardHeap):
            player.HandCards_area.append(start_card)
    # check_skill()
    # 回合开始
    round = 1  # 轮数
    while 1:
        for player in Items.PlayerList:
            current_player = player
            print('当前回合角色为:{}号位 {}'.format(current_player.idx, current_player.commander.name))
            Game_Process(player, Items)

        round += 1
