import numpy as np

from Card import *
from Commander import Commander
from Player import *
from Process import *

if __name__ == '__main__':
    player_num = 5  # 游戏人数
    print('游戏开始')
    commanders = []
    players = []
    for i in range(player_num):
        commander = Commander('WEI01', '曹操', 4, 4, '魏', [])
        commanders.append(commander)
        players.append(player(commander))
    # 初始化
    get_card_heap = Get_Card_Heap()
    get_card_heap.init_card_heap()  # 初始化摸牌堆
    identity_card_heap = Identity_Card_Heap()
    identity_card_heap.init_card_heap(player_num)  # 初始化身份牌堆
    left_card_heap = Left_Card_Heap()

    # 抽取身份
    for i in range(player_num):
        players[i].identity = identity_card_heap.get_identity()

    # 确定座次
    players_cache = np.zeros(player_num, player)
    players_copy = players.copy()
    for player in players_copy:
        if player.identity == '主公':
            player.idx = 1
            player.max_HP += 1
            player.current_HP += 1
            player.max_HandCards += 1
            players_cache[0] = player
            del players[players.index(player)]
        else:
            while True:
                idx = random.randint(1, player_num - 1)
                if players_cache[idx] == 0:
                    player.idx = idx + 1
                    players_cache[idx] = player
                    del players[players.index(player)]
                    break
    players = list(players_cache.copy())
    for i in range(len(players)):
        if i == 0:
            players[i].next = players[i + 1]
            players[i].pre = players[-1]
        elif i == len(players) - 1:
            players[i].next = players[0]
            players[i].pre = players[i - 1]
        else:
            players[i].next = players[i + 1]
            players[i].pre = players[i - 1]
    del players_cache, players_copy

    # 分发起始手牌
    # check_skill()
    for player in players:
        for start_card in get_card_heap.get_card(4):
            player.HandCards_area.append(start_card)
    # check_skill()
    # 回合开始
    round = 1  # 轮数
    while 1:
        for player in players:
            current_player = player
            print('当前回合角色为:{}号位 {}'.format(current_player.idx, current_player.commander.name))
            Game_Process(player, players, get_card_heap, left_card_heap)

        round += 1
