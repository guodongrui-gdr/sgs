# init.py

import Process
import random
import numpy as np
from Commander import commander
from Card import Get_Card_Heap, Identity_Card_Heap, Left_Card_Heap
from Player import player

player_num = 2  # 游戏人数
print('游戏开始')
commanders = []
player_list = []
commanders = [commander('WEI01', '曹操', 4, 4, '魏', []) for _ in range(player_num)]
player_list = [player(c) for c in commanders]
# 初始化
get_card_heap = Get_Card_Heap()
get_card_heap.init_card_heap()  # 初始化摸牌堆
identity_card_heap = Identity_Card_Heap()
identity_card_heap.init_card_heap(player_num)  # 初始化身份牌堆
left_card_heap = Left_Card_Heap()
tmp_card = []  # 处理区

# 抽取身份
for i in range(player_num):
    player_list[i].identity = identity_card_heap.get_identity()

# 确定座次
player_list_cache = np.zeros(player_num, player)
player_list_copy = player_list.copy()
for player in player_list_copy:
    if player.identity == '主公':
        player.idx = 1
        player.max_HP += 1
        player.current_HP += 1
        player.max_HandCards += 1
        player_list_cache[0] = player
        del player_list[player_list.index(player)]
    else:
        while True:
            idx = random.randint(1, player_num - 1)
            if player_list_cache[idx] == 0:
                player.idx = idx + 1
                player_list_cache[idx] = player
                del player_list[player_list.index(player)]
                break
player_list = list(player_list_cache.copy())
for i in range(len(player_list)):
    if i == 0:
        player_list[i].next = player_list[i + 1]
        player_list[i].pre = player_list[-1]
    elif i == len(player_list) - 1:
        player_list[i].next = player_list[0]
        player_list[i].pre = player_list[i - 1]
    else:
        player_list[i].next = player_list[i + 1]
        player_list[i].pre = player_list[i - 1]
del player_list_cache, player_list_copy

# 分发起始手牌
# check_skill()
for player in player_list:
    for start_card in get_card_heap.get_card(4, left_card_heap):
        player.HandCards_area.append(start_card)
# check_skill()
# 回合开始
