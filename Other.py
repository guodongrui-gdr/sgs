from Player import player
import numpy as np


def isAreaEmpty(player: player) -> bool:  # 判断玩家区域内是否有牌
    return len(player.HandCards_area) == 0 and len(player.pandin_area) == 0 and np.array(
        [v.name for k, v in player.equipment_area.items()]).all() is None


def cal_dis(player, player_list):  # 计算距离
    res_next = {}
    res_pre = {}
    res = {}
    player_tmp = player
    dis = 0
    delta = 0
    if player.equipment_area['进攻坐骑'].name is not None:
        delta -= 1
    while player_tmp.next != player:
        dis += 1
        if player_tmp.next.equipment_area['防御坐骑'].name is not None:
            res_next[player_tmp.next.idx] = max(dis + 1 - delta, 1)
        else:
            res_next[player_tmp.next.idx] = max(dis - delta, 1)
        player_tmp = player_tmp.next

    player_tmp = player
    dis = 0
    while player_tmp.pre != player:
        dis += 1
        if player_tmp.pre.equipment_area['防御坐骑'].name is not None:
            res_pre[player_tmp.pre.idx] = max(dis + 1 - delta, 1)
        else:
            res_pre[player_tmp.pre.idx] = max(dis - delta, 1)
        player_tmp = player_tmp.pre

    for key in res_next.keys():
        res[player_list[key - 1]] = min(res_next[key], res_pre[key])

    return res


def print_player(player):  # 打印玩家信息
    hengzhi_dic = {True: '是', False: '否'}
    print('{}号位'.format(player.idx))
    print('武将为:{}'.format(player.commander.name))
    print('当前体力值为:{}/{}'.format(player.current_HP, player.max_HP))
    print('是否被横置:{}'.format(hengzhi_dic[player.hengzhi]))
    print('装备区有武器牌:{},防具牌:{}, 进攻坐骑:{}, 防御坐骑:{},宝物:{}'.format(player.equipment_area['武器'].name,
                                                             player.equipment_area['防具'].name,
                                                             player.equipment_area['进攻坐骑'].name,
                                                             player.equipment_area['防御坐骑'].name,
                                                             player.equipment_area['宝物'].name))


# 检查是否满足游戏胜利条件
def check_vic(dead_player: player, player_list) -> int:
    '''

    dead_player: 死亡角色
    return: 1: 主公和忠臣获胜
            2: 反贼获胜
            3: 内奸获胜
    '''
    if dead_player.identity == '主公':
        if len(player_list) == 1 and player_list[0].identity == '内奸':
            return 3
        else:
            return 2
    elif (dead_player.identity != '主公') and ('反贼' not in [player.identity for player in player_list]) and (
            '内奸' not in [player.identity for player in player_list]):
        return 1
    else:
        return 0
