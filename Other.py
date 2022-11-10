from Player import *


def cal_dis(player, player_list):  # 计算距离
    res_next = {}
    res_pre = {}
    res = {}
    player_tmp = player
    dis = 0
    if player.equipment_area['horse-1'].name is not None:
        dis -= 1
    while player_tmp.next != player:
        dis += 1
        if player_tmp.next.equipment_area['horse+1'].name is not None:
            res_next[player_tmp.next.idx] = dis + 1
        else:
            res_next[player_tmp.next.idx] = dis
        player_tmp = player_tmp.next

    player_tmp = player
    dis = 0
    if player.equipment_area['horse-1'].name is not None:
        dis -= 1
    while player_tmp.pre != player:
        dis += 1
        if player_tmp.pre.equipment_area['horse+1'].name is not None:
            res_pre[player_tmp.pre.idx] = dis + 1
        else:
            res_pre[player_tmp.pre.idx] = dis
        player_tmp = player_tmp.pre

    for key in res_next.keys():
        res[player_list[key - 1]] = min(res_next[key], res_pre[key])

    return res


def print_player(player):  # 打印玩家信息
    print('{}号位'.format(player.idx))
    print('武将为:{}'.format(player.commander.name))
    print('当前体力值为:{}/{}'.format(player.current_HP, player.max_HP))
    print('装备区有武器牌:{},防具牌:{}, 进攻坐骑:{}, 防御坐骑:{}'.format(player.equipment_area['weapon'].name,
                                                       player.equipment_area['armour'].name,
                                                       player.equipment_area['horse-1'].name,
                                                       player.equipment_area['horse+1'].name))


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