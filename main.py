import numpy as np

from Card import *
from Commander import Commander
from Player import *
from Process import *


# 游戏流程
def Game_Process(player):
    game_process_tmp = game_process.copy()
    isEnd = 0
    for time in game_process_tmp:
        # check_skill(time) 检查是否有武将技能发动
        if time == 'pandin':
            while len(player.pandin_area) > 0:
                pandin_name = player.pandin_area[-1].name
                print('即将判定{}'.format(pandin_name))
                # ask_wuxiekeji()
                result = Pandin_Process(player)
                if pandin_name == '兵粮寸断':
                    if result[0] != '梅花':
                        del game_process_tmp[7:10]
                elif pandin_name == '乐不思蜀':
                    if result[0] != '红桃':
                        del game_process_tmp[11:14]
                elif pandin_name == '闪电':
                    if (result[0] == '黑桃') & (result[1] >= 2) & (result[1] <= 9):
                        player.current_HP -= 3
                        if player.current_HP <= 0:
                            binsi(player)
                left_card_heap.card_list.append(player.pandin_area[-1])
                del player.pandin_area[-1]
        elif time == 'getcard':
            for card in get_card_heap.get_card(2):
                player.HandCards_area.append(card)

        elif time == 'usecard':
            while 1:
                print('你当前体力值为:{}/{}'.format(player.current_HP, player.max_HP))
                print('你的势力为:{}'.format(player.identity))
                handcards = [[card.name, card.color, card.point] for card in player.HandCards_area]
                print('你当前有的手牌为:{}'.format(handcards))
                isEnd = eval(input('是否结束出牌阶段(0表示否, 1表示是):'))
                if isEnd != 0:
                    player.jiu = 0
                    player.use_sha_count = 0
                    player.use_jiu_count = 0
                    break
                idx = eval(input('请选择你要使用第几张牌:')) - 1
                Use_Card_process(player.HandCards_area[idx], player)
                print('当前场上所有角色信息为:')
                for i in players:
                    print_player(i)
        elif time == 'leftcard':
            while len(player.HandCards_area) > player.max_HandCards:
                print('你当前有的手牌为:{}'.format([[card.name, card.color, card.point] for card in player.HandCards_area]))
                need_left = len(player.HandCards_area) - player.max_HandCards
                left_cards = eval(input('请弃置{}张牌'.format(need_left)))
                if type(left_cards) == int:
                    left_cards = list(map(int, str(left_cards)))
                else:
                    left_cards = list(left_cards)
                for i in range(len(left_cards)):
                    left_card_heap.card_list.append(player.HandCards_area[left_cards[-1] - 1])
                    del player.HandCards_area[left_cards[-1] - 1], left_cards[-1]


# 判定流程
def Pandin_Process(player):
    for time in pandin_process:
        # check_skill(time) 检查是否有武将技能发动
        if time == 'when_pandin':
            res_card = get_card_heap.get_card(1)
            res = [res_card[0].color, res_card[0].point]
            print(res)
            left_card_heap.card_list.append(res_card)
    return res


# 使用牌流程
def Use_Card_process(card: card, player: Player):
    time_id = 0
    while time_id < len(use_card_process):
        time = use_card_process[time_id]
        # check_skill(time)
        # isEmptyTarget()
        if time == 'af_state_use':
            if '杀' in card.name:
                if player.use_sha_count == 1:
                    print('不能再使用杀了')
                    break
                if player.equipment_area['weapon'].name is not None:  # 如果未装备武器牌,则杀的距离为1
                    card.dis = player.equipment_area['weapon'].dis
            elif card.name == '桃':
                if player.current_HP == player.max_HP:  # 若当前玩家体力值等于体力上限,则无法指定自己为目标
                    break
            # 　指定目标
            if card.target is None:
                legal_target = [k for k, v in cal_dis(player).items() if v <= card.dis]
                if len(legal_target) > 0:
                    print('你能选择的目标有:')
                    for target in legal_target:
                        print_player(target)
                    target = players[eval(input('请选择目标:')) - 1]
                else:
                    break
            elif (np.array([player.current_HP for player in players]).any() <= 0) and ('binsi_player' in card.target):
                binsi_player = [player for player in players if player.current_HP <= 0][0]
                target = binsi_player
            elif 'player' in card.target:
                target = player
        elif time == 'when_use':
            if '杀' in card.name:
                player.use_sha_count += 1
            if card.name == '酒':
                if player.current_HP > 0:
                    player.use_jiu_count += 1
                    player.jiu += 1
                else:
                    player.current_HP += 1
                    player.max_HandCards += 1
            del player.HandCards_area[player.HandCards_area.index(card)]
        elif time == 'bef_effect':
            if '杀' in card.name:
                print('{}号位手牌为{}'.format(target.idx,
                                         [[card.name, card.color, card.point] for card in target.HandCards_area]))
                shan = eval(input('{}号位是否出闪, 0表示不出闪, i表示出第i张闪'.format(target.idx)))

                if shan:
                    left_card_heap.card_list.append(target.HandCards_area[shan - 1])
                    del target.HandCards_area[shan - 1]
                    time_id = len(use_card_process)
        elif time == 'when_pre_clear_end':
            if type(card) is weapon_card:
                if player.equipment_area['weapon'] != '':
                    left_card_heap.card_list.append(player.equipment_area['weapon'])
                player.equipment_area['weapon'] = card
                break
            elif type(card) is armour_card:
                if player.equipment_area['armour'] != '':
                    left_card_heap.card_list.append(player.equipment_area['armour'])
                player.equipment_area['armour'] = card
                break
            elif type(card) is attack_horse_card:
                if player.equipment_area['horse-1'] != '':
                    left_card_heap.card_list.append(player.equipment_area['horse-1'])
                player.equipment_area['horse-1'] = card
                break
            elif type(card) is defense_horse_card:
                if player.equipment_area['horse+1'] != '':
                    left_card_heap.card_list.append(player.equipment_area['horse+1'])
                player.equipment_area['horse+1'] = card
                break

        elif time == 'af_effect':
            if '杀' in card.name:
                if not (shan):
                    Damage_Process(player, target, 1 + player.jiu, card.is_shuxing)
            if card.name == '桃' and target == player:
                player.current_HP += 1
                player.max_HandCards = player.current_HP
            elif card.name == '桃' and target == binsi_player:
                binsi_player.current_HP += 1
                binsi_player.max_HandCards += 1

        time_id += 1
    left_card_heap.card_list.append(card)


# 伤害流程
def Damage_Process(source: Player or card,
                   hurt_player: Player,
                   damage_num: int,
                   is_shuxing: bool):
    """
    
    source: 来源
    hurted_player: 受到伤害的角色
    damage_num: 伤害值
    is_shuxing: 是否为属性伤害
 
    """

    for time in damage_process:
        # check_skill(time)
        if time == 'when_hurt':
            if hurt_player.hengzhi and is_shuxing:  # 若受伤角色处于横置状态且受到的伤害为属性伤害,则其重置
                hurt_player.hengzhi = False
            Deducted_HP_Process(hurt_player, damage_num)
            # tiesuolianhuan()
            # 之后若受伤角色受到的是不为连环伤害的属性伤害且有其他角色处于横置状态,则触发铁索连环


# 扣减体力流程(造成伤害事件)
def Deducted_HP_Process(hurt_player: Player, damage_num):
    for time in deducted_HP_process:
        # check_skill(time)
        if time == 'when_deducted_HP':
            hurt_player.current_HP -= damage_num
            hurt_player.max_HandCards = hurt_player.current_HP
        if time == 'af_deducted_HP':
            if hurt_player.current_HP <= 0:
                binsi(hurt_player)


# 濒死流程
def binsi(binsi_player: Player):
    for time in binsi_process:
        # check_skill
        if time == 'when_binsi':
            current_player_tmp = current_player
            count = 0
            while (binsi_player.current_HP <= 0) and (count < player_num):
                print('{}濒死,需要{}颗桃'.format(binsi_player.commander.name, 1 - binsi_player.current_HP))
                if current_player_tmp == binsi_player:
                    print('{}号位手牌为:{}'.format(binsi_player.idx, [[card.name, card.color, card.point] for card in
                                                                 binsi_player.HandCards_area]))
                    print('{}号位是否使用桃或酒(0表示不使用,i表示使用第i张牌):'.format(binsi_player.idx))
                else:
                    print('{}号位手牌为:{}'.format(current_player_tmp.idx, [[card.name, card.color, card.point] for card in
                                                                       current_player_tmp.HandCards_area]))
                    print('{}号位是否使用桃(0表示不使用,i表示使用第i张牌):'.format(current_player_tmp.idx))
                tao = eval(input())
                if tao > 0:
                    Use_Card_process(current_player_tmp.HandCards_area[tao], current_player_tmp)
                count += 1
                current_player_tmp = current_player_tmp.next
            if binsi_player.current_HP <= 0:
                death_process(binsi_player)


# 死亡流程
def death_process(dead_player: Player):
    # 确认身份前
    global players
    print('死亡角色身份为:{}'.format(dead_player.identity))
    players_tmp = players
    del players[players.index(dead_player)]
    vic = check_vic(dead_player)
    if vic != 0:
        if vic == 1:
            print('主公阵营获胜\n游戏结束')
            exit(0)
        elif vic == 2:
            print('反贼阵营获胜\n游戏结束')
            exit(0)
        elif vic == 3:
            print('内奸获胜\n游戏结束')
            exit(0)
    # 死亡时
    # check_skill('when_dead')
    players = players_tmp


def cal_dis(player):  # 计算距离
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
    while player_tmp.pre != player:
        dis += 1
        if player_tmp.pre.equipment_area['horse+1'].name is not None:
            res_pre[player_tmp.pre.idx] = dis + 1
        else:
            res_pre[player_tmp.pre.idx] = dis
        player_tmp = player_tmp.pre

    for key in res_next.keys():
        res[players[key - 1]] = min(res_next[key], res_pre[key])

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
def check_vic(dead_player: Player) -> int:
    '''

    dead_player: 死亡角色
    return: 1: 主公和忠臣获胜
            2: 反贼获胜
            3: 内奸获胜
    '''
    if dead_player.identity == '主公':
        if len(players) == 1 and players[0].identity == '内奸':
            return 3
        else:
            return 2
    elif (dead_player.identity != '主公') and ('反贼' not in [player.identity for player in players]) and (
            '内奸' not in [player.identity for player in players]):
        return 1
    else:
        return 0


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
    left_card_heap = left_card_heap()

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
            Game_Process(player)

        round += 1
