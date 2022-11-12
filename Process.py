# 流程
from Card import *
from Other import *
import numpy as np

# 游戏流程
game_process = ['bef_huihe_start',  # 回合开始前
                'when_huihe_start',  # 回合开始时
                'when_prepare_start',  # 准备阶段开始时
                # 'prepare' 准备阶段(暂时没有作用)
                # 'when_prepare_end' 准备阶段结束时(暂时没有作用)
                'bef_pandin_af_prepare',  # 准备阶段与判定阶段间
                'when_pandin_start',  # 判断阶段开始时
                'pandin',  # 判定阶段
                # 'when_pandin_end' 判定阶段结束时(暂时没有作用)
                'bef_getcard_af_pandin',  # 判定阶段与摸牌阶段间
                'when_getcard_start',  # 摸牌阶段开始时
                'getcard',  # 摸牌阶段
                'when_getcard_end',  # 摸牌阶段结束时
                'bef_usecard_af_getcard',  # 摸牌阶段与出牌阶段间
                'when_usecard_start',  # 出牌阶段开始时
                'usecard',  # 出牌阶段
                'when_usecard_end'  # 出牌阶段结束时
                'bef_leftcard_af_usecard',  # 出牌阶段与弃牌阶段间
                'when_leftcard_start',  # 弃牌阶段开始时
                'leftcard',  # 弃牌阶段
                'when_leftcard_end',  # 弃牌阶段结束时
                # 'bef_end_af_leftcard' 弃牌阶段与结束阶段间(暂时没有作用)
                'when_end',  # 结束阶段开始时
                # 'end' 结束阶段(暂时没有作用)
                # 'when_end_end' 结束阶段结束时(暂时没有作用)
                'when_huihe_end',  # 回合结束时
                'af_huihe_end',  # 回合结束后
                ]
# 使用牌的流程
use_card_process = ['af_state_use',  # 声明使用牌后
                    'af_choose_target',  # 选择目标后
                    'when_use',  # 使用时
                    'when_specified_target',  # 指定目标时
                    'when_targeted',  # 成为目标时
                    'af_specified_target',  # 指定目标后
                    'af_targeted',  # 成为目标后
                    'when_pre_clear_end',  # 使用结算准备结算结束时
                    ]
# 使用结算的流程
use_clear_process = ['when_clear_start',  # 使用结算开始时
                     'bef_effect',  # 生效前
                     'when_effect',  # 生效时
                     'af_effect',  # 生效后
                     # 'when_clear_end' 使用结算结束时(暂时没有作用)
                     'af_clear_end',  # 使用结算结束后
                     ]

# 判定流程
pandin_process = ['when_pandin',  # 判定时
                  # 'af_pandin',  成为判定牌后(暂时没有作用)
                  'bef_pandin_result',  # 判定结果确定前
                  'af_pandin_result',  # 判定结果确定后
                  ]

# 伤害流程
damage_process = ['bef_damage_jiesuan',  # 伤害结算开始前
                  'when_damage',  # 造成伤害时
                  'when_hurt',  # 受到伤害时
                  'af_damage',  # 造成伤害后
                  'af_hurt',  # 受到伤害后
                  ]

# 扣减体力流程
deducted_HP_process = ['bef_deducted_HP',  # 扣减体力前
                       'when_deducted_HP',  # 扣减体力时
                       'af_deducted_HP',  # 扣减体力后
                       ]
# 濒死流程
binsi_process = ['when_into_binsi',  # 进入濒死状态时
                 'af_into_binsi',  # 进入濒死状态后
                 'when_binsi',  # 处于濒死状态时
                 ]


def Game_Process(player: Player,
                 player_list: list,
                 get_card_heap: Get_Card_Heap,
                 left_card_heap: Left_Card_Heap):
    """

    :param player: 当前回合角色
    :param player_list: 所以玩家列表
    :param get_card_heap: 摸牌堆
    :param left_card_heap: 弃牌堆

    """
    game_process_tmp = game_process.copy()
    isEnd = 0
    for time in game_process_tmp:
        # check_skill(time) 检查是否有武将技能发动
        if time == 'pandin':
            while len(player.pandin_area) > 0:
                pandin_name = player.pandin_area[-1].name
                print('即将判定{}'.format(pandin_name))
                # ask_wuxiekeji()
                result = Pandin_Process(player, get_card_heap, left_card_heap)
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
                            binsi(player, player, len(player_list))
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
                Use_Card_process(player.HandCards_area[idx], player, player_list, get_card_heap, left_card_heap)
                print('当前场上所有角色信息为:')
                for i in player_list:
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
def Pandin_Process(player, get_card_heap, left_card_heap):
    """

    player: 判定角色
    get_card_heap: 摸牌堆
    left_card_heap: 弃牌堆
    return: 判定结果

    """
    for time in pandin_process:
        # check_skill(time) 检查是否有武将技能发动
        if time == 'when_pandin':
            res_card = get_card_heap.get_card(1)
            res = [res_card[0].color, res_card[0].point]
            print('判定结果为:',res)
            left_card_heap.card_list.append(res_card)
    return res


# 使用牌流程
def Use_Card_process(card: card,
                     player: player,
                     player_list,
                     get_card_heap: Get_Card_Heap,
                     left_card_heap: Left_Card_Heap,
                     target_card: card = None, ):
    """

    card: 被使用的牌
    player: 使用牌的玩家
    get_card_heap: 摸牌堆
    left_card_heap: 弃牌堆
    target_card: 如果被使用的牌的目标是牌,则输入目标牌

    """

    # 声明使用牌后
    # check_skill()
    if '杀' in card.name:
        if player.use_sha_count == 1:
            print('不能再使用杀了')
            return
        if player.equipment_area['weapon'].name is not None:  # 如果未装备武器牌,则杀的距离为1
            card.dis = player.equipment_area['weapon'].dis
    elif card.name == '桃':
        if player.current_HP == player.max_HP:  # 若当前玩家体力值等于体力上限,则无法指定自己为目标
            return

    # 选择目标
    # check_skill()
    if card.target is None:
        legal_target = [k for k, v in cal_dis(player, player_list).items() if v <= card.dis]
        if len(legal_target) > 0:
            print('你能选择的目标有:')
            for target in legal_target:
                print_player(target)
            target_idx = eval(input('请选择目标:'))
            target = [i for i in legal_target if i.idx == target_idx]
        else:
            return
    elif (np.array([player.current_HP for player in player_list]).any() <= 0) and ('binsi_player' in card.target):
        binsi_player = [player for player in player_list if player.current_HP <= 0]
        target = binsi_player
    elif 'player' in card.target:
        target = [player]
    elif card.target == '杀':
        target = target_card

    # 使用时
    # check_skill()
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

    # 指定目标时
    # for t in target:
    # checkskill

    # 成为目标时
    # for t in target:
    # checkskill

    # 指定目标后
    # for t in target:
    # checkskill

    # 成为目标后
    # for t in target:
    # checkskill

    # 使用结算准备结算结束时
    if type(card) is weapon_card:
        if player.equipment_area['weapon'] != '':
            left_card_heap.card_list.append(player.equipment_area['weapon'])
        player.equipment_area['weapon'] = card
    elif type(card) is armour_card:
        if player.equipment_area['armour'] != '':
            left_card_heap.card_list.append(player.equipment_area['armour'])
        player.equipment_area['armour'] = card
    elif type(card) is attack_horse_card:
        if player.equipment_area['horse-1'] != '':
            left_card_heap.card_list.append(player.equipment_area['horse-1'])
        player.equipment_area['horse-1'] = card
    elif type(card) is defense_horse_card:
        if player.equipment_area['horse+1'] != '':
            left_card_heap.card_list.append(player.equipment_area['horse+1'])
        player.equipment_area['horse+1'] = card

    return Use_Clear_Process(player, player_list, card, target, get_card_heap, left_card_heap)


# 使用生效的流程
def Use_Clear_Process(player: player,player_list, card: Card, target: player, get_card_heap, left_card_heap):
    for i in range(len(target)):
        # 首先判定此牌对当前目标是否有效,若无效,则不会生成'使用结算开始时'时机

        # 使用结算开始时
        # checkskill()
        # 然后若此牌对当前目标无效,则不会生成'生效前'时机

        # 生效前
        if '杀' in card.name:
            print('{}号位手牌为{}'.format(target[i].idx,
                                     [[card.name, card.color, card.point] for card in target[i].HandCards_area]))
            shan_idx = target[i].HandCards_area[eval(input('{}号位是否出闪, 0表示不出闪, i表示出第i张牌'.format(target[i].idx))) - 1]
            shan = 0
            if shan_idx.name == '闪':
                shan = Use_Card_process(shan_idx, target[i], player_list, get_card_heap, left_card_heap, card)
            if shan:
                return
                # 生效时

        # 生效后
        if '杀' in card.name:
            target[i].current_HP -= 1 + player.jiu
        elif card.name == '闪':
            return 1
        elif card.name == '桃' and target[i] == player:
            player.current_HP += 1
            player.max_HandCards = player.current_HP
        elif card.name == '桃' and target.current_HP <= 0:
            target[i].current_HP += 1
            target[i].max_HandCards += 1

    left_card_heap.card_list.append(card)


# 伤害流程
def Damage_Process(source: player or card,
                   hurt_player: player,
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
def Deducted_HP_Process(hurt_player: player, damage_num):
    for time in deducted_HP_process:
        # check_skill(time)
        if time == 'when_deducted_HP':
            hurt_player.current_HP -= damage_num
            hurt_player.max_HandCards = hurt_player.current_HP
        if time == 'af_deducted_HP':
            if hurt_player.current_HP <= 0:
                binsi(hurt_player)


# 濒死流程
def binsi(binsi_player: player, current_player: player, player_num):
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
def death_process(dead_player: player):
    # 确认身份前
    global player_list
    print('死亡角色身份为:{}'.format(dead_player.identity))
    player_list_tmp = player_list
    dead_player.pre.next = dead_player.next
    dead_player.next.pre = dead_player.pre
    del player_list[player_list.index(dead_player)]
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
    player_list = player_list_tmp
