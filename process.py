# process.py

# 流程
from typing import List

import items
from card import Card, BasicCard, YanshiJinnangCard, WeaponCard, ArmourCard, AttackHorseCard, DefenseHorseCard, \
    TreasureCard
from other import *

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
                'when_usecard_end',  # 出牌阶段结束时
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
wuxie_count = 0


def Game_Process(player: Player, Items: items.Items):
    """

    player: 当前回合角色

    """
    game_process_tmp = game_process.copy()
    isend = 0
    for time in game_process_tmp:
        # check_skill(time) 检查是否有武将技能发动
        if time == 'pandin':
            while len(player.pandin_area) > 0:
                card = player.pandin_area[-1]
                pandin_name = card.name
                print('即将判定{}'.format(pandin_name))
                if wuxie(Items, card, wuxie_count):
                    result = Pandin_Process(player, Items)
                    if pandin_name == '兵粮寸断':
                        if result[0] != '梅花':
                            game_process_tmp.remove('when_getcard_start')
                            game_process_tmp.remove('getcard')
                            game_process_tmp.remove('when_getcard_end')
                        Items.LeftCardHeap.card_list.append(card)
                        del card
                    elif pandin_name == '乐不思蜀':
                        if result[0] != '红桃':
                            game_process_tmp.remove('when_usecard_start')
                            game_process_tmp.remove('usecard')
                            game_process_tmp.remove('when_usecard_end')
                        Items.LeftCardHeap.card_list.append(card)
                        del card
                    elif pandin_name == '闪电':
                        if (result[0] == '黑桃') & (result[1] >= 2) & (result[1] <= 9):
                            Damage_Process(None, player.pandin_area[-1], player, player, 3, True, Items)
                        else:
                            card.target = player.next
                            player.next.pandin_area.append(card)
                            del card
                else:
                    if pandin_name == '闪电':
                        card.target = player.next
                        player.next.pandin_area.append(card)
                        del card
                    else:
                        Items.LeftCardHeap.card_list.append(card)
                        del card
        elif time == 'getcard':
            for card in Items.GetCardHeap.get_card(2, Items.LeftCardHeap):
                player.HandCards_area.append(card)

        elif time == 'usecard':
            while 1:
                print('你当前体力值为:{}/{}'.format(player.current_HP, player.max_HP))
                print('你的势力为:{}'.format(player.identity))
                print('你的技能为：{}'.format([skill.name for skill in player.skills]))
                print('装备区有武器牌:{},防具牌:{}, 进攻坐骑:{}, 防御坐骑:{},宝物:{}'.format(
                    player.equipment_area['武器'].name,
                    player.equipment_area['防具'].name,
                    player.equipment_area['进攻坐骑'].name,
                    player.equipment_area['防御坐骑'].name,
                    player.equipment_area['宝物'].name))
                handcards = [[card.name, card.color, card.point] for card in player.HandCards_area]
                print('你当前有的手牌为:{}'.format(handcards))
                while 1:
                    isend = input('是否结束出牌阶段(0表示否, 1表示是):')
                    try:
                        isend = eval(isend)
                    except SyntaxError and NameError:
                        continue
                    break
                if isend == 1:
                    player.jiu = 0
                    player.use_sha_count = 0
                    player.use_jiu_count = 0
                    break
                elif isend == 0:
                    while 1:
                        idx = eval(input('请选择你要使用第几张牌:')) - 1
                        try:
                            card = player.HandCards_area[idx]
                        except IndexError:
                            continue
                        if card.name == '铁索连环':
                            player_input = eval(input('请选择重铸或使用(0表示重铸,1表示使用):'))
                            if not player_input:
                                Items.LeftCardHeap.card_list.append(card)
                                player.HandCards_area.remove(card)
                                player.HandCards_area.append(Items.GetCardHeap.get_card(1, Items.LeftCardHeap)[0])
                                break
                        use_card_process(card, player, Items)
                        break
                    print('当前场上所有角色信息为:')
                    for i in Items.PlayerList:
                        PrintPlayer(i)
        elif time == 'leftcard':
            while len(player.HandCards_area) > player.max_HandCards:
                print('你当前有的手牌为:{}'.format(
                    [[card.name, card.color, card.point] for card in player.HandCards_area]))
                need_left = len(player.HandCards_area) - player.max_HandCards
                left_cards = eval(input('请弃置{}张牌'.format(need_left)))
                if type(left_cards) == int:
                    left_cards = list(map(int, str(left_cards)))
                else:
                    left_cards = list(left_cards)
                for i in range(len(left_cards)):
                    Items.LeftCardHeap.card_list.append(player.HandCards_area[left_cards[-1] - 1])
                    del player.HandCards_area[left_cards[-1] - 1], left_cards[-1]


# 判定流程
def Pandin_Process(player, Items):
    """

    player: 判定角色
    Items: 场上状态
    return: 判定结果

    """
    res = []
    for time in pandin_process:
        # check_skill(time) 检查是否有武将技能发动
        if time == 'when_pandin':
            res_card = Items.GetCardHeap.get_card(1, Items.LeftCardHeap)
            res = [res_card[0].color, res_card[0].point]
            print('判定结果为:', res)
            Items.LeftCardHeap.card_list.append(res_card)
    return res


# 使用牌流程
def use_card_process(c: Card,
                     player: Player,
                     Items: items.Items):
    """

    card: 被使用的牌
    player: 使用牌的玩家
    Items: 场上状态

    """

    # 声明使用牌后
    # check_skill()

    if '杀' in c.name and isinstance(c, BasicCard):
        if player.use_sha_count == 1:
            print('不能再使用杀了')
            return
        if player.equipment_area['武器'].name is not None:  # 如果未装备武器牌,则杀的距离为1
            c.dis = player.equipment_area['武器'].dis
    elif c.name == '酒':
        if player.use_jiu_count == 1:
            print('不能再使用酒了')
            return
    elif c.name == '桃':
        if player.current_HP == player.max_HP:  # 若当前玩家体力值等于体力上限,则无法指定自己为目标
            return
    elif c.name == '闪' and c.target == ['杀']:
        return
    # 选择目标
    # check_skill()
    target = []
    target_2 = []
    if isinstance(c.target, Player) or isinstance(c.target, Card):
        target.append(c.target)
    elif 'another player' in c.target:
        legal_target: List[player] = [p for p in Items.PlayerList if p != player]
        if isinstance(c.target, YanshiJinnangCard):  # 若目标判定区内有同名牌,则不能成为合法目标
            for t in legal_target:
                if len(t.pandin_area) > 0:
                    if c.name in [k.name for k in t.pandin_area]:
                        legal_target.remove(t)
        if '杀' in c.name and isinstance(c.target, BasicCard):
            legal_target = [k for k, v in cal_dis(player, Items.PlayerList).items() if v <= c.dis]
        elif c.name == '借刀杀人':
            legal_target = []
            for p in Items.PlayerList:  # 只有有武器的角色能成为合法目标
                if p.equipment_area['武器'].name is not None and p is not player:
                    legal_target.append(p)
        elif c.name == '顺手牵羊':  # 过河拆桥和顺手牵羊的目标为区域里有牌的角色
            legal_target = [k for k, v in cal_dis(player, Items.PlayerList).items() if v <= c.dis]
            for t in legal_target:
                if isAreaEmpty(t):
                    legal_target.remove(t)
        elif c.name == '过河拆桥':
            for t in legal_target:
                if isAreaEmpty(t):
                    legal_target.remove(t)
        if len(legal_target) > 0:  # 若合法目标列表为空,则结束使用流程
            print('你能选择的目标有:')
            for t in legal_target:
                PrintPlayer(t)
            while 1:
                target_idx = list(input('请选择目标:'))
                for i in target_idx:
                    for p in legal_target:
                        if p.idx == int(i):
                            target.append(p)
                if len(target) > 0:
                    break
            if c.name == '借刀杀人':
                target_2: List[player] = [k for k, v in cal_dis(target[0], Items.PlayerList).items()
                                          # 被杀目标必须在借刀杀人的目标的攻击距离之内
                                          if v <= target[0].equipment_area['武器'].dis]
                print('你能选择的目标有:')
                for t in target_2:
                    PrintPlayer(t)
                while 1:
                    target_idx = input('请选择被杀目标:')
                    try:
                        target_idx = eval(target_idx)
                    except SyntaxError:
                        continue
                    target_2 = [i for i in target_2 if i.idx == target_idx]
                    if len(target_2) > 0:
                        break
        else:
            return
    elif (np.array([player.current_HP for player in Items.PlayerList]).any() <= 0) and ('binsi_player' in c.target):
        binsi_player = [player for player in Items.PlayerList if player.current_HP <= 0]
        target = binsi_player
    elif 'player' in c.target:
        target = [player]
    elif 'all players' in c.target:
        tmp = player
        for i in range(len(Items.PlayerList)):
            target.append(tmp)
            tmp = tmp.next
    elif 'all other players' in c.target:
        tmp = player
        while tmp.next != player:
            target.append(tmp.next)
            tmp = tmp.next
    elif c.name == '火攻':  # 火攻的目标是一名有手牌的角色
        legal_target = []
        for p in Items.PlayerList:
            if len(p.HandCards_area) > 0:
                legal_target.append(p)
        print('你能选择的目标有:')
        for t in legal_target:
            PrintPlayer(t)
        while 1:
            target_idx = list(input('请选择目标:'))
            for i in target_idx:
                for p in legal_target:
                    if p.idx == int(i):
                        target.append(p)
            if len(target) > 0:
                break
    elif c.name == '铁索连环':
        print('你能选择的目标有:')
        for t in Items.PlayerList:
            PrintPlayer(t)
        while 1:
            player_input = input('请选择1至2个目标:').split(',')
            for i in player_input:
                for p in Items.PlayerList:
                    if p.idx == int(i):
                        target.append(p)
            if len(target) > 0:
                break

    # 使用时
    # check_skill()
    if '杀' in c.name:
        player.use_sha_count += 1
    if c.name == '酒':
        if player.current_HP > 0:
            player.use_jiu_count += 1
            player.jiu += 1
        else:
            player.current_HP += 1
            player.max_HandCards += 1
    player.HandCards_area.remove(c)
    Items.TmpCard.append(c)
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
    if c.name == '五谷丰登':
        for i in range(len(c.target)):
            Items.TmpCard.append(Items.GetCardHeap.get_card(1, Items.LeftCardHeap)[0])
        print([[card.name, card.color, card.point] for card in Items.TmpCard[-len(c.target):]])
        Use_Clear_Process(player, c, target, Items)
        return
    if isinstance(c.target, YanshiJinnangCard):
        Items.TmpCard.remove(0)
        target[0].pandin_area.append(c)
        return
    if isinstance(c.target, WeaponCard):
        if player.equipment_area['武器'] is not None:
            Items.LeftCardHeap.card_list.append(player.equipment_area['武器'])
        Items.TmpCard.remove(c)
        player.equipment_area['武器'] = c
        c.player = player
    elif isinstance(c.target, ArmourCard):
        if player.equipment_area['防具'] is not None:
            Items.LeftCardHeap.card_list.append(player.equipment_area['防具'])
        Items.TmpCard.remove(c)
        player.equipment_area['防具'] = c
        c.player = player
    elif isinstance(c.target, AttackHorseCard):
        if player.equipment_area['进攻坐骑'] is not None:
            Items.LeftCardHeap.card_list.append(player.equipment_area['进攻坐骑'])
        Items.TmpCard.remove(c)
        player.equipment_area['进攻坐骑'] = c
        c.player = player
    elif isinstance(c.target, DefenseHorseCard):
        if player.equipment_area['防御坐骑'] is not None:
            Items.LeftCardHeap.card_list.append(player.equipment_area['防御坐骑'])
        Items.TmpCard.remove(c)
        player.equipment_area['防御坐骑'] = c
        c.player = player
    elif isinstance(c.target, TreasureCard):
        if player.equipment_area['宝物'] is not None:
            Items.LeftCardHeap.card_list.append(player.equipment_area['宝物'])
        Items.TmpCard.remove(c)
        player.equipment_area['宝物'] = c
        c.player = player
    if len(target_2) > 0:
        return Use_Clear_Process(player, c, target, target_2[0])
    return Use_Clear_Process(player, c, target, Items)


# 使用生效的流程
def Use_Clear_Process(player: Player,
                      ca: Card,
                      target: List[Player],
                      Items: items.Items,
                      b_target: Player = None):
    """

    player: 使用玩家
    ca: 被使用的牌
    target: 目标列表
    Items: 场上状态
    b_target: 借刀杀人指定的被杀目标

    """
    global wuxie_count

    for i in range(len(target)):
        # 首先判定此牌对当前目标是否有效,若无效,则不会生成'使用结算开始时'时机

        # 使用结算开始时
        # checkskill()
        # 然后若此牌对当前目标无效,则不会生成'生效前'时机
        if ca.name == '桃园结义' and target[i].current_HP == target[i].max_HP:
            continue
        # 生效前
        if isinstance(ca.target, Card):
            if wuxie(Items, ca, wuxie_count) != 0:
                continue
            '''
            for j in Items.PlayerList:
                if '无懈可击' not in [w.name for w in j.HandCards_area]:
                    continue
                print('{}号位手牌为{}'.format(j.idx, [[card.name, card.color, card.point] for card in j.HandCards_area]))
                wuxie = eval(input('{}号位是否使用无懈可击, 0表示不使用无懈可击, i表示使用第i张牌'.format(j.idx)))
                if not wuxie:
                    continue
                wuxie = j.HandCards_area[wuxie - 1]
                if wuxie.name == '无懈可击':
                    wuxie.target = card
                    wuxie_count += Use_Card_process(wuxie, j, Items)
                    if wuxie_count:
                        return 0
            '''
        if '杀' in ca.name and isinstance(ca.target, BasicCard):
            ca: BasicCard = ca
            shan = 0
            if '闪' not in [card.name for card in target[i].HandCards_area]:
                pass
            else:
                for k in range(ca.need_shan):
                    print('{}号位手牌为{}'.format(target[i].idx,
                                                  [[card.name, card.color, card.point] for card in
                                                   target[i].HandCards_area]))

                    player_input = eval(
                        input('{}号位是否使用闪, 0表示不使用闪, i表示使用第i张牌'.format(target[i].idx)))
                    if not player_input:
                        break
                    shan_idx = target[i].HandCards_area[player_input - 1]

                    if shan_idx.name == '闪':
                        shan_idx.target = ca
                        use_card_process(shan_idx, target[i], Items)
                        shan += 1
                if shan == ca.need_shan:
                    # 被抵消后
                    return

        # 生效时

        # 生效后
        if '杀' in ca.name and isinstance(ca.target, BasicCard):
            Damage_Process(player, ca, player, target[i], 1 + player.jiu, ca.is_shuxing, Items)
        elif ca.name == '闪':
            return 1
        elif ca.name == '桃':
            target[i].current_HP += 1
            target[i].max_HandCards += 1
        elif ca.name == '无懈可击':
            return 1
        elif ca.name == '无中生有':
            for get_card in Items.GetCardHeap.get_card(2, Items.LeftCardHeap):
                player.HandCards_area.append(get_card)
        elif ca.name == '五谷丰登':
            idx = eval(input('{}号位请选择获得一张牌:'.format(target[i].idx)))
            target[i].HandCards_area.append(Items.TmpCard[idx])
            del Items.TmpCard[idx]
            print([[card.name, card.color, card.point] for card in Items.TmpCard])
        elif ca.name == '铁索连环':
            target[i].hengzhi = not target[i].hengzhi
        elif ca.name == '火攻':
            print('{}号位手牌为{}'.format(target[i].idx,
                                          [[card.name, card.color, card.point] for card in target[i].HandCards_area]))
            show_card: Card = target[i].HandCards_area[
                eval(input('{}号位请展示一张手牌'.format(target[i].idx))) - 1]
            print(show_card.name, show_card.color, show_card.point)
            print('{}号位手牌为{}'.format(target[i].idx,
                                          [[card.name, card.color, card.point] for card in player.HandCards_area]))
            left_card: int = eval(input('{}号位可以弃置一张与之花色相同的牌,对其造成1点火焰伤害:'.format(player.idx)))
            if not left_card:
                continue
            left_card: Card = player.HandCards_area[left_card - 1]
            if show_card.color == left_card.color:
                Items.LeftCardHeap.card_list.append(left_card)
                player.HandCards_area.remove(left_card)
                Damage_Process(player, ca, player, target[i], 1, True, Items)
        elif ca.name == '桃园结义':
            target[i].current_HP += 1
            target[i].max_HandCards += 1
        elif ca.name == '南蛮入侵':
            k = 0
            for card_name in [c.name for c in target[i].HandCards_area]:
                k += card_name.find('杀')
            if k == -len(target[i].HandCards_area):
                Damage_Process(player, ca, player, target[i], 1, False, Items)
            else:
                print('{}号位手牌为{}'.format(target[i].idx,
                                              [[card.name, card.color, card.point] for card in
                                               target[i].HandCards_area]))
                player_input = eval(input('{}号位是否打出杀, 0表示不出杀, i表示出第i张牌'.format(target[i].idx)))
                sha_idx = target[i].HandCards_area[player_input - 1]
                if '杀' in sha_idx.name:
                    continue
        elif ca.name == '万箭齐发':
            if '闪' not in [c.name for c in target[i].HandCards_area]:
                Damage_Process(player, ca, player, target[i], 1, False, Items)
            else:
                print('{}号位手牌为{}'.format(target[i].idx,
                                              [[card.name, card.color, card.point] for card in
                                               target[i].HandCards_area]))
                player_input = eval(input('{}号位是否打出闪, 0表示不出闪, i表示出第i张牌'.format(target[i].idx)))
                shan_idx = target[i].HandCards_area[player_input - 1]
                if shan_idx.name == '闪':
                    continue
        elif ca.name == '过河拆桥':
            left_card: Card = Card(None, None, None)
            while 1:
                print('目标有{}张手牌'.format(len(target[i].HandCards_area)))
                print('目标装备区有武器牌:{},防具牌:{}, 进攻坐骑:{}, 防御坐骑:{},宝物:{}'.format(
                    target[i].equipment_area['武器'].name,
                    target[i].equipment_area['防具'].name,
                    target[i].equipment_area['进攻坐骑'].name,
                    target[i].equipment_area['防御坐骑'].name,
                    target[i].equipment_area['宝物'].name))
                print('目标判定区有:{}'.format([card.name for card in target[i].pandin_area]))
                player_input = input('请选择一张牌弃置(i表示弃置第i张手牌,w表示弃置武器牌,a表示弃置防具牌,'
                                     'j表示弃置进攻坐骑,f表示弃置防御坐骑,t表示弃置宝物牌,l表示弃置乐不思蜀,'
                                     'b表示弃置兵粮寸断,s表示弃置闪电):')
                try:
                    player_input = eval(player_input)
                except NameError:
                    if player_input == 'w' and target[i].equipment_area['武器'].name is not None:
                        Items.LeftCardHeap.card_list.append(target[i].equipment_area['武器'])
                        target[i].equipment_area['武器'] = WeaponCard(None, None, None, None)
                        break
                    elif player_input == 'a' and target[i].equipment_area['防具'].name is not None:
                        Items.LeftCardHeap.card_list.append(target[i].equipment_area['防具'])
                        target[i].equipment_area['防具'] = ArmourCard(None, None, None)
                        break
                    elif player_input == '-1' and target[i].equipment_area['进攻坐骑'].name is not None:
                        Items.LeftCardHeap.card_list.append(target[i].equipment_area['进攻坐骑'])
                        target[i].equipment_area['进攻坐骑'] = DefenseHorseCard(None, None, None)
                        break
                    elif player_input == '+1' and target[i].equipment_area['防御坐骑'].name is not None:
                        Items.LeftCardHeap.card_list.append(target[i].equipment_area['防御坐骑'])
                        target[i].equipment_area['防御坐骑'] = AttackHorseCard(None, None, None)
                        break
                    elif player_input == 't' and target[i].equipment_area['宝物'].name is not None:
                        Items.LeftCardHeap.card_list.append(target[i].equipment_area['宝物'])
                        target[i].equipment_area['宝物'] = TreasureCard(None, None, None)
                        break
                    elif player_input == 'b' and '兵粮寸断' in [card.name for card in target[i].pandin_area]:
                        Items.LeftCardHeap.card_list.append(
                            [card for card in target[i].pandin_area if card.name == '兵粮寸断'][0])
                        target[i].pandin_area.remove(
                            [card for card in target[i].pandin_area if card.name == '兵粮寸断'][0])
                        break
                    elif player_input == 'l' and '乐不思蜀' in [card.name for card in target[i].pandin_area]:
                        Items.LeftCardHeap.card_list.append(
                            [card for card in target[i].pandin_area if card.name == '乐不思蜀'][0])
                        target[i].pandin_area.remove(
                            [card for card in target[i].pandin_area if card.name == '乐不思蜀'][0])
                        break
                    elif player_input == 's' and '闪电' in [card.name for card in target[i].pandin_area]:
                        Items.LeftCardHeap.card_list.append(
                            [card for card in target[i].pandin_area if card.name == '闪电'][0])
                        target[i].pandin_area.remove([card for card in target[i].pandin_area if card.name == '闪电'][0])
                        break

                try:
                    left_card = target[i].HandCards_area[player_input - 1]
                    break
                except (IndexError, TypeError):
                    continue
            if type(player_input) == int:
                Items.LeftCardHeap.card_list.append(left_card)
                del target[i].HandCards_area[player_input - 1]
            print('被弃置的牌为:{}'.format(Items.LeftCardHeap.card_list[-1].name))
        elif ca.name == '顺手牵羊':
            get_card: Card = Card(None, None, None)
            while 1:
                print('目标有{}张手牌'.format(len(target[i].HandCards_area)))
                print('目标装备区有武器牌:{},防具牌:{}, 进攻坐骑:{}, 防御坐骑:{},宝物:{}'.format(
                    target[i].equipment_area['武器'].name,
                    target[i].equipment_area['防具'].name,
                    target[i].equipment_area['进攻坐骑'].name,
                    target[i].equipment_area['防御坐骑'].name,
                    target[i].equipment_area['宝物'].name))
                print('目标判定区有:{}'.format([card.name for card in target[i].pandin_area]))
                player_input = input('请选择一张牌获得(i表示获得第i张手牌,w表示获得武器牌,a表示获得防具牌,'
                                     'j表示获得进攻坐骑,f表示获得防御坐骑,t表示获得宝物牌,l表示获得乐不思蜀,'
                                     'b表示获得兵粮寸断,s表示获得闪电):')
                try:
                    player_input = eval(player_input)
                except NameError:
                    if player_input == 'w' and target[i].equipment_area['武器'].name is not None:
                        player.HandCards_area.append(target[i].equipment_area['武器'])
                        target[i].equipment_area['武器'] = WeaponCard(None, None, None, None)
                        break
                    elif player_input == 'a' and target[i].equipment_area['防具'].name is not None:
                        player.HandCards_area.append(target[i].equipment_area['防具'])
                        target[i].equipment_area['防具'] = ArmourCard(None, None, None)
                        break
                    elif player_input == 'j' and target[i].equipment_area['进攻坐骑'].name is not None:
                        player.HandCards_area.append(target[i].equipment_area['进攻坐骑'])
                        target[i].equipment_area['进攻坐骑'] = DefenseHorseCard(None, None, None)
                        break
                    elif player_input == 'f' and target[i].equipment_area['防御坐骑'].name is not None:
                        player.HandCards_area.append(target[i].equipment_area['防御坐骑'])
                        target[i].equipment_area['防御坐骑'] = AttackHorseCard(None, None, None)
                        break
                    elif player_input == 't' and target[i].equipment_area['宝物'].name is not None:
                        player.HandCards_area.append(target[i].equipment_area['宝物'])
                        target[i].equipment_area['宝物'] = TreasureCard(None, None, None)
                        break
                    elif player_input == 'b' and '兵粮寸断' in [card.name for card in target[i].pandin_area]:
                        player.HandCards_area.append(
                            [card for card in target[i].pandin_area if card.name == '兵粮寸断'][0])
                        target[i].pandin_area.remove(
                            [card for card in target[i].pandin_area if card.name == '兵粮寸断'][0])
                        break
                    elif player_input == 'l' and '乐不思蜀' in [card.name for card in target[i].pandin_area]:
                        player.HandCards_area.append(
                            [card for card in target[i].pandin_area if card.name == '乐不思蜀'][0])
                        target[i].pandin_area.remove(
                            [card for card in target[i].pandin_area if card.name == '乐不思蜀'][0])
                        break
                    elif player_input == 's' and '闪电' in [card.name for card in target[i].pandin_area]:
                        player.HandCards_area.append([card for card in target[i].pandin_area if card.name == '闪电'][0])
                        target[i].pandin_area.remove([card for card in target[i].pandin_area if card.name == '闪电'][0])
                        break
                try:
                    get_card = target[i].HandCards_area[player_input - 1]
                    break
                except (IndexError, TypeError):
                    continue
            if type(player_input) == int:
                player.HandCards_area.append(get_card)
                del target[i].HandCards_area[player_input - 1]
        elif ca.name == '借刀杀人':
            k = 0
            for card_name in [c.name for c in target[i].HandCards_area]:
                k += card_name.find('杀')
            if k == -len(target[i].HandCards_area):
                player.HandCards_area.append(target[i].equipment_area['武器'])
                target[i].equipment_area['武器'] = WeaponCard(None, None, None, None)
            else:
                print('{}号位手牌为{}'.format(target[i].idx,
                                              [[card.name, card.color, card.point] for card in
                                               target[i].HandCards_area]))
                player_input = eval(input('{}号位是否打出杀, 0表示不出杀, i表示出第i张牌'.format(target[i].idx)))
                sha_idx = target[i].HandCards_area[player_input - 1]
                if '杀' in sha_idx.name:
                    sha_idx.target = b_target
                    use_card_process(sha_idx, target[i], Items)
                    continue
        elif ca.name == '决斗':
            while True:
                card = int(input(f'{target}, please play a card: '))
                if card == 0:
                    Damage_Process(player, ca, player, target[i], 1, False, Items)
                    break
                card = int(input(f'{player}, please play a card: '))
                if card == 0:
                    Damage_Process(target[i], ca, player, player, 1, False, Items)
                    break

    if len(Items.TmpCard) > 0:
        for i in Items.TmpCard:
            Items.LeftCardHeap.card_list.append(i)
            del Items.TmpCard[-1]
    wuxie_count = 0


# 伤害流程
def Damage_Process(source,
                   channel,
                   current_player,
                   hurt_player: Player,
                   damage_num: int,
                   is_shuxing: bool,
                   Items: items.Items,
                   is_lianhuan=False):
    """

    source: 来源
    channel: 渠道
    current_player: 当前回合角色
    hurt_player: 受到伤害的角色
    damage_num: 伤害值
    is_shuxing: 是否为属性伤害
    Items: 场上状态
    is_lianhuan: 是否为连环伤害

    """

    for time in damage_process:
        TriggerableSkill = check_skill(time, current_player, hurt_player, Items)
        if TriggerableSkill:
            for skill in TriggerableSkill:
                skill.trigger(Items)
        if time == 'when_hurt':
            if hurt_player.hengzhi and is_shuxing:  # 若受伤角色处于横置状态且受到的伤害为属性伤害,则其重置
                hurt_player.hengzhi = False
            Deducted_HP_Process(hurt_player, current_player, damage_num, Items)

    # 之后若受伤角色受到的是不为连环伤害的属性伤害且有其他角色处于横置状态,则触发铁索连环
    if not is_lianhuan and is_shuxing:
        player_tmp = current_player
        count = 1
        while count != len(Items.PlayerList):
            if player_tmp.hengzhi:
                Damage_Process(source, channel, current_player, player_tmp, damage_num, is_shuxing, Items,
                               is_lianhuan=True)
            player_tmp = player_tmp.next
            count += 1


# 扣减体力流程(造成伤害事件)
def Deducted_HP_Process(hurt_player: Player, current_player, damage_num, Items):
    for time in deducted_HP_process:
        # check_skill(time)
        if time == 'when_deducted_HP':
            hurt_player.current_HP -= damage_num
            hurt_player.max_HandCards = hurt_player.current_HP
        if time == 'af_deducted_HP':
            if hurt_player.current_HP <= 0:
                binsi(hurt_player, current_player, Items)


# 濒死流程
def binsi(binsi_player: Player, current_player: Player, Items):
    for time in binsi_process:
        # check_skill
        if time == 'when_binsi':
            current_player_tmp = current_player
            count = 0
            while (binsi_player.current_HP <= 0) and (count < len(Items.PlayerList)):
                print('{}濒死,需要{}颗桃'.format(binsi_player.commander.name, 1 - binsi_player.current_HP))
                if current_player_tmp == binsi_player:
                    print('{}号位手牌为:{}'.format(binsi_player.idx, [[card.name, card.color, card.point] for card in
                                                                      binsi_player.HandCards_area]))
                    print('{}号位是否使用桃或酒(0表示不使用,i表示使用第i张牌):'.format(binsi_player.idx))
                else:
                    print('{}号位手牌为:{}'.format(current_player_tmp.idx,
                                                   [[card.name, card.color, card.point] for card in
                                                    current_player_tmp.HandCards_area]))
                    print('{}号位是否使用桃(0表示不使用,i表示使用第i张牌):'.format(current_player_tmp.idx))
                tao = eval(input())
                if tao > 0:
                    use_card_process(current_player_tmp.HandCards_area[tao - 1], current_player_tmp, Items)
                count += 1
                current_player_tmp = current_player_tmp.next
            if binsi_player.current_HP <= 0:
                death_process(binsi_player, Items)


# 死亡流程
def death_process(dead_player: Player, Items):
    # 确认身份前
    print('死亡角色身份为:{}'.format(dead_player.identity))
    Items.PlayerList_tmp = Items.PlayerList
    dead_player.pre.next = dead_player.next
    dead_player.next.pre = dead_player.pre
    del Items.PlayerList[Items.PlayerList.index(dead_player)]
    vic = check_vic(dead_player, Items.PlayerList)
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
    Items.PlayerList = Items.PlayerList_tmp
