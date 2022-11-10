# 流程

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
                    'when_clear_start',  # 使用结算开始时
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

#

# 询问无懈可击
# def ask_wuxiekeji():
# for i in range(len(players))
