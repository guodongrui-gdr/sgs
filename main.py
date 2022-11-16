from Process import *

if __name__ == '__main__':
    round = 1  # 轮数
    while 1:
        for player in player_list:
            current_player = player
            print('当前回合角色为:{}号位 {}'.format(current_player.idx, current_player.commander.name))
            Game_Process(player)

        round += 1
