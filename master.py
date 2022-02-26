import uuid
import time
from excpetion import RPSException
from game import Game


class Master(object):
    def __init__(self):
        self.id = uuid.uuid1()
        self.game = None

    def create_game(self):
        self.game = Game()
        self.game.music_player.play()
        self.game.add_player(is_bot=True)
        self.game.add_player(True)

    def add_player(self, is_bot=False):
        self.game.add_player(is_bot)

    def destroy_game(self):
        self.game = None


def master_game():
    game_master = Master()
    game_master.create_game()
    counter = 0
    while(True):
        time.sleep(3)
        game_master.game.play()
        print('round: ' + str(counter))
        print('p1: ' + str(game_master.game.player_one.score))
        print('p2: ' + str(game_master.game.player_two.score))
        counter += 1


master_game()
