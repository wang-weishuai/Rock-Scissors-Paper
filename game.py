import uuid
from enum import Enum
from excpetion import *
from music import music
import hand_classifier
import random


class PlayerType(Enum):
    human = 1
    bot = 0


class RPScy(Enum):
    NotSet = 0
    Rock = 1
    Paper = 2
    Scissor = 3


class Controller(object):
    def __init__(self):
        pass

    @staticmethod
    def action():
        pass


class BotController(Controller):
    @staticmethod
    def action():
        return RPScy(random.randint(1, 3))


class HumanController(Controller):
    def __init__(self):
        super().__init__()
        self.controller = hand_classifier.model()

    def action(self):
        return RPScy(random.randint(1, 3))


class Player(object):
    def __init__(self, game, is_bot=False):
        self.id = uuid.uuid1()
        self.score = 0
        self.action = RPScy.NotSet
        self.game = game
        if is_bot:
            self.type = PlayerType.bot
            self.controller = BotController()
        else:
            self.type = PlayerType.human
            self.controller = HumanController()

    def get_action(self):
        self.action = self.controller.action()

    def reset_action(self):
        self.action = RPScy.NotSet


class Game(object):
    def __init__(self):
        self.id = uuid.uuid1()
        self.player_one = None
        self.player_two = None
        self.music_player = music.Music()

    def add_player(self, is_bot=False):
        if self.player_one is None:
            self.player_one = Player(self, is_bot)
        elif self.player_two is None:
            self.player_two = Player(self, is_bot)
        else:
            raise RPSException(300, 'Unable to create new player')

    def play(self):
        winner = self.judge()
        if winner == 0:
            pass
        elif winner == 1:
            self.player_one.score += 1
        elif winner == 2:
            self.player_two.score += 1
        else:
            raise RPSException(400, 'Player wining is illegal')
        self.reset_action()

    def judge(self):
        self.player_one.get_action()
        self.player_two.get_action()
        # The range of possible action value shall be range(1,4)
        # As range(1,3) only contains (1,2).
        if self.player_one.action.value not in range(1, 4) or self.player_two.action.value not in range(1, 4):
            raise RPSException(500, 'Illegal action status')
        if self.player_one.action == self.player_two.action:
            return 0
        else:
            if self.player_one.action == RPScy.Rock:
                if self.player_two.action == RPScy.Paper:
                    return 2
                elif self.player_two.action == RPScy.Scissor:
                    return 1
            elif self.player_one.action == RPScy.Paper:
                if self.player_two.action == RPScy.Rock:
                    return 1
                elif self.player_two.action == RPScy.Scissor:
                    return 2
            elif self.player_one.action == RPScy.Scissor:
                if self.player_two.action == RPScy.Rock:
                    return 2
                elif self.player_two.action == RPScy.Paper:
                    return 1

    def reset_action(self):
        self.player_one.reset_action()
        self.player_two.reset_action()
