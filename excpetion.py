class RPSException(Exception):
    def __init__(self, code=999, message='Undefined Error'):
        self.code = code
        self.message = message
