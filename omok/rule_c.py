import ctypes
import numpy as np

board_size = 15
empty = 0
black_stone = 1
white_stone = -1
tie = 100

lib = ctypes.CDLL('omok/rule.so', winmode=0)

lib.CRule_new.argtypes = []
lib.CRule_new.restype = ctypes.c_void_p

lib.CRule_SetBoard.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_short)]
lib.CRule_SetBoard.restype = None

lib.CRule_ForbiddenPoint.argtypes = [ctypes.c_void_p, ctypes.c_short, ctypes.c_short, ctypes.c_short]
lib.CRule_ForbiddenPoint.restype = ctypes.c_bool

lib.CRule_isGameOver.argtypes = [ctypes.c_void_p, ctypes.c_short, ctypes.c_short, ctypes.c_short]
lib.CRule_isGameOver.restype = ctypes.c_bool

lib.CRule_searchGameover.argtypes = [ctypes.c_void_p, ctypes.c_short]
lib.CRule_searchGameover.restype = ctypes.c_bool


class Rule(object):
    def __init__(self, board):
        self.c_rule = lib.CRule_new()
        self.board = board
        self.set_c_board()

    def set_c_board(self):
        c_board = self.board.astype(np.short)
        c_board = c_board.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        lib.CRule_SetBoard(self.c_rule, c_board)

    def is_gameover(self, x, y, stone):
        self.set_c_board()
        return lib.CRule_isGameOver(self.c_rule, x, y, stone)

    def search_gameover(self, stone):
        self.set_c_board()
        return lib.CRule_searchGameover(self.c_rule, stone)

    def forbidden_point(self, x, y, stone):
        self.set_c_board()
        print(lib.CRule_ForbiddenPoint(self.c_rule, x, y, stone))
        return lib.CRule_ForbiddenPoint(self.c_rule, x, y, stone)

    def get_forbidden_points(self, stone):
        self.set_c_board()
        coords = []
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x]:
                    continue
                if lib.CRule_ForbiddenPoint(self.c_rule, x, y, stone):
                    coords.append((x, y))
        return coords

