import numpy as np
import logging
from omok.rule import *


class Game:

	def __init__(self):
		self.currentPlayer = black_stone
		self.grid_shape = (15,15)
		self.input_shape = (3,15,15)
		self.gameState = GameState(np.zeros(self.grid_shape[0]*self.grid_shape[1], dtype=np.int), black_stone)
		self.actionSpace = np.zeros(self.grid_shape[0]*self.grid_shape[1], dtype=np.int)
		self.pieces = {'1': 'O', '0': '-', '-1': 'X'}
		self.name = 'omok'
		self.state_size = len(self.gameState.binary)
		self.action_size = len(self.actionSpace)

	def reset(self):
		self.gameState = GameState(np.zeros(self.grid_shape[0]*self.grid_shape[1], dtype=np.int), black_stone)
		self.currentPlayer = black_stone
		return self.gameState

	def step(self, action):
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		self.currentPlayer = -self.currentPlayer
		info = None
		return ((next_state, value, done, info))

	def identities(self, state, actionValues):
		identities = [(state,actionValues)]

		currentBoard = np.reshape(state.board, (self.grid_shape[0], self.grid_shape[1]))
		currentAV = np.reshape(actionValues, (self.grid_shape[0], self.grid_shape[1]))

		currentBoard = np.rot90(currentBoard)
		currentAV = np.rot90(currentAV)
		identities.append((GameState(np.reshape(currentBoard, -1), state.playerTurn), np.reshape(currentAV, -1)))

		currentBoard = np.rot90(currentBoard)
		currentAV = np.rot90(currentAV)
		identities.append((GameState(np.reshape(currentBoard, -1), state.playerTurn), np.reshape(currentAV, -1)))

		currentBoard = np.rot90(currentBoard)
		currentAV = np.rot90(currentAV)
		identities.append((GameState(np.reshape(currentBoard, -1), state.playerTurn), np.reshape(currentAV, -1)))

		return identities


class GameState():
	def __init__(self, board, playerTurn):
		self.board = board
		self.playerTurn = playerTurn
		self.rule = Rule(np.reshape(self.board, (15, 15)))
		self.pieces = {'1': 'O', '0': '-', '-1': 'X'}
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = None
		self.isEndGame = None
		self.value = None
		self.score = None

	def get_allowed_actions(self):
		if self.allowedActions is not None:
			return self.allowedActions

		if np.all(self.board == 0):
			return [112]

		allowed = []
		allowed.extend([i for i, stone in enumerate(self.board) if stone == 0])

		if self.playerTurn == black_stone:
			forbidden_points = self.rule.get_forbidden_points(self.playerTurn)
			while forbidden_points:
				x, y = forbidden_points.pop()
				allowed.remove(y*15 + x)

		self.allowedActions = allowed

		return allowed

	def _binary(self):
		black_stone_position = np.zeros(len(self.board), dtype=np.int)
		black_stone_position[self.board == 1] = 1

		white_stone_position = np.zeros(len(self.board), dtype=np.int)
		white_stone_position[self.board == -1] = 1

		if self.playerTurn == 1:
			player_turn = np.ones(len(self.board), dtype=np.int)
		else:
			player_turn = np.zeros(len(self.board), dtype=np.int)

		position = np.append(black_stone_position, white_stone_position)
		position = np.append(position, player_turn)

		return (position)

	def _convertStateToId(self):
		player1_position = np.zeros(len(self.board), dtype=np.int)
		player1_position[self.board==1] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-1] = 1

		position = np.append(player1_position,other_position)

		id = ''.join(map(str,position))

		return id

	def is_end_game(self):
		if self.isEndGame is not None:
			return self.isEndGame

		if np.count_nonzero(self.board) == 15 * 15:
			self.isEndGame = 1
			return 1

		if np.count_nonzero(self.board) > 210 and not self.get_allowed_actions():
			self.isEndGame = 1
			return 1

		if self.rule.search_gameover(-self.playerTurn):
			self.isEndGame = 1
			return 1

		self.isEndGame = 0
		return 0

	def get_value(self):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose
		if self.value is not None:
			return self.value

		if self.is_end_game() == 1:
			self.value = (-1, -1, 1)
			return (-1, -1, 1)
		else:
			self.value = (0, 0, 0)
			return (0, 0, 0)

	def get_score(self):
		if self.score is not None:
			return self.score

		tmp = self.get_value()
		self.score = (tmp[1], tmp[2])
		return (tmp[1], tmp[2])

	def takeAction(self, action):
		newBoard = np.array(self.board)
		newBoard[action] = self.playerTurn

		newState = GameState(newBoard, -self.playerTurn)

		value = 0
		done = 0

		if newState.is_end_game():
			value = newState.get_value()[0]
			done = 1

		return (newState, value, done)

	def render(self, logger):
		convert_board = np.reshape(self.board, (15, 15))
		for stone in convert_board:
			logger.info([self.pieces[str(s)] for s in stone])
		logger.info('--------------------')


