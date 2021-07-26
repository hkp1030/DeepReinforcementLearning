import numpy as np
import logging
from omok.rule import *


class Game:

	def __init__(self):
		self.currentPlayer = black_stone
		self.grid_shape = (15,15)
		self.input_shape = (2,15,15)
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
		self.pieces = {'1': 'O', '0': '-', '-1': 'X'}
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = self._allowedActions()
		self.isEndGame = self._checkForEndGame()
		self.value = self._getValue()
		self.score = self._getScore()

	def _allowedActions(self):
		if np.all(self.board == 0):
			return [63]

		allowed = []
		allowed.extend([i for i, stone in enumerate(self.board) if stone == 0])

		if self.playerTurn == black_stone:
			convert_board = np.reshape(self.board, (15, 15))
			forbidden_points = Rule(convert_board).get_forbidden_points(self.playerTurn)
			while forbidden_points:
				x, y = forbidden_points.pop()
				allowed.remove(y*15 + x)

		return allowed

	def _binary(self):
		currentplayer_position = np.zeros(len(self.board), dtype=np.int)
		currentplayer_position[self.board==self.playerTurn] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-self.playerTurn] = 1

		position = np.append(currentplayer_position,other_position)

		return (position)

	def _convertStateToId(self):
		player1_position = np.zeros(len(self.board), dtype=np.int)
		player1_position[self.board==1] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-1] = 1

		position = np.append(player1_position,other_position)

		id = ''.join(map(str,position))

		return id

	def _checkForEndGame(self):
		if np.count_nonzero(self.board) == 15 * 15:
			return 1

		if not self.allowedActions:
			return 1

		convert_board = np.reshape(self.board, (15, 15))
		if Rule(convert_board).search_gameover(-self.playerTurn):
			return 1

		return 0

	def _getValue(self):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose
		if self.isEndGame == 1:
			return (-1, -1, 1)
		else:
			return (0, 0, 0)

	def _getScore(self):
		tmp = self.value
		return (tmp[1], tmp[2])

	def takeAction(self, action):
		newBoard = np.array(self.board)
		newBoard[action] = self.playerTurn

		newState = GameState(newBoard, -self.playerTurn)

		value = 0
		done = 0

		if newState.isEndGame:
			value = newState.value[0]
			done = 1

		return (newState, value, done)

	def render(self, logger):
		for stone in self.board:
			logger.info([self.pieces[str(s)] for s in stone])
		logger.info('--------------------')

