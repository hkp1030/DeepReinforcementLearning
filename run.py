import threading
import time

import numpy as np

np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload

from tensorflow.keras.utils import plot_model

from game import Game, GameState
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle


def self_play():
    global memory

    # 에이전트 생성
    best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
    iteration = 0

    # 자가경기(SELF PLAY)
    while 1:
        iteration += 1

        if iteration % 50 == 0:
            reload(lg)
            reload(config)

        lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)

        if iteration % 10 == 0:
            print('현재 자가경기 횟수 :', iteration)

        _, memory, _, _ = playMatches(best_player, best_player, 1, lg.logger_main,
                                      turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=memory)

        memory.clear_stmemory()


def retraining():
    global best_player_version

    # 에이전트 생성
    current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT,current_NN)
    best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
    iteration = 0

    while 1:
        if len(memory.ltmemory) >= config.MEMORY_SIZE:

            ######## 신경망 다시 학습하기(RETRAINING) ########
            print('')
            print('RETRAINING...')
            print('BEST PLAYER VERSION :', best_player_version)
            current_player.replay(memory.ltmemory)
            print('')

            if iteration % 2 == 0:
                pickle.dump(memory, open(run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb"))

            lg.logger_memory.info('====================')
            lg.logger_memory.info('NEW MEMORIES')
            lg.logger_memory.info('====================')

            memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))

            for s in memory_samp:
                current_value, current_probs, _ = current_player.get_preds(s['state'])
                best_value, best_probs, _ = best_player.get_preds(s['state'])

                lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
                lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
                lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
                lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']])
                lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in current_probs])
                lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in best_probs])
                lg.logger_memory.info('ID: %s', s['state'].id)
                lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))

                s['state'].render(lg.logger_memory)

            ######## 신경망 평가하기(TOURNAMENT) ########
            print('TOURNAMENT...')
            scores, _, points, sp_scores = playMatches(best_player, current_player, config.EVAL_EPISODES,
                                                       lg.logger_tourney, turns_until_tau0=0, memory=None)
            print('\nSCORES')
            print(scores)
            print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
            print(sp_scores)
            # print(points)

            print('\n\n')

            if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
                best_player_version = best_player_version + 1
                best_NN.model.set_weights(current_NN.model.get_weights())
                best_NN.write(env.name, best_player_version)

        else:
            print('MEMORY SIZE: ' + str(len(memory.ltmemory)))
            time.sleep(180)


lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

# 기존 신경망을 로드하는 경우 구성 파일을 루트에 복사
if initialise.INITIAL_RUN_NUMBER is not None:
    copyfile(run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
             './config.py')

import config

######## 메모리 로드 ########

if initialise.INITIAL_MEMORY_VERSION is None:
    memory = Memory(config.MEMORY_SIZE)
else:
    print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) + '...')
    memory = pickle.load(open(run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) +
                              "/memory/memory" + str(initialise.INITIAL_MEMORY_VERSION).zfill(4) + ".p", "rb"))

######## 모델 로드 ########

# 훈련되지 않은 신경망 객체 생성
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                          config.HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                       config.HIDDEN_CNN_LAYERS)

# 기존 신경망을 로드하는 경우 해당 모델의 가중치 설정
if initialise.INITIAL_MODEL_VERSION is not None:
    best_player_version = initialise.INITIAL_MODEL_VERSION
    print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
    m_tmp = best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
    current_NN.model.set_weights(m_tmp.get_weights())
    best_NN.model.set_weights(m_tmp.get_weights())
# 그렇지 않으면 두 신경망의 가중치를 동일하게 설정
else:
    best_player_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())

# config.py 파일을 run 폴더에 복사
copyfile('./config.py', run_folder + 'config.py')
plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes=True)

print('\n')

# 셀프 플레이 활성화
threading.Thread(target=self_play).start()
threading.Thread(target=retraining).start()
