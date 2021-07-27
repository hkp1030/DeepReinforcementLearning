# 자가 경기
EPISODES = 30
MCTS_SIMS = 100
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10  # 결정론적으로(deterministically) 게임하기 시작하는 때
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

# 재학습
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
]

# 평가
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3
