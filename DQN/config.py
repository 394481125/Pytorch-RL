import torch

GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

ENV_NAME='Carnival-v0'

BATCH_SIZE = 256
LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99

NUM_EPISODE = 3000
EPS_INIT = 1
EPS_DENCY = 0.995
EPS_MIN = 0.05
MAX_T = 1500
CONSTANT = 0.99
NUM_FRAME = 2