import os

L = 1.0
H = 1.0
E = 1.0
MU = 0.3
TRACTION_Y = 0.01

N_INNER = 2500
N_EDGE = 250

ADAM_LR = 1e-2
ADAM_EPOCHS = 5000

LBFGS_MAX_ITERS = 20000

RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)