import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
import torchvision.utils
import numpy as np
import random
import os
import gym
from agents import TD3
from baseline import SMBO
from hyperopt import fmin,tpe,hp,partial,Trials
from Index.PGM import Parameter_change
from baseline.random_search import random_search
from baseline.grid_search import grid_search
from tqdm import tqdm
from utils import utils
from agents import TD3
from envs.env import LinearFitting, PGMIndex,ALEXIndex,CARMIIndex
from envs.linear_fitting import Linear_model
from agents import DDPG
from agents import dqn


def eval_policy(policy, data, query_type, search_time=150):
    if args.Index == "ALEX":
        eval_env = ALEXIndex(data,query_type=query_type)
    if args.Index == "PGM":
        eval_env = PGMIndex(data)
    if args.Index == "CARMI":
        eval_env = CARMIIndex(data,query_type=query_type)


    eval_env.reset()
    state = []
    best_state = [] 
    best_runtime= np.inf
    runtime_list = []
    avg_reward = 0.
    start_time = time.time()
    ep = 0
    # search_time = 150 #seconds
    while (time.time() - start_time) < search_time:
        state, done = eval_env.reset(), False
        eval_env.seed(100+random.randint(1,10))
        ep += 1
        for _ in range(16):
            if done:
                break
            raw_action = (policy.select_action(np.array(state)))
            action = eval_env.action_converter(raw_action)
            runtime, next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state = next_state
            runtime_list.append(runtime)
            if runtime <= best_runtime:
                best_runtime = runtime
                best_action = action
            
    epsodic_reward = avg_reward / ep
    print("---------------------------------------")
    print(f"Evaluation: parameter: {best_action} best runtime:{best_runtime:.5f}")
    print("---------------------------------------")
    return [epsodic_reward, best_action,best_runtime,runtime_list]
