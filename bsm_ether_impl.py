
# The implementation of BSM-Ether: Bribery Selfish Mining in Blockchain-based Healthcare Systems


import mdptoolbox
import random
import numpy as np
import networkx as nx
import matplotlib

matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import queue
from matplotlib.colors import LogNorm
import seaborn

seaborn.set(font_scale=2.3)
seaborn.set_style("whitegrid")
import sys


class State:

    def __init__(self, l_a, l_h, fork="no_fork"):

        self.length_a = l_a
        self.length_h = l_h
        self.fork_able = fork


    def __hash__(self):
        return hash((self.length_a, self.length_h,  self.fork_able))

    def __eq__(self, other):
        try:
            return (self.length_a, self.length_h,  self.fork_able) == (
            other.length_a, other.length_h, other.fork_able)
        except:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "(%d, %d, %s)" % (self.length_a, self.length_h, self.fork_able)



def reward(a, h, rho):
    return (1 - rho) * a - rho * h

def uncle_reward(l):
    if l>=1 and l<=6:
        return (8-l)/8
    else:
        return 0

def nephew_reward(l):
    if l>=1 and l<=6:
        return 1/32
    else:
        return 0



uncle_rewards_of_attacker = 0
nephew_rewards_of_attacker = 0

uncle_rewards_of_rational = 0
nephew_rewards_of_rational = 0



def optimal_strategy_mdp(alpha, beta, gamma, bribe, cutoff, rho):
    '''
    :param alpha: attacker mining power as fraction of total mining power
    :param beta:  attacker puts the bribes on the his private chain with probability beta
    :param gamma: rational miners choose to append the new block to attacker's branch with probability gamma,
                    when attacker does not bribe them with probability 1−beta.
    :param bribe: the value b to perform bribery attack
    :param cutoff: the maximum number of blocks in private chain
    :param rho:
    :return:
    '''


    global uncle_rewards_of_attacker, nephew_rewards_of_rational, uncle_rewards_of_rational
    states = {}
    states_inverted = {}
    fork_ables = ["no_fork", "fork", "fork_b"]


    states_counter = 0

    for l_a in range(cutoff + 1):
        for l_h in range(cutoff + 1):
            for fork_able in fork_ables:

                state = State(l_a, l_h, fork_able)
                states[states_counter] = state
                states_inverted[state] = states_counter
                states_counter += 1



    P_ADOPT = np.zeros(shape=(states_counter, states_counter))
    P_WAIT = np.zeros(shape=(states_counter, states_counter))
    P_MATCH = np.zeros(shape=(states_counter, states_counter))
    P_OVERRIDE = np.zeros(shape=(states_counter, states_counter))


    REV_ADOPT = np.zeros(shape=(states_counter, states_counter))
    REV_WAIT = np.zeros(shape=(states_counter, states_counter))
    REV_MATCH = np.zeros(shape=(states_counter, states_counter))
    REV_OVERRIDE = np.zeros(shape=(states_counter, states_counter))


    for state_idx, state in states.items():
        l_a = state.length_a
        l_h = state.length_h
        fork_able = state.fork_able  # no_fork，fork，fork_b


        # action : adopt
        # Note that the adopt action is necessary.
        # attacker mines next block
        P_ADOPT[state_idx, states_inverted[State(1, 0, "no_fork")]] = alpha
        REV_ADOPT[state_idx, states_inverted[State(1, 0, "no_fork")]] = reward(0 , l_h, rho)
        # network mines next block
        P_ADOPT[state_idx, states_inverted[State(0, 1, "no_fork")]] = 1-alpha
        REV_ADOPT[state_idx, states_inverted[State(0, 1, "no_fork")]] = reward(0 , l_h, rho)

        # action : Override
        if l_a > l_h  and l_h > 0:

            # attacker mines next block
            P_OVERRIDE[state_idx, states_inverted[State(1, 0, "no_fork")]] = alpha
            REV_OVERRIDE[state_idx, states_inverted[State(1, 0, "no_fork")]] = reward(l_a, 0, rho)

            # network mines next block
            P_OVERRIDE[state_idx, states_inverted[State(0, 1, "no_fork")]] = 1-alpha
            REV_OVERRIDE[state_idx, states_inverted[State(0, 1, "no_fork")]] = reward(l_a, 0, rho)
        else:
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_OVERRIDE[state_idx, state_idx] = 1
            REV_OVERRIDE[state_idx, state_idx] = -100


        if l_a == cutoff or l_h == cutoff:
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_MATCH[state_idx, state_idx] = 1
            REV_MATCH[state_idx, state_idx] = -100

            P_WAIT[state_idx, state_idx] = 1
            REV_WAIT[state_idx, state_idx] = -100
            continue


        # action : match
        if l_a >= l_h and l_h > 0:

            rand_num = random.uniform(0,1)
            if rand_num <= beta:
                # attacker mines next block
                P_MATCH[state_idx, states_inverted[State(l_a + 1, l_h, 'fork_b')]] = alpha
                REV_MATCH[state_idx, states_inverted[State(l_a + 1, l_h, 'fork_b')]] = reward(0, 0, rho)

                # network mines next block on chain released by attacker
                P_MATCH[state_idx, states_inverted[State(l_a - l_h, 1, "fork_b")]] = gamma * (1-alpha)
                REV_MATCH[state_idx, states_inverted[State(l_a - l_h, 1, "fork_b")]] = reward(l_h , 0, rho)

                # network mines next block on honest chain
                P_MATCH[state_idx, states_inverted[State(l_a, l_h + 1, "fork_b")]] = (1 - gamma) * (1-alpha)
                REV_MATCH[state_idx, states_inverted[State(l_a, l_h + 1, "fork_b")]] = reward(0, 0 ,rho)

            else:
                # attacker mines next block
                P_MATCH[state_idx, states_inverted[State(l_a + 1, l_h, 'fork')]] = alpha
                REV_MATCH[state_idx, states_inverted[State(l_a + 1, l_h, 'fork')]] = reward(0, 0, rho)

                # network mines next block on chain released by attacker
                P_MATCH[state_idx, states_inverted[State(l_a - l_h, 1, "fork")]] = gamma * (1 - alpha)
                REV_MATCH[state_idx, states_inverted[State(l_a - l_h, 1, "fork")]] = reward(l_h, 0, rho)

                # network mines next block on honest chain
                P_MATCH[state_idx, states_inverted[State(l_a, l_h + 1, "fork")]] = (1 - gamma) * (1 - alpha)
                REV_MATCH[state_idx, states_inverted[State(l_a, l_h + 1, "fork")]] = reward(0, 0, rho)

        else:
            P_MATCH[state_idx, state_idx] = 1
            REV_MATCH[state_idx, state_idx] = -100



        ## action : Wait
        if l_a-l_h==0:
            # attacker mines next block
            P_WAIT[state_idx, states_inverted[State(l_a+1, l_h, fork_able)]] = alpha
            REV_WAIT[state_idx, states_inverted[State(l_a+1, l_h, fork_able)]] = reward(uncle_reward(l_a), nephew_reward(l_a), rho)
            uncle_rewards_of_attacker += uncle_reward(l_a)
            nephew_rewards_of_rational += nephew_reward(l_a)


            # network mines next block
            if fork_able == 'fork':
                P_WAIT[state_idx, states_inverted[State(0, 1, 'no_fork')]] = (1 - alpha) * gamma
                REV_WAIT[state_idx, states_inverted[State(0, 1, 'no_fork')]] = reward(l_a, uncle_reward(l_a)+nephew_reward(l_a) , rho)
                uncle_rewards_of_rational += uncle_reward(l_a)
                nephew_rewards_of_rational += nephew_reward(l_a)

                P_WAIT[state_idx, states_inverted[State(l_a, l_h+1, 'fork')]] = (1 - alpha) * (1-gamma)
                REV_WAIT[state_idx, states_inverted[State(l_a, l_h+1, 'fork')]] = reward(uncle_reward(l_a), nephew_reward(l_a), rho)
                uncle_rewards_of_attacker += uncle_reward(l_a)
                nephew_rewards_of_rational += nephew_reward(l_a)

            # network mines next block
            elif fork_able == 'fork_b':
                P_WAIT[state_idx, states_inverted[State(0, 1, 'no_fork')]] = 1 - alpha
                REV_WAIT[state_idx, states_inverted[State(0, 1, 'no_fork')]] = reward(l_a - bribe, uncle_reward(l_a) + nephew_reward(l_a) + bribe, rho)
                uncle_rewards_of_rational += uncle_reward(l_a)
                nephew_rewards_of_rational += nephew_reward(l_a)

            else:
                P_WAIT[state_idx, states_inverted[State(l_a, l_h+1, 'no_fork')]] = 1 - alpha
                REV_WAIT[state_idx, states_inverted[State(l_a, l_h+1, 'no_fork')]] = reward(0, 0, rho)


        elif l_a > l_h :
            # attacker mines next block
            P_WAIT[state_idx, states_inverted[State(l_a + 1, l_h, fork_able)]] = alpha
            REV_WAIT[state_idx, states_inverted[State(l_a + 1, l_h, fork_able)]] = reward(0, 0, rho)

            if l_h == 0:
                P_WAIT[state_idx, states_inverted[State(l_a, 1, 'no_fork')]] = (1 - alpha)
                REV_WAIT[state_idx, states_inverted[State(l_a, 1, 'no_fork')]] = reward(0, 0, rho)

            else:
                if fork_able == 'fork':
                    # network mines next block
                    P_WAIT[state_idx, states_inverted[State(l_a - l_h, 1, 'fork')]] = (1 - alpha) * gamma
                    REV_WAIT[state_idx, states_inverted[State(l_a - l_h, 1, 'fork')]] = reward(l_h, 0, rho)

                    P_WAIT[state_idx, states_inverted[State(l_a, l_h + 1, 'fork')]] = (1 - alpha) * (1 - gamma)
                    REV_WAIT[state_idx, states_inverted[State(l_a, l_h + 1, 'fork')]] = reward(0, 0, rho)

                elif fork_able == 'fork_b':
                    P_WAIT[state_idx, states_inverted[State(l_a - l_h, 1, 'fork_b')]] = (1 - alpha) * beta
                    REV_WAIT[state_idx, states_inverted[State(l_a - l_h, 1, 'fork_b')]] = reward(l_h - bribe, bribe, rho)

                    P_WAIT[state_idx, states_inverted[State(l_a - l_h, 1, 'fork')]] = (1 - alpha) * (1 - beta)
                    REV_WAIT[state_idx, states_inverted[State(l_a - l_h, 1, 'fork')]] = reward(l_h - bribe, bribe, rho)

                else:
                    P_WAIT[state_idx, states_inverted[State(l_a, l_h + 1, 'no_fork')]] = 1 - alpha
                    REV_WAIT[state_idx, states_inverted[State(l_a, l_h + 1, 'no_fork')]] = reward(0, 0, rho)


        else:
            P_WAIT[state_idx, state_idx] = 1
            REV_WAIT[state_idx, state_idx] = -100



    P = [P_ADOPT, P_OVERRIDE, P_WAIT, P_MATCH]

    # print(" check matrix is stochastic or not : ")
    for i, p in enumerate(P):
        try:
            mdptoolbox.util.checkSquareStochastic(p)
        except:
            print("not stochastic:", i)
            print(p)


    R = [REV_ADOPT, REV_OVERRIDE, REV_WAIT, REV_MATCH]

    mdp = mdptoolbox.mdp.PolicyIteration(P, R, 0.999)

    mdp.run()
    return mdp, states



def optimal_strategy(alpha, beta, gamma, bribe, cutoff, eps=0.00001):
    low = 0.0
    high = 1.0
    while high - low >= eps / 8:
        rho = (low + high) / 2
        mdp, states = optimal_strategy_mdp(alpha, beta, gamma, bribe, cutoff, rho)
        if mdp.V[0] > 0:
            low = rho
        else:
            high = rho
    return mdp, states, rho


def relative_reward(beta, gamma, bribe, cutoff, eps):
    ps = np.arange(0.025, 0.5, 0.025) # attacker's mining power
    rev = np.zeros(ps.shape)

    for idx, alpha in enumerate(ps):
        mdp, states, rho = optimal_strategy(alpha, beta, gamma, bribe, cutoff, eps)
        rev[idx] = rho
        print(alpha,rho)

    np.save("./data/rev_beta%.2f.npy" % (beta), rev)



def main():
    beta = 1.00 # the probability of launching a bribery attack
    gamma = 0.75
    bribe = 0.01
    cutoff = 12
    eps = 0.00001
    relative_reward(beta, gamma, bribe, cutoff, eps)

if __name__ == "__main__":
    main()
