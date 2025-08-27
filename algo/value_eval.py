import numpy as np
import torch

def mc_eval(policy, env, iterations, discount=0.9, alpha=0.01, theta=0.00001):
    """
    Evaluate a policy using TD(0) policy evaluation.

    policy_eval:
        Args:
            policy (numpy.ndarray): The policy to evaluate.
            env (gym.Env): The environment to evaluate the policy on.
            discount (float): The discount factor for future rewards.
            alpha (float): The learning rate for updating the value function.
            theta (float): The threshold for stopping the evaluation.

        Returns:
            tuple: A tuple containing:
                - v_table (numpy.ndarray): The value function for the given policy.
                - delta_list (list): A list of the TD errors for each iteration of the evaluation.
    """
    initial_state, _ = env.reset()
    v_table = np.zeros(env.observation_space.n)
    cnt = np.zeros(env.observation_space.n)
    delta_list = []

    # debug_cnt = 0
    # TD update
    # for _ in range(iterations):
    #     obs, _ = env.reset()
    #     done = False
    #     delta = 0
    #     while not done:
    #         action = policy[obs]
    #         obs_, reward, done, _, info = env.step(action)
    #         TD_error = reward + discount * v_table[obs_] - v_table[obs]
    #         delta = max(delta, np.abs(TD_error*alpha))
    #         v_table[obs] += TD_error * alpha
    #         obs = obs_
    #         # # on goal
    #         # x, y = env.to_xy(obs)
    #         # env.agent_xy = (x, y)
    #         # if env.on_goal():
    #         #     v_table[obs] = env.get_reward(x, y)
    #         #     break
    #     delta_list.append(delta)
    #     # print(delta)
    #     if delta < theta:
    #         break
    # env.close()

    # return v_table, delta_list

    # MC update
    for _ in range(iterations):
        obs, _ = env.reset()
        done = False
        states_visited = []
        returns = []
        delta = 0
        while not done:
            action = policy[obs]
            obs_, reward, done, _, info = env.step(action)
            states_visited.append(obs)
            returns.append(reward)
            obs = obs_

        for i, state in enumerate(states_visited):
            cnt[state] += 1 
            G = sum(returns[i:]*discount**np.arange(len(returns)-i))
            v_table[state] += (G - v_table[state]) / cnt[state]
            delta = max(delta, np.abs(G - v_table[state])/cnt[state])

        if delta < theta:
            break
        delta_list.append(delta)
    return v_table[initial_state], v_table, delta_list





