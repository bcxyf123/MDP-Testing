import numpy as np

def dp_eval(policy, mdp, discount=0.9,theta=0.00001):
    """
    Evaluate a policy using iterative policy evaluation algorithm.

    Args:
        policy (numpy array): policy to be evaluated
        mdp (MDP object): Markov Decision Process object
        discount (float): discount factor (default: 0.9)
        theta (float): stopping threshold (default: 0.00001)

    Returns:
        v_table (numpy array): state-value function for the given policy
        delta_list (list): list of delta values for each iteration
    """
    initial_state, _ = mdp.reset()
    v_table = np.zeros(mdp.observation_space.n)
    delta_list = []
    while True:
        delta = 0
        for s in range(mdp.observation_space.n):
            # in wall -- v=0
            x, y = mdp.to_xy(s)
            # if not mdp.is_free(x, y):
            #     continue
            mdp.agent_xy = (x, y)
            # # on goal
            # if mdp.on_goal():
            #     v_table[s] = 0
            #     # v_table[s] = mdp.get_reward(x, y)
            #     continue
            v = 0
            a = policy[s]
            next_states, probs, rewards = mdp.get_transitions(s, a)
            for s_, prob, r in zip(next_states, probs, rewards):
                v += prob*(r + discount* v_table[s_])
            delta=max(delta, np.abs(v-v_table[s]))
            v_table[s]=v
        delta_list.append(delta)
        if delta < theta:
            break
    mdp.close()

    return v_table[initial_state], v_table, delta_list