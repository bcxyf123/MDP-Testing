import sys
sys.path.append('..')
import numpy as np
from envs.gym_simplegrid.generator import *
from utils.utils import *
from algo.env_model import baseNet

def iterate_mdp(mdp):
    state_space = mdp.observation_space.n
    action_space = mdp.action_space.n
    transition_map = []
    for state in range(state_space):
        next_state_list = [state-mdp.ncol, state+mdp.ncol, state-1, state+1, state]
        for action in range(action_space):
            next_states, probs, rewards = mdp.get_transitions(state, action)
            trans_probs = [0 for _ in range(len(next_state_list))]
            for idx, s_ in enumerate(next_state_list):
                if s_ in next_states:
                    indices = [i for i, x in enumerate(next_states) if x == s_]
                    trans_probs[idx] = sum([probs[i] for i in indices])
            transition_map.append(trans_probs)
    transition_map = np.array(transition_map).reshape(state_space, -1)
    save_csv(transition_map, 'save/env/transition_map/mdp_map.csv')


def iterate_model(model, env):
    state_space = env.observation_space.n
    action_space = env.action_space.n
    transition_map = []
    for state in range(state_space):
        next_state_list = [state-env.ncol, state+env.ncol, state-1, state+1, state]
        for action in range(action_space):
            # trans_probs = [0 for _ in range(len(next_state_list))]
            with torch.no_grad():
                x = torch.tensor(np.array([state, action]), dtype=torch.float32).unsqueeze(0)
                output = model(x)
            output = output.detach().numpy()
            trans_probs = output
            transition_map.append(trans_probs)
    transition_map = np.array(transition_map).reshape(state_space, -1)
    save_csv(transition_map, 'save/env/transition_map/model_map.csv')

def main():
    # save mdp transition
    mdp = genMDP(map_size=(4,8), map_id=1, start_loc=(3,0), goal_loc=(0,7), deterministic=False, seed=222333)
    model_path = 'save/model/forward_model/map_1_forward_model.pt'
    model = baseNet(2, 5)
    model.load_state_dict(torch.load(model_path))
    iterate_mdp(mdp)
    iterate_model(model, mdp)

if __name__ == "__main__":
    main()