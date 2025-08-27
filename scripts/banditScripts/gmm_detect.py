import sys
sys.path.append('../..')
import numpy as np
from algo.pedm.ood_baselines.classical.gmm_detector import GMM_Detector

from pedm_detect import *
from utils.utils import t_paired_test

def gmm_predict(default_detector, test_detector, solver, num_trajs, device="cpu"):
    
    traj_states, traj_actions, traj_delta_states = collect_trajectories(solver, num_trajs)
    powers = []
    for states, actions, delta_states in zip(traj_states, traj_actions, traj_delta_states):
        states = np.array(states)
        actions = np.array(actions)
        delta_states = np.array(delta_states)
        obs = np.concatenate((states, delta_states), axis=1)

        default_outputs = default_detector._predict_scores(obs)
        test_outputs = test_detector._predict_scores(obs)

        power = t_paired_test(default_outputs, test_outputs)
        powers.append(power)
        
    print(f'power: {np.mean(powers):.3f} +- {np.std(powers):.3f}')
    
    return np.mean(powers)


def train_gmm_detector(solver, detector):
    obs, actions, delta_obs = collect_trajectories(solver, episodes=20)
    obs = np.array(obs).reshape(-1, 1)
    delta_obs = np.array(delta_obs).reshape(-1, 1)
    train_data = np.concatenate((obs, delta_obs), axis=1)
    detector.fit(train_data, delta_obs)

@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    default_bandit = BernoulliBandit(p=0.1)
    default_solver = ThompsonSampling(default_bandit)
    default_solver.policy_fixed = True
    default_detector = GMM_Detector(default_solver, n_components=2, n_init=5, normalize_data=False)
    train_gmm_detector(default_solver, default_detector,)

    a = deepcopy(default_solver._a)
    b = deepcopy(default_solver._b)

    p_list = np.arange(0.1, 1.0, 0.005)
    powers = []
    for p in p_list:
        print(f"arm probability: ", p)
        test_bandit = BernoulliBandit(p=p)
        test_solver = ThompsonSampling(test_bandit)
        test_solver._a = a
        test_solver._b = b
        test_solver.policy_fixed = True
        test_detector = GMM_Detector(test_solver, n_components=2, n_init=5, normalize_data=False)
        train_gmm_detector(test_solver, test_detector,)
        power = gmm_predict(default_detector, test_detector, test_solver, num_trajs=cfg.detect.collect_trajs, device='cpu')
        powers.append(power)
        print()
    
    data = {
        'powers': powers
    }
    df = pd.DataFrame(data)
    df.to_csv(f'tables/gmm.csv', index=False)

    # plot
    df = pd.read_csv(f'tables/gmm.csv')
    powers = df['powers']

    plt.figure()
    plt.plot(p_list, powers, label='Power')
    plt.scatter(p_list[0], powers[0], color='red', label='default power')
    plt.xlabel(f'{cfg.detect.param_name}')
    plt.ylabel('Power')
    plt.legend()
    plt.savefig(f'figs/pdf/gmm_power.pdf', dpi=300)
    plt.savefig(f'figs/png/gmm_power.png', dpi=300)


if __name__ == '__main__':
    main()
