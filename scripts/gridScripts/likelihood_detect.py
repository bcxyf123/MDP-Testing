import sys
sys.path.append("../..")
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.utils import *
from envs.gym_simplegrid.generator import *
from train_forward_model import collect_trajectories, TransitionModel

def map_test(map_id=0, n_eps=10):
    
    policy_dir = 'save/model/policy/map_0_policy_2000.csv'
    default_model_dir = 'save/model/forward_model/forward_model_default.pth'
    test_model_dir = f'save/model/forward_model/forward_model_{map_id}.pth'

    policy = read_csv(policy_dir)[:, -1]
    default_model = TransitionModel(obs_dim=1, action_dim=1)
    default_model.load_state_dict(torch.load(default_model_dir))
    test_model = TransitionModel(obs_dim=1, action_dim=1)
    test_model.load_state_dict(torch.load(test_model_dir))
    
    env = genEnv(map_size=(4,8), map_id=map_id, max_prob=0.9, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    trajectories = collect_trajectories(env, policy, episodes=n_eps)
    model0 = default_model
    model1 = test_model

    likelihood0_list = []
    likelihood1_list = []
    ratios = []

    for trj in trajectories:
        likelihood0, likelihood1, ratio = likelihoodRatio(trj, model0, model1)
        likelihood0_list.append(likelihood0)
        likelihood1_list.append(likelihood1)
        ratios.append(ratio)
    
    # power = chi_test(ratios, df=2)

    # print(f'likelihood0: {np.mean(likelihood0_list):.3f} +- {np.std(likelihood0_list):.3f}')
    # print(f'likelihood1: {np.mean(likelihood1_list):.3f} +- {np.std(likelihood1_list):.3f}')
    # print(f'ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}')
    # print(f'power: {power:.2f}')

    return ratios

def collect_transitions(policy, env, episodes):
    transitions = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done, _, info = env.step(action)
            transitions.append((state, action, next_state))
            state = next_state
    return transitions

def action_transform_batch(actions):
    action_map = np.array([-8, 8, -1, 1])
    # actions = np.array(actions)
    return action_map[actions]

def estimate_prob(policy, env, num_epsisodes=50):

    transitions = collect_transitions(policy, env, episodes=num_epsisodes)
    states, actions, next_states = zip(*transitions)
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    delta_states = next_states - states
    
    expected_delta_states = action_transform_batch(actions)
    probs = np.zeros(5)

    # Compare expected and actual delta states
    equal_states = expected_delta_states == delta_states
    opposite_states = expected_delta_states == -delta_states
    zero_states = delta_states == 0

    # Update probabilities based on comparisons
    probs[0] += np.sum(equal_states)
    probs[1] += np.sum(opposite_states)
    probs[-1] += np.sum(zero_states)
    
    rest_prob = len(states) - np.sum(probs)
    probs[2] += rest_prob / 2
    probs[3] += rest_prob / 2
    
    probs = probs / len(states)
    # print(sum(probs))

    return probs

def lrt(trajs, prob_0, prob_1):

    def compute_likelihood(traj, probs):
        states, actions, _, delta_states = zip(*traj)
        states = np.array(states)
        actions = np.array(actions)
        delta_states = np.array(delta_states)
        expected_delta_states = action_transform_batch(actions)
        # Compare expected and actual delta states
        equal_states = np.sum(expected_delta_states == delta_states)
        opposite_states = np.sum(expected_delta_states == -delta_states)
        zero_states = np.sum(delta_states == 0)
        other_states = len(states) - equal_states - opposite_states - zero_states
        # Update probabilities based on comparisons
        epsilon = np.finfo(float).eps
        probs = np.where(probs == 0, epsilon, probs)
        likelihood = np.sum(np.log(probs[0]) * equal_states + np.log(probs[1]) * opposite_states + np.log(probs[-1]) * zero_states + np.log(probs[2]) * other_states)
        return likelihood

    ratio_list = []
    log_likelihood_0_list = []
    log_likelihood_1_list = []
    for traj in trajs:
        log_likelihood_0 = compute_likelihood(traj, prob_0)
        log_likelihood_1 = compute_likelihood(traj, prob_1)
        ratio = -2 * (log_likelihood_0 - log_likelihood_1)
        ratio_list.append(ratio)
        log_likelihood_0_list.append(log_likelihood_0)
        log_likelihood_1_list.append(log_likelihood_1)

    # print(f'ratio: {np.mean(ratio_list):.3f} +- {np.std(ratio_list):.3f}')

    return ratio_list


def main():
    likelihood_0_means = []
    likelihood_0_stds = []
    likelihood_1_means = []
    likelihood_1_stds = []
    ratio_means = []
    ratio_stds = []
    powers = []
    for i in range(7):
        likelihood_0, liklihood_1, ratio, power = trajectory_test(map_id=i, sample_size=100)
        likelihood_0_means.append(np.mean(likelihood_0))
        likelihood_0_stds.append(np.std(likelihood_0))
        likelihood_1_means.append(np.mean(liklihood_1))
        likelihood_1_stds.append(np.std(liklihood_1))
        ratio_means.append(np.mean(ratio))
        ratio_stds.append(np.std(ratio))
        powers.append(power)

    data = {
        'likelihood_0_means': likelihood_0_means,
        'likelihood_0_stds': likelihood_0_stds,
        'likelihood_1_means': likelihood_1_means,
        'likelihood_1_stds': likelihood_1_stds,
        'ratio_means': ratio_means,
        'ratio_stds': ratio_stds,
        'powers': powers,
    }

    df = pd.DataFrame(data)
    df.to_csv(f'tables/map_lrt.csv', index=False)

    # plot
    plt.rcParams.update({'font.size': 12})

    plt.figure()
    plt.plot(np.arange(7), likelihood_0_means, label='$\ell_0$')
    plt.fill_between(np.arange(7), np.array(likelihood_0_means)-np.array(likelihood_0_stds), np.array(likelihood_0_means)+np.array(likelihood_0_stds), alpha=0.2)
    plt.plot(np.arange(7), likelihood_1_means, label='$\ell_1$')
    plt.fill_between(np.arange(7), np.array(likelihood_1_means)-np.array(likelihood_1_stds), np.array(likelihood_1_means)+np.array(likelihood_1_stds), alpha=0.2)
    plt.xlabel('Map ID')
    plt.ylabel('Likelihood')
    # plt.title('Likelihood of Different Test Environments')
    plt.legend()
    plt.savefig('figs/pdf/likelihood_map.pdf', dpi=300)

    plt.figure()
    plt.plot(np.arange(7), ratio_means, label='$\log{\Lambda}$')
    plt.fill_between(np.arange(7), np.array(ratio_means)-np.array(ratio_stds), np.array(ratio_means)+np.array(ratio_stds), alpha=0.2)
    plt.xlabel('Map ID')
    plt.ylabel('Log Likelihood Ratio')
    # plt.title('Log Likelihood Ratio of Different Test Environments')
    plt.legend()
    plt.savefig('figs/pdf/ratio_map.pdf', dpi=300)


def prob_test(test_p, n_eps=10, es_size=10):
    default_p = 0.9
     # load trained policy
    policy_path = f'save/model/policy/map_0_policy_2000.csv'
    policy = read_csv(policy_path)[:, -1]
    # default_env
    env_0 = genEnv(map_size=(4,8), map_id=0, max_prob=default_p, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    # estimate probability
    probs_0 = estimate_prob(policy, env_0, num_epsisodes=es_size)
    # test env
    env_1 = genEnv(map_size=(4,8), map_id=0, max_prob=test_p, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    # estimate probability
    probs_1 = estimate_prob(policy, env_1, num_epsisodes=es_size)
    # print(probs_1)
    trajs = collect_trajectories(env_1, policy, n_eps)
    # ratios = abs(np.array(lrt(trajs, probs_0, probs_1)))
    ratio_list = lrt(trajs, probs_0, probs_1)
    # power = chi_test(ratios, df=5)
    # powers.append(power)
    # plt.hist(ratios, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='likelihood ratio')
    # plt.show()
    # print()
    # plt.plot(p_list, powers)
    # plt.show()
    return ratio_list


class maze_lrt_tester(lrt_tester):
    def __init__(self, is_map_test, policy, n_trials=10, gamma=0.99):
        super().__init__(
            n_trials=n_trials
        )
        self.is_map_test = is_map_test
        self.policy = policy
        self.gamma = gamma
        self.default_map = 0
        self.default_prob = 0.9

    def sample(
            self, 
            is_save_metadata: bool = False,
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> np.ndarray:
        if self.is_map_test:
            ratio_list = map_test(map_id=test_map, n_eps=self.n_trials)
        else:
            ratio_list = prob_test(test_p=test_prob, n_eps=self.n_trials, es_size=50)

        return np.array([ratio_list])
    
    def dist_estimation(
            self, 
            n_samples: int, 
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> tuple[float, float]:

        chi_stats = []
        for _ in range(n_samples):
            if self.is_map_test:
                ratio_arr = self.sample(test_map=test_map,)
            else:
                ratio_arr = self.sample(test_prob=test_prob,)
            chi_stat = np.mean(ratio_arr)
            chi_stats.append(chi_stat)
        print('data collection finished!')
        # sns.histplot(t_statistics, kde=False, stat='density', alpha=0.5, edgecolor='black', linewidth=0.5,)
        # plt.show()

        # # use kernel density estimation to estimate the distribution
        kde = gaussian_kde(np.array(chi_stats))
        # kde_estimation(np.array(t_statistics))
        # pdf range
        x_grid = np.linspace(min(chi_stats) - 1, max(chi_stats) + 1, 1000)
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, is_plot=True)

        return reject_region
    
    def run_test(
            self, 
            reject_region: float, 
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> bool:

        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        if self.is_map_test:
            ratio_arr = self.sample(test_map=test_map, is_save_metadata=True,)
        else:
            ratio_arr = self.sample(test_prob=test_prob, is_save_metadata=True,)
        chi_stat = np.mean(ratio_arr)
        if chi_stat < reject_region[0] or chi_stat > reject_region[1]:
            # print(f't: {t:.4f}, reject')
            return True
        else:
            # print(f't: {t:.4f}, accept')
            return False
    
    def power_analysis(
            self, 
            reject_region: float, 
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> float:
        
        powers = []
        for _ in range(100):
            if self.is_map_test:
                powers.append(self.run_test(reject_region, test_map=test_map,))
            else:
                powers.append(self.run_test(reject_region, test_prob=test_prob,))
        power = sum(powers) / 100

        return power


if __name__=='__main__':

    is_map_test = False
    prob = np.arange(0.9, 0.1, -0.01)
    map_list = range(7)

    powers = []

    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tester = maze_lrt_tester(is_map_test=is_map_test, policy=None, n_trials=1, gamma=0.99)
    reject_region = tester.dist_estimation(n_samples=1000, )

    print('current trial number: 1, current sample number: 100')

    if is_map_test:
        for m in map_list:
            power = tester.power_analysis(reject_region, test_prob=0.9, test_map=m,)
            powers.append(power)
            print(f'map: {m}, power: {power:.3f}')
        data = pd.DataFrame({
        'map_id': map_list, 
        'power': powers,
        })
        save_path = f'{save_dir}/lrt_map_test_results_1_100.csv'
    else:
        for p in prob:
            power = tester.power_analysis(reject_region, test_map=0, test_prob=p,)
            powers.append(power)
            print(f'prob: {p}, power: {power:.3f}')
        data = pd.DataFrame({
        'prob': prob, 
        'power': powers,
        })
        save_path = f'{save_dir}/lrt_prob_test_results_1_100.csv'
    
    data.to_csv(save_path, index=False)