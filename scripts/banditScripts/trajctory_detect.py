
import sys
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import *

from envs.bandit.bandit import BernoulliBandit
from solverScript import ThompsonSampling

def trajectory_test(default_p=0.1, test_p=0.1, n_trials=100, n_steps=10000):

    # print(f'{test_p}')
    default_bandit = BernoulliBandit(p=default_p)
    default_solver = ThompsonSampling(default_bandit)
    test_bandit = BernoulliBandit(p=test_p)
    test_solver = ThompsonSampling(test_bandit)

    ratio_list = []
    log_ratio_list = []
    likelihood0_list = []
    likelihood1_list = []

    default_solver.run(20)
    test_solver._a = default_solver._a
    test_solver._b = default_solver._b
    default_solver.policy_fixed = True
    test_solver.policy_fixed = True
    default_solver.reset()

    for _ in range(n_trials):

        test_solver.run(n_steps, update=False)
        # print(test_solver._a, test_solver._b)
        # print(test_solver._a_bar, test_solver._b_bar)
        actions = test_solver.actions
        rewards = test_solver.rewards
        # use default env model
        likelihood0 = (sum(np.log(default_bandit.probs[a] if r == 1 else 1 - default_bandit.probs[a]) 
                           for a, r in zip(actions, rewards) if r in [0, 1]))
        # use frequency estimation
        likelihood1 = (sum(np.log(test_solver.calculate_bernoulli_pdf(a, r)) 
                           for a, r in zip(actions, rewards) if r in [0, 1]))
        likelihood0_list.append(likelihood0)
        likelihood1_list.append(likelihood1)
        ratio = -2*(likelihood0 - likelihood1)
        if ratio == 0:
            ratio += 1e-10
        ratio_list.append(ratio)
        log_ratio_list.append(np.log(ratio))
        test_solver.reset()

    power = chi_test(ratio_list, df=5)

    # plot histogram
    # plt.figure()
    # # plt.hist(ratio_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$\Lambda$')
    # sns.distplot(ratio_list, kde=False, fit=chi2, hist_kws={'edgecolor':'black', 'linewidth':1}, label='$\Lambda$')
    # plt.xlabel('likelihood ratio')
    # plt.ylabel('density')
    # plt.legend()
    # plt.savefig(f'ratio_{n_trials}.pdf', dpi=300)
    # plt.savefig(f'ratio_{n_trials}.png', dpi=300)

    # # plot histogram
    # plt.figure()
    # plt.hist(likelihood0_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$\ell_0$')
    # # plt.title('likelihood 0')
    # plt.xlabel('likelihood')
    # plt.ylabel('density')
    # plt.legend()
    # plt.savefig(f'likelihood0_{n_trials}.pdf', dpi=300)

    # # plot histogram
    # plt.figure()
    # plt.hist(ratio_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$\ell_1$')
    # # plt.title('likelihood_1')
    # plt.xlabel('likelihood')
    # plt.ylabel('density')
    # plt.legend()
    # plt.savefig(f'likelihood1_{n_trials}.pdf', dpi=300)

    # print(f'likelihood0 {np.mean(likelihood0_list):.3f}', f'{np.std(likelihood0_list):.3f}')
    # print(f'likelihood1 {np.mean(likelihood1_list):.3f}', f'{np.std(likelihood1_list):.3f}')
    # print(f'ratio {np.mean(ratio_list):.3f}', f'{np.std(ratio_list):.3f}')
    # print(f'power {power}')
    # print()

    return likelihood0_list, likelihood1_list, ratio_list, log_ratio_list, power


def trajectory_test_fixed(default_p=0.1, test_p=0.1, n_trials=100):

    print(f'{test_p}')
    default_bandit = BernoulliBandit(p=default_p)
    default_solver = ThompsonSampling(default_bandit)

    test_bandit = BernoulliBandit(p=test_p)
    test_solver = ThompsonSampling(test_bandit)

    ratio_list = []
    log_ratio_list = []
    likelihood0_list = []
    likelihood1_list = []
    for _ in range(n_trials):
        # use samples from test bandit
        default_solver.run(10000)
        actions = default_solver.actions
        rewards = default_solver.rewards
        # use default env model
        likelihood0 = (sum(np.log(default_bandit.probs[a] if r == 1 else 1 - default_bandit.probs[a]) 
                           for a, r in zip(actions, rewards) if r in [0, 1]))
        # use frequency estimation
        likelihood1 = (sum(np.log(default_solver.calculate_bernoulli_pdf(a, r)) 
                           for a, r in zip(actions, rewards) if r in [0, 1]))
        likelihood0_list.append(likelihood0)
        likelihood1_list.append(likelihood1)
        ratio = -2*(likelihood0 - likelihood1)
        ratio_list.append(ratio)
        log_ratio_list.append(np.log(ratio))
        default_solver.reset()
        test_solver.reset()

    power = chi_test(ratio_list, df=5)

    # plot histogram
    plt.figure()
    plt.hist(ratio_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$\Lambda$')
    # plt.title('likelihood ratio distribution')
    plt.xlabel('likelihood ratio')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(f'ratio_{n_trials}.pdf', dpi=300)

    # plot histogram
    plt.figure()
    plt.hist(likelihood0_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$\ell_0$')
    # plt.title('likelihood 0')
    plt.xlabel('likelihood')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(f'likelihood0_{n_trials}.pdf', dpi=300)

    # plot histogram
    plt.figure()
    plt.hist(ratio_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$\ell_1$')
    # plt.title('likelihood_1')
    plt.xlabel('likelihood')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(f'likelihood1_{n_trials}.pdf', dpi=300)

    print(f'likelihood0 {np.mean(likelihood0_list):.3f}', f'{np.std(likelihood0_list):.3f}')
    print(f'likelihood1 {np.mean(likelihood1_list):.3f}', f'{np.std(likelihood1_list):.3f}')
    print(f'ratio {np.mean(ratio_list):.3f}', f'{np.std(ratio_list):.3f}')
    print(f'power {power}')

    return likelihood0_list, likelihood1_list, ratio_list, log_ratio_list, power


def main():

    # trajectory_test(default_p=0.1, test_p=0.1, n_trials=200, n_steps=2000)

    p_list = np.arange(0.9, 0.1, -0.01)
    log_ratio_means = []
    log_ratio_stds = []
    likelihood0_means = []
    likelihood0_stds = []
    likelihood1_means = []
    likelihood1_stds = []
    powers = []

    for test_p in p_list:
        likelihood0_list, likelihood1_list, ratio_list, log_ratio_list, power = trajectory_test(default_p=0.9, test_p=test_p, n_trials=100, n_steps=500)
        
        print(test_p, power)
        likelihood0_means.append(np.mean(likelihood0_list))
        likelihood0_stds.append(np.std(likelihood0_list))
        likelihood1_means.append(np.mean(likelihood1_list))
        likelihood1_stds.append(np.std(likelihood1_list))
        log_ratio_means.append(np.mean(log_ratio_list))
        log_ratio_stds.append(np.std(log_ratio_list))
        powers.append(power)

    data = {
        'likelihood0_means': likelihood0_means,
        'likelihood0_stds': likelihood0_stds,
        'likelihood1_means': likelihood1_means,
        'likelihood1_stds': likelihood1_stds,
        'log_ratio_means': log_ratio_means,
        'log_ratio_stds': log_ratio_stds,
        'powers': powers
    }
    df = pd.DataFrame(data)
    df.to_csv('lrt.csv', index=False)

    
    # df = pd.read_csv('tables/trajectory.csv')

    # likelihood0_means = df['likelihood0_means']
    # n = len(likelihood0_means)//2
    # likelihood0_means = df['likelihood0_means'][:n]
    # likelihood0_stds = df['likelihood0_stds'][:n]
    # likelihood1_means = df['likelihood1_means'][:n]
    # likelihood1_stds = df['likelihood1_stds'][:n]
    # log_ratio_means = df['log_ratio_means'][:n]
    # log_ratio_stds = df['log_ratio_stds'][:n]
    # powers = df['powers'][:n]

    # p_list = p_list[:n]

    # plt.figure()
    # plt.plot(p_list, likelihood0_means, label='$\ell_0$')
    # plt.fill_between(p_list, np.array(likelihood0_means) - np.array(likelihood0_stds), np.array(likelihood0_means) + np.array(likelihood0_stds), alpha=0.5)
    # plt.plot(p_list, likelihood1_means, label='$\ell_1$')
    # plt.fill_between(p_list, np.array(likelihood1_means) - np.array(likelihood1_stds), np.array(likelihood1_means) + np.array(likelihood1_stds), alpha=0.5)
    # # plt.title('likelihood distribution')
    # plt.xlabel('arm probability $p$')
    # plt.ylabel('likelihood $\ell$')
    # # plt.ylim(-6950, -6900)
    # plt.legend()
    # plt.savefig(f'figs/pdf/traj_likelihood.pdf', dpi=300)
    # plt.savefig(f'figs/png/traj_likelihood.png', dpi=300)

    # plt.figure()
    # plt.plot(p_list, log_ratio_means, label='$\log(\Lambda)$')
    # plt.fill_between(p_list, np.array(log_ratio_means) - np.array(log_ratio_stds), np.array(log_ratio_means) + np.array(log_ratio_stds), alpha=0.5)
    # # plt.title('log likelihood ratio distribution')
    # plt.xlabel('arm probability')
    # plt.ylabel('log likelihood ratio $\log(\Lambda)$')
    # # plt.ylim(0, 5)
    # plt.legend()
    # plt.savefig(f'figs/pdf/traj_ratio.pdf', dpi=300)
    # plt.savefig(f'figs/png/traj_ratio.png', dpi=300)

    # plt.figure()
    # plt.plot(p_list, powers, label='power')
    # # plt.title('power')
    # plt.xlabel('arm probability $p$')
    # plt.ylabel('power')
    # # plt.ylim(0, 1)
    # plt.legend()
    # plt.savefig(f'figs/pdf/traj_power.pdf', dpi=300)
    # plt.savefig(f'figs/png/traj_power.png', dpi=300)


def traj_single_test(n_trials=10):
    default_p = 0.9
    test_p = 0.9

    print(n_trials)
    powers = []
    # for _ in range(100):
    likelihood0_list, likelihood1_list, ratio_list, log_ratio_list, power = trajectory_test(default_p=default_p, test_p=test_p, n_trials=n_trials, n_steps=500)
    powers.append(power)
    
    print(f'power {np.mean(powers):.3f}')

    # plt.hist(ratio_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$\Lambda$')
    sns.histplot(ratio_list, kde=False, stat='density', alpha=0.5, edgecolor='black', linewidth=0.5, label='$\Lambda$')

    # 获取当前的matplotlib轴
    plt_axis = plt.gca()

    # 生成卡方分布的理论 PDF 曲线
    x = np.linspace(0, max(ratio_list), 1000)
    y = chi2.pdf(x, df=5)  # 假设自由度为 2

    # 使用 matplotlib 绘制卡方分布的 PDF
    sns.lineplot(x=x, y=y, linewidth=1.5)

    plt.xlabel('likelihood ratio')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(f'figs/pdf/ratio_{n_trials}.pdf', dpi=300)
    plt.show()

    return powers



class bandit_lrt_tester(lrt_tester):
    def __init__(self, n_trials=10, default_prob=0.9, n_steps=500,):
        super().__init__(n_trials,)
        self.default_p = default_prob
        self.n_steps = n_steps

    def sample(
            self, 
            is_save_metadata: bool = False,
            test_p: float = 0.9,
        ) -> np.ndarray:
        
        default_bandit = BernoulliBandit(p=self.default_p)
        default_solver = ThompsonSampling(default_bandit)
        test_bandit = BernoulliBandit(p=test_p)
        test_solver = ThompsonSampling(test_bandit)

        ratio_list = []
        log_ratio_list = []
        likelihood0_list = []
        likelihood1_list = []

        default_solver.run(self.n_steps)
        test_solver._a = default_solver._a
        test_solver._b = default_solver._b
        default_solver.policy_fixed = True
        test_solver.policy_fixed = True
        default_solver.reset()

        for _ in range(self.n_trials):

            test_solver.run(self.n_steps, update=False)
            # print(test_solver._a, test_solver._b)
            # print(test_solver._a_bar, test_solver._b_bar)
            actions = test_solver.actions
            rewards = test_solver.rewards
            # use default env model
            likelihood0 = (sum(np.log(default_bandit.probs[a] if r == 1 else 1 - default_bandit.probs[a]) 
                            for a, r in zip(actions, rewards) if r in [0, 1]))
            # use frequency estimation
            likelihood1 = (sum(np.log(test_solver.calculate_bernoulli_pdf(a, r)) 
                            for a, r in zip(actions, rewards) if r in [0, 1]))
            likelihood0_list.append(likelihood0)
            likelihood1_list.append(likelihood1)
            ratio = -2*(likelihood0 - likelihood1)
            if ratio == 0:
                ratio += 1e-10
            ratio_list.append(ratio)
            log_ratio_list.append(np.log(ratio))
            test_solver.reset()

        return np.array(ratio_list)
    
    def dist_estimation(
            self, 
            n_samples: int
        ) -> tuple[float, float]:

        chi_stats = []
        for _ in range(n_samples):
            ratio_arr = self.sample(test_p=self.default_p,)
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
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, )

        return reject_region

    def run_test(
            self, 
            reject_region: tuple[float, float],
            test_p: float,
        ) -> bool:

        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        ratio_arr = self.sample(test_p=test_p, is_save_metadata=True,)
        chi_stat = np.mean(ratio_arr)
        if chi_stat < reject_region[0] or chi_stat > reject_region[1]:
            # print(f't: {t:.4f}, reject')
            return True
        else:
            # print(f't: {t:.4f}, accept')
            return False
    
    def power_analysis(
            self, 
            reject_region: tuple[float, float],
            test_p: float,
        ) -> float:

        powers = []
        for _ in range(100):
            powers.append(self.run_test(reject_region, test_p, ))
        power = sum(powers) / 100

        return power


if __name__ == '__main__':
    prob = np.arange(0.9, 0.1, -0.01)

    tester = bandit_lrt_tester(n_trials=10, n_steps=500, default_prob=0.9)

    reject_region = tester.dist_estimation(n_samples=1000, )
    powers = []
    for p in prob:
        power = tester.power_analysis(reject_region, test_p=p, )
        powers.append(power)
        print(f'prob: {p}, power: {power:.3f}')
        
    data = pd.DataFrame({
        'prob': prob, 
        'power': powers,
        })
    save_path = 'data/lrt_test_results_new.csv'
    data.to_csv(save_path, index=False)