import sys
sys.path.append('../..')
from matplotlib import pyplot as plt

from envs.bandit.bandit import BernoulliBandit
from solverScript import ThompsonSampling

def bandit_run():
    bandit = BernoulliBandit(p=0.9)
    solver = ThompsonSampling(bandit)
    solver.run(500)

    plt.rcParams.update({'font.size': 12})
    plt.plot(solver.regrets)
    plt.xlabel('Steps')
    plt.ylabel('Regret')
    # plt.show()
    plt.savefig('train_regrets.pdf', dpi=300)


if __name__ == '__main__':
    bandit_run()