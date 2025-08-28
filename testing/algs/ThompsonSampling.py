import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *

class Solver:
    """
    The Solver class represents a solver for the multi-armed bandit problem.
    It implements various strategies for selecting actions and updating regrets.

    Attributes:
        bandit (Bandit): The bandit instance representing the multi-armed bandit problem.
        counts (list): A list to keep track of the number of times each action is selected.
        actions (list): A list to store the selected actions.
        regrets (list): A list to store the cumulative regrets.
        rewards (list): A list to store the rewards obtained.
    """

    def __init__(self, bandit):
        self.bandit = bandit

    def reset(self):
        """
        Resets the solver's internal state.
        """
        pass

    def update_regret(self, k, reward):
        """
        Updates the cumulative regret and stores the regret and reward values.

        Args:
            k (int): The index of the selected action.
            reward (float): The reward obtained from the selected action.
        """
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
        self.rewards.append(reward)

    def run_one_step(self):
        """
        Runs one step of the solver and returns the index of the selected action.

        Returns:
            int: The index of the selected action.
        """
        raise NotImplementedError

    def run(self, num_steps, update=True):
        """
        Runs the solver for a specified number of steps.

        Args:
            num_steps (int): The total number of steps to run.
            update (bool): Whether to update the solver's internal state.

        Raises:
            NotImplementedError: If the run_one_step method is not implemented in a subclass.
        """
        for _ in range(num_steps):
            k, r = self.run_one_step(update)
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k, r)


class ThompsonSampling(Solver):
    """
    Thompson Sampling algorithm for multi-armed bandit problem.
    """

    def __init__(self, bandit):
        super().__init__(bandit)
        self.policy_fixed = False
        self.reset()

    def reset(self):
        """
        Reset the internal state of the ThompsonSampling solver.
        """
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0.
        self.actions = []
        self.regrets = []
        self.rewards = []
        if not self.policy_fixed:
            self._a = np.ones(self.bandit.K)
            self._b = np.ones(self.bandit.K)
        self._a_bar = np.ones(self.bandit.K)
        self._b_bar = np.ones(self.bandit.K)

    def run_one_step(self, update=True):
        """
        Run one step of the Thompson Sampling algorithm.

        Args:
            update (bool): Whether to update the internal state based on the chosen action and reward.

        Returns:
            int: The index of the chosen action.
            float: The reward obtained from the chosen action.
        """
        samples = np.random.beta(self._a, self._b)  # Sample a set of reward samples from the Beta distribution
        k = np.argmax(samples)  # Choose the action with the highest sampled reward
        r = self.bandit.step(k)
        if update:
            self.update_beta_pdf(k, r)
        self.update_bernoulli_pdf(k, r)
        return k, r
    
    def choose_action(self):
        """
        Choose an action based on the probabilities calculated from the normalized self._a_bar.

        Returns:
            int: The index of the chosen action.
        """
        probabilities = self._a_bar / self._a_bar.sum()  # Normalize self._a_bar to get probabilities
        k = np.random.choice(np.arange(len(probabilities)), p=probabilities)  # Choose an action based on the probabilities
        return k

    def update_beta_pdf(self, k, r):
        """
        Update the Beta distribution parameters based on the chosen action and reward.

        Args:
            k (int): The index of the chosen action.
            r (float): The reward obtained from the chosen action.
        """
        self._a[k] += r
        self._b[k] += (1 - r)
    
    def update_bernoulli_pdf(self, k, r):
        """
        Update the Bernoulli distribution parameters based on the chosen action and reward.

        Args:
            k (int): The index of the chosen action.
            r (float): The reward obtained from the chosen action.
        """
        self._a_bar[k] += r
        self._b_bar[k] += (1 - r)
    
    def calculate_bernoulli_pdf(self, k, r):
        """
        Calculate the probability of a Bernoulli distribution based on the chosen action and reward.

        Args:
            k (int): The index of the chosen action.
            r (float): The reward obtained from the chosen action.

        Returns:
            float: The probability of the Bernoulli distribution.
        """
        prob = (self._a_bar[k]-1)/(self._a_bar[k] + self._b_bar[k]-2) if r == 1 else 1-(self._a_bar[k]-1)/(self._a_bar[k] + self._b_bar[k]-2)
        return prob


def plot_results(solvers, solver_names):
    """
    Plots the cumulative regrets of different solvers over time.

    Args:
        solvers (list): A list of solver objects.
        solver_names (list): A list of names corresponding to the solvers.

    Returns:
        None
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

    