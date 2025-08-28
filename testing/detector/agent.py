import gymnasium as gym

class Agent(object):
    def __init__(self):
        pass

    def reset(self):
        pass
    
    def predict(self, state, deterministic=False):
        raise NotImplementedError("act method is not implemented")


class BanditAgent(Agent):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.steps = 0
        self.id = "Default Bandit Agent"
        
    def predict(self, state, deterministic=False):
        action = self.steps % self.num_actions
        self.steps += 1
        
        return action
    
    def reset(self):
        self.steps = 0

    def compute_log_likelihood(self, transition):
        s, a, r, s_ = transition
        if a == s % self.num_actions:
            return 0
        else:
            raise ValueError("Invalid action for bandit agent")
        
        
class CustomAgent(Agent):
    def __init__(self, action_space: gym.spaces, policy: callable):
        super().__init__()
        self.policy = policy
        self.action_space = action_space
        self.steps = 0
        self.id = "Default Custom Agent"
    
    def predict(self, state, deterministic=False):
        return self.policy(self.action_space, state).rvs()
    
    def compute_log_likelihood(self, transition):
        s, a, r, s_ = transition
        
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.policy(self.action_space, s).logpmf(a)
        else:
            return self.policy(self.action_space, s).logpdf(a)
    
    
    def reset(self):
        self.steps = 0
        