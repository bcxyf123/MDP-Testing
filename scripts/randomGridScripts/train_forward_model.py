import sys
sys.path.append('../..')
from utils.utils import *
from envs.gym_randomgrid.generator import *
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataset import random_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    
def collect_trajectories(env, policy, episodes=10):
    trajectories = []
    traj_ends = []
    for i in range(episodes):
        transitions = []
        obs = env.reset()[0]
        done = False
        truncated = False
        steps = 0
        # begin token
        transition = [np.ones_like(obs)*(-1), -1, 0.0, 5]
        transitions.append(transition)
        while not (done or truncated):
            steps += 1
            action = policy[obs]
            new_obs, reward, done, truncated, info = env.step(action)
            delta_obs = map_label(new_obs - obs)
            if done or truncated:
                # end token
                obs = np.ones_like(obs)*(-1)
                action = -1
                delta_obs = 5
            transition = [obs, action, reward, delta_obs]
            transitions.append(transition)
            obs = new_obs
        trajectories.append(transitions)
        traj_ends.append(transition)
    return trajectories


def trajs_to_trans(trajectories):
    observations, actions, delta_obs_list = [], [], []
    for trj in trajectories:
        for transition in trj:
            observations.append(transition[0])
            actions.append(transition[1])
            delta_obs_list.append(transition[3])
    return np.array(observations), np.array(actions).reshape(-1, 1), np.array(delta_obs_list)

def collect_transitions(env, policy, num_steps=1000):
    obs = env.reset()[0]
    observations, actions, delta_obs_list = [], [], []
    # add start token
    observations.append(np.ones_like(obs)*(-1))
    actions.append(-1)
    delta_obs_list.append(5)
    for _ in range(num_steps):
        # action, _ = policy.predict(obs, deterministic=False)
        action = policy[obs]
        new_obs, _, done, truncated, _ = env.step(action)
        delta_obs = map_label(new_obs - obs)
        observations.append(obs)
        actions.append(action)
        delta_obs_list.append(delta_obs)
        if done or truncated:
            # add end token
            observations.append(np.ones_like(obs)*(-1))
            actions.append(-1)
            delta_obs_list.append(5)
            obs = env.reset()[0]
            # add start token
            observations.append(np.ones_like(obs)*(-1))
            actions.append(-1)
            delta_obs_list.append(5)
        else:
            obs = new_obs
        
    return np.array(observations), np.array(actions), np.array(delta_obs_list)


class TransitionModel(nn.Module):
    def __init__(self, obs_dim, action_dim, num_classes=6, hidden_size=32):
        super(TransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim+action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.output_type = 'categorical'
    
    def forward(self, obs, action):
        # x = torch.cat([obs, action], dim=-1)
        x = torch.stack([obs, action], dim=1)
        x = self.fc(x)
        output = self.softmax(x)
        return output

def map_label(delta_obs):
    if delta_obs == -8:
        return 0
    elif delta_obs == 8:
        return 1
    elif delta_obs == -1:
        return 2
    elif delta_obs == 1:
        return 3
    elif delta_obs == 0:
        return 4
    # elif delta_obs == 1000:
    #     return 5
    else:
        raise ValueError('Invalid delta_obs')

def train_transition_model(model, observations, actions, delta_observations, epochs=100, batch_size=32, save_dir=None, val_split=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # Use cross entropy loss for softmax output
    dataset = torch.utils.data.TensorDataset(torch.tensor(observations, dtype=torch.float32),
                                             torch.tensor(actions, dtype=torch.float32),
                                             torch.tensor(delta_observations, dtype=torch.long))  # Labels should be long type for CrossEntropyLoss
    
    # Split the dataset into training set and validation set
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    loss_list = []
    for epoch in range(epochs):
        for obs_batch, action_batch, delta_obs_batch in train_loader:
            prediction_batch = model(obs_batch, action_batch)
            loss = criterion(prediction_batch, delta_obs_batch)  # Compute the cross entropy loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        # print(f'Epoch {epoch}: Loss = {loss.item()}')
        
        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(obs_batch, action_batch), delta_obs_batch) for obs_batch, action_batch, delta_obs_batch in val_loader)
            print(f'Validation Loss = {val_loss / len(val_loader)}')
        model.train()

    if save_dir is None:
        save_dir = 'transition_model.pth'
    torch.save(model.state_dict(), save_dir)

@hydra.main(config_path="../../configs", config_name="gridenv")
def main(cfg: DictConfig):
    # train default model
    load_dir = os.path.join(cfg.train.load_dir, f'map_{cfg.train.policy_id}_policy_2000.csv')
    save_dir = os.path.join(cfg.train.save_dir, 'forward_model_default.pth')
    policy = read_csv(load_dir)[:, -1]
    # generate test environment
    env = genEnv(map_size=(4,8), map_id=0, start_loc=(3,0), goal_loc=(0,7), deterministic=cfg.train.deterministic, seed=222333)
    # # train forward model
    obs, actions, delta_obs = collect_transitions(env, policy, num_steps=cfg.train.collect_steps)
    # use model ensemble
    # models = [TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1) for _ in range(cfg.train.model_num)]
    transition_model = TransitionModel(obs_dim=1, action_dim=1)
    # transition_model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models',
    #                                                                 f'transition_model_0_16.pth')))
    train_transition_model(transition_model, obs, actions, delta_obs, save_dir=save_dir)

    prob_list = np.arange(0.1, 1.0, 0.01)
    for i, prob in enumerate(prob_list):
        load_dir = os.path.join(cfg.train.load_dir, f'map_0_policy_2000.csv')
        save_dir = os.path.join(cfg.train.save_dir, f'forward_model_{prob:.2f}.pth')
        policy = read_csv(load_dir)[:, -1]
        # generate test environment
        env = genEnv(map_size=(4,8), map_id=0, max_prob=prob, start_loc=(3,0), goal_loc=(0,7), deterministic=cfg.train.deterministic, seed=222333)
        # # train forward model
        obs, actions, delta_obs = collect_transitions(env, policy, num_steps=cfg.train.collect_steps)
        # use model ensemble
        # models = [TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1) for _ in range(cfg.train.model_num)]
        transition_model = TransitionModel(obs_dim=1, action_dim=1)
        transition_model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'forward_model_default.pth')))
        train_transition_model(transition_model, obs, actions, delta_obs, save_dir=save_dir)

if __name__ == "__main__":
    main()