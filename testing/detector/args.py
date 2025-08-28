import argparse

def get_args():

    parser = argparse.ArgumentParser(description="testing environment shift in RL")

    # alg
    parser.add_argument("--alg", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gamma", type=int, default=0.99, help="Discount factor")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--n-eval", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--save_policy_dir", type=str, default=None, help="Directory to save metadata")

    # environment
    parser.add_argument("--env-name", type=str, default="maze", help="environment name")
    parser.add_argument("--map-size", type=int, default=10, help="Size of the map")
    parser.add_argument("--n-goals", type=int, default=1, help="Number of goals")
    
    # detector
    parser.add_argument("--detector-type", type=str, default="value", help="Type of detector to use")
    parser.add_argument("--kernel", type=str, default="rbf", help="Type of kernel to use in mmd detection")
    parser.add_argument("--sample-method", type=str, default="MC", help="sampling method for distribution estimation")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials to run per test")
    parser.add_argument("--is_save_metadata", type=bool, default=False, help="Whether to save metadata")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save metadata")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to save model")
    parser.add_argument("--policy_path", type=str, default="model/optimal_policy.csv", help="Path to optimal policy")

    # Parse arguments
    args = parser.parse_args()

    return args
    