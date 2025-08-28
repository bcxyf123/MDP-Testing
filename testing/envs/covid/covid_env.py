# ── testing/envs/covid_env.py ─────────────────────────────────────────
import types
import numpy as np
from gymnasium import Env, spaces
from scipy.stats import poisson

class CovidEnv(Env):
    """
    单条 comuna 的病例时间序列 → 奖励为当天新增病例
    action_space 只有一个哑动作（策略固定）
    """
    def __init__(self, comuna_df,
                 max_steps: int = 30,
                 random_start: bool = True,
                 env_id: str = "covid"):
        super().__init__()

        self.comuna_df = comuna_df.reset_index(drop=True)
        self.max_steps     = max_steps
        self.random_start  = random_start

        # 允许随机起点的最大索引
        self.max_possible_start = max(0, len(self.comuna_df) - max_steps - 1)
        self.start_day = 0            # 会在 reset() 里更新
        self.cur_step  = 0

        # gym 接口
        self.action_space      = spaces.Discrete(1)
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=(1,), dtype=np.float32)

        # ★ sample() 依赖 env.spec.id
        self.spec = types.SimpleNamespace(id=env_id)

    # --------- 必要接口 ---------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 每次 episode 随机选一个起点，保证轨迹多样性
        if self.random_start and self.max_possible_start > 0:
            self.start_day = np.random.randint(0, self.max_possible_start + 1)
        else:
            self.start_day = 0
        self.cur_step = self.start_day
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.cur_step + 1 >= len(self.comuna_df):
            done   = True
            reward = 0.0
            return self._get_obs(), reward, done, False, {}

        # reward = max(
        #     self.comuna_df.loc[self.cur_step + 1, "daily_cases"]
        #   - self.comuna_df.loc[self.cur_step,     "daily_cases"],
        #     0.0)

        reward = self.comuna_df.loc[self.cur_step + 1, "daily_cases"] 
        # - self.comuna_df.loc[self.cur_step, "daily_cases"]


        self.cur_step += 1
        done = (self.cur_step - self.start_day) >= self.max_steps
        return self._get_obs(), reward, done, False, {}

    # --------- 工具函数 ---------
    def _get_obs(self):
        # 这里的观测值暂时就是“当前日期索引”
        return np.array([self.cur_step], dtype=np.float32)

    def compute_log_likelihood(self, transition):
        # 简单示例：假设奖励服从poisson distribution
        s, a, r, s_ = transition
        # Poisson 的 λ 是预测的 cases 数（必须 > 0）
        lam = max(self.comuna_df.loc[int(s_[0]), "daily_cases"], 1e-6)
        # 观测值 r 必须是非负整数
        r_int = max(int(round(r)), 0)
        return poisson(lam).logpmf(r_int)
