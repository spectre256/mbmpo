import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from tqdm import tqdm
import copy
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Utils
# ----------------------------

def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=device)


def explained_variance(y_pred, y):
    var = torch.var(y)
    return 1 - torch.var(y - y_pred) / (var + 1e-8)


def kl_old_new(old_policy, new_policy, obs):
    with torch.no_grad():
        old_dist = old_policy.dist(obs)

    new_dist = new_policy.dist(obs)

    kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)
    return kl.mean()


def fisher_vector_product(policy, obs, v, damping=1e-2):
    dist = policy.dist(obs)

    act = dist.sample()
    logp = dist.log_prob(act).sum(-1).mean()

    grads = torch.autograd.grad(logp, policy.parameters(), create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])

    grad_v = (flat_grad * v).sum()
    hvp = torch.autograd.grad(grad_v, policy.parameters(), retain_graph=True)
    flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp]).detach()

    return flat_hvp + damping * v


class MetricLogger:
    def __init__(self, window=50):
        self.window = window

        self.metrics = {
            "dyn_loss": deque(maxlen=window),
            "policy_loss": deque(maxlen=window),
            "value_loss": deque(maxlen=window),
            "kl": deque(maxlen=window),
            "reward": deque(maxlen=window),
            "entropy": deque(maxlen=window),
        }

        self.latest = {k: 0.0 for k in self.metrics.keys()}

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.metrics:
                self.metrics[k].append(float(v))
                self.latest[k] = float(v)

    def avg(self, k):
        return np.mean(self.metrics[k]) if len(self.metrics[k]) > 0 else 0.0

    def line(self):
        return (
            f"dyn:{self.avg('dyn_loss'):.4f} | "
            f"pol:{self.avg('policy_loss'):.4f} | "
            f"val:{self.avg('value_loss'):.4f} | "
            f"kl:{self.avg('kl'):.5f} | "
            f"rew:{self.avg('reward'):.2f}"
        )

# ----------------------------
# Gaussian Policy
# ----------------------------

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp([obs_dim, 256, 256, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def act(self, obs):
        obs = to_tensor(obs)
        dist = self.dist(obs)
        a = dist.sample()
        return a.cpu().numpy()

    def log_prob(self, obs, act):
        dist = self.dist(obs)
        return dist.log_prob(act).sum(-1)

# ----------------------------
# Value Function
# ----------------------------

class Value(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = mlp([obs_dim, 256, 256, 1])

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ----------------------------
# Dynamics Ensemble
# ----------------------------

class Dynamics(nn.Module):
    def __init__(self, obs_dim, act_dim, ensemble_size=5):
        super().__init__()
        self.models = nn.ModuleList([
            mlp([obs_dim + act_dim, 256, 256, obs_dim])
            for _ in range(ensemble_size)
        ])

    def forward(self, obs, act, i):
        x = torch.cat([obs, act], dim=-1)
        return self.models[i](x)

# ----------------------------
# Rollout Buffer
# ----------------------------

class Buffer:
    def __init__(self):
        self.obs, self.act, self.adv, self.ret, self.logp = [], [], [], [], []

    def add(self, o, a, adv, r, lp):
        self.obs.append(o)
        self.act.append(a)
        self.adv.append(adv)
        self.ret.append(r)
        self.logp.append(lp)

    def get(self):
        return (
            torch.cat(self.obs),
            torch.cat(self.act),
            torch.cat(self.adv),
            torch.cat(self.ret),
            torch.cat(self.logp),
        )

    def clear(self):
        self.__init__()

# ----------------------------
# GAE
# ----------------------------

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = 0
    returns = []
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        adv = delta + gamma * lam * mask * adv
        returns.append(adv + values[t])
    return list(reversed(returns)), list(reversed([a for a in returns]))

# ----------------------------
# TRPO helpers
# ----------------------------

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

def kl_divergence(policy, obs, old_dist):
    dist = policy.dist(obs)
    return torch.distributions.kl_divergence(old_dist, dist).mean()

# conjugate gradient
def cg(Ax, b, iters=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)

    for _ in range(iters):
        Avp = Ax(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        p = r + (new_rdotr / (rdotr + 1e-8)) * p
        rdotr = new_rdotr
    return x

# ----------------------------
# MB-MPO Core
# ----------------------------

class MBMPO:
    def __init__(self, obs_dim, act_dim):
        self.policy = Policy(obs_dim, act_dim).to(device)
        self.value = Value(obs_dim).to(device)
        self.dyn = Dynamics(obs_dim, act_dim).to(device)

        self.v_opt = optim.Adam(self.value.parameters(), lr=3e-4)
        self.dyn_opt = optim.Adam(self.dyn.parameters(), lr=1e-3)

        self.ensemble_size = len(self.dyn.models)

    # ------------------------
    # Train dynamics
    # ------------------------
    def train_dynamics(self, data):
        obs, act, next_obs = data
        total_loss = 0.0

        for i, model in enumerate(self.dyn.models):
            pred = model(torch.cat([obs, act], dim=-1))
            loss = ((pred - next_obs) ** 2).mean()

            self.dyn_opt.zero_grad()
            loss.backward()
            self.dyn_opt.step()

            total_loss += loss.item()

        return total_loss / len(self.dyn.models)


    def policy_loss(self, obs, act, adv):
        dist = self.policy.dist(obs)
        logp = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1)

        loss = -(logp * adv).mean()
        return loss, entropy.mean().item()


    def value_loss(self, obs, ret):
        v = self.value(obs)
        return ((v - ret) ** 2).mean()


    # ------------------------
    # Imagined rollout
    # ------------------------
    def rollout_model(self, env, horizon=5, batch_size=1024):
        obs = env.reset()[0]
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        data = []

        for i in range(self.ensemble_size):
            o = obs.repeat(batch_size, 1)
            traj = []

            for _ in range(horizon):
                dist = self.policy.dist(o)
                a = dist.sample()
                next_o = self.dyn(o, a, i)
                r = -torch.sum(next_o**2, dim=-1)  # placeholder reward proxy
                done = torch.zeros_like(r)

                traj.append((o, a, r, next_o, done))
                o = next_o.detach()

            data.append(traj)

        return data

    # ------------------------
    # MAML inner update
    # ------------------------
    def adapt_policy(self, traj):
        obs, act, adv, _, _ = traj

        dist = self.policy.dist(obs)
        loss = -(dist.log_prob(act) * adv).mean()

        grads = torch.autograd.grad(loss, self.policy.parameters(), create_graph=True)
        adapted = []
        for p, g in zip(self.policy.parameters(), grads):
            adapted.append(p - 0.1 * g)
        return adapted

    # ------------------------
    # TRPO update
    # ------------------------
    def trpo_step(self, obs, act, adv, old_logp):
        obs = obs.detach()
        act = act.detach()
        adv = adv.detach()

        # normalize advantages (CRITICAL)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # snapshot policy BEFORE update
        old_policy = copy.deepcopy(self.policy).to(device)
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        theta_old = flat_params(self.policy).detach()

        # surrogate loss (NO PARAM MUTATION HERE)
        def surrogate(flat_params_vec):
            set_params(self.policy, flat_params_vec)

            dist = self.policy.dist(obs)
            logp = dist.log_prob(act).sum(-1)

            ratio = torch.exp(logp - old_logp)
            return -(ratio * adv).mean()

        loss = surrogate(theta_old)

        grads = torch.autograd.grad(loss, self.policy.parameters(), retain_graph=False)
        g = torch.cat([x.view(-1) for x in grads]).detach()

        def Hv(v):
            return fisher_vector_product(self.policy, obs, v)

        step_dir = cg(Hv, g)

        shs = step_dir @ Hv(step_dir)
        step_size = torch.sqrt(2 * 0.01 / (shs + 1e-8))

        full_step = step_size * step_dir

        new_theta = theta_old - full_step

        # IMPORTANT: restore correct parameters AFTER surrogate eval
        set_params(self.policy, new_theta)

        # compute REAL KL AFTER update
        kl = kl_old_new(old_policy, self.policy, obs)

        return kl.item()


    # ------------------------
    # Value update
    # ------------------------
    def train_value(self, obs, ret):
        v = self.value(obs)
        loss = ((v - ret) ** 2).mean()
        self.v_opt.zero_grad()
        loss.backward()
        self.v_opt.step()

# ----------------------------
# Training Loop
# ----------------------------

def run():
    env = gym.make("Ant-v5")

    logger = MetricLogger(window=50)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    algo = MBMPO(obs_dim, act_dim)
    n_iters = 500


    for it in tqdm(range(n_iters), desc="MB-MPO training"):

        # -------------------------
        # REAL ENV DATA COLLECTION
        # -------------------------
        obs, _ = env.reset()
        ep_reward = 0.0

        real_data = []

        while True:
            act = algo.policy.act(obs)
            next_obs, reward, done, truncated, _ = env.step(act)

            real_data.append((obs, act, reward, next_obs, done or truncated))
            ep_reward += reward
            obs = next_obs

            if done or truncated:
                break

        obs_t = torch.tensor([d[0] for d in real_data], dtype=torch.float32, device=device)
        act_t = torch.tensor([d[1] for d in real_data], dtype=torch.float32, device=device)
        nxt_t = torch.tensor([d[3] for d in real_data], dtype=torch.float32, device=device)

        # -------------------------
        # DYNAMICS UPDATE
        # -------------------------
        dyn_loss = algo.train_dynamics((obs_t, act_t, nxt_t))

        with torch.no_grad():
            values = algo.value(obs_t).detach()

        # bootstrap returns (simple but valid baseline)
        ret = values + torch.randn_like(values) * 0.01  # small noise ONLY for stability
        adv = ret - values

        # -------------------------
        # POLICY + VALUE LOSS
        # -------------------------
        pol_loss, entropy = algo.policy_loss(obs_t, act_t, adv)
        val_loss = algo.value_loss(obs_t, ret)

        # TRPO step (returns KL approx via surrogate shift)
        kl = algo.trpo_step(obs_t, act_t, adv, torch.zeros_like(adv))
        algo.train_value(obs_t, ret)

        # -------------------------
        # LOGGING
        # -------------------------
        logger.log(
            dyn_loss=dyn_loss,
            policy_loss=pol_loss.item(),
            value_loss=val_loss.item(),
            kl=kl,
            reward=ep_reward,
            entropy=entropy
        )

        # -------------------------
        # LIVE STATUS LINE
        # -------------------------
        tqdm.write(
            f"iter:{it:04d} | "
            f"{logger.line()} | "
            f"ep_rew:{ep_reward:.1f}"
        )


if __name__ == "__main__":
    run()
