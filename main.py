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
    old_dist = policy.dist(obs)
    old_mu = old_dist.loc.detach()
    old_std = old_dist.scale.detach()

    new_dist = policy.dist(obs)
    kl = torch.distributions.kl_divergence(Normal(old_mu, old_std), new_dist).mean()

    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
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
# Gaussian Policy (With Functional Graph Support)
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

    def functional_dist(self, obs, params=None):
        if params is None:
            return self.dist(obs)
            
        x = obs
        x = torch.nn.functional.linear(x, params['net.0.weight'], params['net.0.bias'])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, params['net.2.weight'], params['net.2.bias'])
        x = torch.nn.functional.relu(x)
        mu = torch.nn.functional.linear(x, params['net.4.weight'], params['net.4.bias'])
        
        std = torch.exp(params['log_std'])
        return Normal(mu, std)

    def act(self, obs, params=None):
        obs = to_tensor(obs)
        with torch.set_grad_enabled(params is not None):
            dist = self.functional_dist(obs, params)
            a = dist.sample()
        return a.detach().cpu().numpy() if params is None else a

    def log_prob(self, obs, act, params=None):
        dist = self.functional_dist(obs, params)
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
# Dynamics Ensemble (Shift 3: State Delta Prediction Architecture)
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
        # Shift 3 Principle: Outputs current state + delta predicted change
        return obs + self.models[i](x)

# ----------------------------
# GAE
# ----------------------------

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advs = []
    returns = []
    adv = 0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        adv = delta + gamma * lam * mask * adv
        advs.append(adv)
        returns.append(adv + values[t])
    return list(reversed(returns)), list(reversed(advs))


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

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
        self.inner_lr = 1e-3 

    def train_dynamics(self, real_data):
        obs_t = torch.tensor(np.array([d[0] for d in real_data]), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array([d[1] for d in real_data]), dtype=torch.float32, device=device)
        next_obs = torch.tensor(np.array([d[3] for d in real_data]), dtype=torch.float32, device=device)

        # Shift 3 Principle: Calculate loss targets based on state differences (deltas)
        delta_targets = next_obs - obs_t

        total_loss = 0.0
        for i in range(self.ensemble_size):
            x = torch.cat([obs_t, act_t], dim=-1)
            
            # --- FIX HERE: Index into the ModuleList properly ---
            pred_delta = self.dyn.models[i](x)  
            
            loss = ((pred_delta - delta_targets) ** 2).mean()

            self.dyn_opt.zero_grad()
            loss.backward()
            self.dyn_opt.step()

            total_loss += loss.item()

        return total_loss / self.ensemble_size

    # ------------------------
    # Task-Specific Simulator Rollout Loop (With Shift 3 Reality Grounding)
    # ------------------------
    def rollout_single_model(self, model_idx, init_obs, params=None, horizon=20, batch_size=1000):
        o = init_obs.repeat(batch_size, 1)
        
        # Track simulated done status per parallel thread batch
        batch_dones = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        obs_seq, act_seq, rew_seq, done_seq = [], [], [], []

        for _ in range(horizon):
            if params is not None:
                dist = self.policy.functional_dist(o, params)
                a = dist.sample()
            else:
                with torch.no_grad():
                    dist = self.policy.dist(o)
                    a = dist.sample()

            next_o = self.dyn(o, a, model_idx)
            
            # ---------------------------------------------
            # Shift 3: Analytical Ant-v5 Reward Logic Implementation
            # ---------------------------------------------
            forward_velocity = next_o[:, 13]  # Index 13 tracks forward velocity in Ant-v5
            control_cost = 0.5 * torch.sum(a ** 2, dim=-1)
            
            # Torso height is checked at index 0 to see if it remains standing
            torso_height = next_o[:, 0]
            is_healthy = (torso_height > 0.2) & (torso_height < 1.0)
            healthy_reward = torch.where(is_healthy, torch.ones_like(torso_height), torch.zeros_like(torso_height))
            
            r = forward_velocity - control_cost + healthy_reward
            
            # ---------------------------------------------
            # Shift 3: Done Boundary Termination Masking
            # ---------------------------------------------
            # If healthy bounds are crossed, flag done status for those indices
            step_dones = torch.where(is_healthy, torch.zeros_like(torso_height), torch.ones_like(torso_height))
            batch_dones = torch.max(batch_dones, step_dones)  # Lock in a done state once hit

            obs_seq.append(o)
            act_seq.append(a)
            rew_seq.append(r)
            done_seq.append(batch_dones.clone())
            
            o = next_o

        return (
            torch.stack(obs_seq, dim=0), 
            torch.stack(act_seq, dim=0), 
            torch.stack(rew_seq, dim=0), 
            torch.stack(done_seq, dim=0)
        )

    def adapt_policy(self, model_idx, init_obs, horizon=20, batch_size=1000):
        obs, act, rew, done = self.rollout_single_model(model_idx, init_obs, params=None, horizon=horizon, batch_size=batch_size)
        
        obs_flat = obs.view(-1, obs.shape[-1])
        act_flat = act.view(-1, act.shape[-1])

        with torch.no_grad():
            vals = self.value(obs_flat).view(obs.shape[0], obs.shape[1]).cpu().numpy()
            val_seq = np.vstack([vals, np.zeros((1, batch_size))])
        
        rew_np = rew.detach().cpu().numpy()
        done_np = done.detach().cpu().numpy()

        all_advs = []
        for b in range(batch_size):
            _, advs = compute_gae(rew_np[:, b], val_seq[:, b], done_np[:, b])
            all_advs.append(advs)
        
        adv_tensor = torch.tensor(np.array(all_advs).T, dtype=torch.float32, device=device).reshape(-1)
        adv_norm = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        logp = self.policy.log_prob(obs_flat, act_flat, params=None)
        inner_loss = -(logp * adv_norm).mean()

        grads = torch.autograd.grad(inner_loss, self.policy.parameters(), create_graph=True)

        adapted_params = {}
        for (name, p), g in zip(self.policy.named_parameters(), grads):
            adapted_params[name] = p - self.inner_lr * g

        return adapted_params

    def meta_trpo_step(self, meta_obs, meta_act, meta_adv, meta_old_logp, adapted_params_list):
        meta_obs = meta_obs.detach()
        meta_act = meta_act.detach()
        meta_adv = meta_adv.detach()
        meta_adv = (meta_adv - meta_adv.mean()) / (meta_adv.std() + 1e-8)

        theta_old = flat_params(self.policy).detach()

        def meta_surrogate(flat_params_vec):
            set_params(self.policy, flat_params_vec)
            
            total_surr = 0.0
            idx_start = 0
            samples_per_model = meta_obs.shape[0] // self.ensemble_size

            for i in range(self.ensemble_size):
                idx_end = idx_start + samples_per_model
                
                m_obs = meta_obs[idx_start:idx_end]
                m_act = meta_act[idx_start:idx_end]
                m_adv = meta_adv[idx_start:idx_end]
                m_old_logp = meta_old_logp[idx_start:idx_end]

                logp = self.policy.log_prob(m_obs, m_act, params=adapted_params_list[i])
                ratio = torch.exp(logp - m_old_logp)
                total_surr += -(ratio * m_adv).mean()
                
                idx_start = idx_end

            return total_surr / self.ensemble_size

        loss = meta_surrogate(theta_old)

        grads = torch.autograd.grad(loss, self.policy.parameters(), retain_graph=False)
        g = torch.cat([x.view(-1) for x in grads]).detach()

        def Hv(v):
            return fisher_vector_product(self.policy, meta_obs, v)

        step_dir = cg(Hv, g)
        shs = step_dir @ Hv(step_dir)
        step_size = torch.sqrt(2 * 0.01 / (shs + 1e-8))
        full_step = step_size * step_dir

        new_theta = theta_old - full_step
        set_params(self.policy, new_theta)

        kl = kl_old_new(copy.deepcopy(self.policy), self.policy, meta_obs)
        return kl.item(), loss.item()

    def train_value(self, obs, ret):
        v = self.value(obs)
        loss = ((v - ret) ** 2).mean()
        self.v_opt.zero_grad()
        loss.backward()
        self.v_opt.step()

    def value_loss(self, obs, ret):
        v = self.value(obs)
        return ((v - ret) ** 2).mean()


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
# Main Execution Loop
# ----------------------------

def run():
    env = gym.make("Ant-v5")

    logger = MetricLogger(window=50)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    algo = MBMPO(obs_dim, act_dim)
    n_iters = 100

    for it in tqdm(range(n_iters), desc="MB-MPO Meta-Training"):

        # 1. Real data environment interaction tracking
        real_data = []
        obs, _ = env.reset()
        ep_reward = 0.0

        while True:
            act = algo.policy.act(obs)
            next_obs, reward, done, truncated, _ = env.step(act)

            real_data.append((obs, act, reward, next_obs, done or truncated))
            ep_reward += reward
            obs = next_obs

            if done or truncated:
                break

        dyn_loss = algo.train_dynamics(real_data)

        # Anchor initial state using the real environment's latest observation point
        init_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        # -------------------------
        # PHASE A: INNER LOOP (TASK ADAPTATION)
        # -------------------------
        adapted_params_list = []
        for i in range(algo.ensemble_size):
            theta_prime = algo.adapt_policy(model_idx=i, init_obs=init_obs_tensor, horizon=20, batch_size=200)
            adapted_params_list.append(theta_prime)

        # -------------------------
        # PHASE B: OUTER LOOP (META-OPTIMIZATION TRAJECTORIES)
        # -------------------------
        meta_obs_list, meta_act_list, meta_rew_list, meta_done_list = [], [], [], []

        for i in range(algo.ensemble_size):
            m_obs, m_act, m_rew, m_done = algo.rollout_single_model(
                model_idx=i, init_obs=init_obs_tensor, params=adapted_params_list[i], horizon=20, batch_size=200
            )
            meta_obs_list.append(m_obs.view(-1, obs_dim))
            meta_act_list.append(m_act.view(-1, act_dim))
            meta_rew_list.append(m_rew.reshape(-1))
            meta_done_list.append(m_done.reshape(-1))

        post_obs = torch.cat(meta_obs_list, dim=0)
        post_act = torch.cat(meta_act_list, dim=0)
        post_rew = torch.cat(meta_rew_list, dim=0)
        post_done = torch.cat(meta_done_list, dim=0)

        with torch.no_grad():
            post_values = algo.value(post_obs).cpu().numpy()
            val_seq = np.append(post_values, 0.0)

        post_rew_np = post_rew.detach().cpu().numpy()
        post_done_np = post_done.detach().cpu().numpy()

        returns, advs = compute_gae(post_rew_np, val_seq, post_done_np)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_tensor = torch.tensor(advs, dtype=torch.float32, device=device)

        with torch.no_grad():
            meta_old_logp_list = []
            idx_start = 0
            samples_per_model = post_obs.shape[0] // algo.ensemble_size
            for i in range(algo.ensemble_size):
                idx_end = idx_start + samples_per_model
                m_o = post_obs[idx_start:idx_end]
                m_a = post_act[idx_start:idx_end]
                meta_old_logp_list.append(algo.policy.log_prob(m_o, m_a, params=adapted_params_list[i]))
                idx_start = idx_end
            meta_old_logp = torch.cat(meta_old_logp_list, dim=0)

        kl, pol_loss_val = algo.meta_trpo_step(post_obs, post_act, adv_tensor, meta_old_logp, adapted_params_list)
        algo.train_value(post_obs, ret_tensor)
        val_loss = algo.value_loss(post_obs, ret_tensor)

        # Extract the average reward calculated across the grounded simulation loop
        sim_rew_avg = post_rew.mean().item()

        # -------------------------
        # LOGGING
        # -------------------------
        logger.log(
            dyn_loss=dyn_loss,
            policy_loss=pol_loss_val,
            value_loss=val_loss.item(),
            kl=kl,
            reward=sim_rew_avg,  # Now displays stabilized grounded simulator rewards
            entropy=0.0
        )

        tqdm.write(
            f"iter:{it:04d} | "
            f"{logger.line()} | "
            f"ep_rew:{ep_reward:.1f}"
        )
        
        if it == n_iters - 1:
            # Save just the meta-policy weights
            torch.save(algo.policy.state_dict(), "mb_mpo_policy.pt")
            tqdm.write(f"--- Saved policy checkpoint to mb_mpo_policy.pt ---")


if __name__ == "__main__":
    run()