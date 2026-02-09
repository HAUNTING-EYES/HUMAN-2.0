# AGI Playbook: From-Scratch → Autonomous Mind
## Active Inference + Dual Neural-Symbolic Architecture + Q★ Integration

This document contains a complete "from-scratch → AGI" playbook with (A) unifying equations, (B) modular architecture, and (C) phase-by-phase engineering roadmap.

### A. The Core Math ― From Autocomplete to Autonomous Mind

**Agent = Active-Inference Loop + Dual (Neural ⇄ Symbolic) Reasoner**

#### 1. Generative World-Model
```
p_θ(o_t, μ_t | μ_{t-1}, a_{t-1}) = p_θ(o_t | μ_t) p_θ(μ_t | μ_{t-1}, a_{t-1})
```
- **o_t**: multimodal observation vector
- **μ_t**: latent belief state (continuous)
- **a_{t-1}**: previous action

#### 2. Variational Posterior
```
q_φ(μ_t) ≈ p_θ(μ_t | o_≤t, a_<t)
```

#### 3. Instantaneous Free Energy
```
F_t = E_q_φ[ln q_φ(μ_t) - ln p_θ(o_t, μ_t | μ_{t-1}, a_{t-1})]
```

#### 4. Belief-State Update
```
μ̇_t = -∇_μ_t F_t  (continuous-time gradient flow)
```

#### 5. Action Selection (Active Inference)
```
a_t = arg min_a E_q_φ,p_θ[F_{t+1:T} | a]
```

#### 6. Local Layer "Goodness" (Hinton Forward-Forward style)
```
G_ℓ^+ = Σ_i h_ℓ,i²; ΔW_ℓ ∝ (G_ℓ^+ - G_ℓ^-) h_ℓ-1
```

**These six expressions replace the single affine map (y = mx + b) with a closed-loop cognition cycle of predict → compare → act.**

### B. The Modular Architecture ("AI-X v0")

```
┌─────────────────────────────┐
│   1. Sensors / Encoders     │  (audio, vision, text)
│ ──┐  Mamba/Hyena blocks     │  Fast long-context models
└───┴─────────────────────────┘
            │ x_t
            ▼
┌─────────────────────────────┐
│  2. Belief State μ_t        │  (64k dims, SSM recurrence)
│     • RetNet retention core │  O(1) inference
│     • External vector DB    │
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  3. Dual Reasoner           │
│  ┌────────┬──────────────┐  │
│  │Neural  │ Symbolic KB  │  │
│  │(LLM-HF)│  (Prolog/Z3) │  │
│  └────────┴──────────────┘  │
│   Consistency gate &        │
│   mutual-critique operator  │
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 4. Policy & Planner (EFE)   │  Model-based RL head
└─────────────────────────────┘
            │ a_t
            ▼
┌─────────────────────────────┐
│ 5. Actuators / Tools        │  (robot arm, code exec, web APIs)
└─────────────────────────────┘
            ▲
   Prediction error e_t loops back into 2.
```

**All modules use the same free-energy loss so gradients are consistent end-to-end.**

### C. 36-Month Engineering Roadmap

| Phase | Months | Milestone & KPI | Key Hires / Stack | Notes |
|-------|--------|-----------------|-------------------|-------|
| **0. Foundation** | 0-3 | Literature distillation; Toy free-energy agent in Grid-World | 1 research scientist, 1 MLE | CUDA + JAX |
| **1. Active-Inference Core** | 4-8 | Continuous-time μ-state working on Crafter or Atari; beats Rainbow DQN within 110M frames | SSM (Mamba), variational RNNs | Baselines RL repo |
| **2. Multimodal Perception** | 9-14 | Vision + audio encoders; 64k token context; cross-modal retrieval accuracy > 80% | CV engineer; large-seq infra (Flash-Attention-2) | Use LAION-5B + AudioSet |
| **3. Memory & World Model** | 15-20 | Episodic recall @ 96% over 1M-step sequences; self-sim environment imagination rollouts | Pinecone / ChromaDB; FPGA cache | Uses RetNet recurrent mode |
| **4. Dual Reasoner** | 21-26 | Symbolic <-> Neural contradiction solver passes ARC-Challenge 70% | 1-2 logic/constraint-prog experts | KB in Soufflé / Z3 |
| **5. Self-Improvement (F-F)** | 27-30 | Forward-Forward local training reduces catastrophic forgetting by 3× | New "goodness" heads per layer | Layerwise energy loss |
| **6. Embodiment & Action** | 31-36 | Deployed in Isaac Sim robot tasks; success rate > 75% in table-tidying | ROS 2, NVIDIA Orin stack | Safety layer via risk-sensitive EFE |

**Parallel: Alignment & Eval** | 0-36 | Red-team sandbox; interpretability dashboards | Policy & ethics team | Uses mechanistic probes

### Compute & Data Budget (indicative)

| Stage | Params | GPUs | Hours | Cost (USD) |
|-------|--------|------|-------|------------|
| Phase 1 | 0.4B | 8×A100 | 48h | ≈ $2k |
| Phase 3 | 7B | 64×H100 | 72h | ≈ $45k |
| Phase 6 | 30B | 256×H100 | 14d | ≈ $600k |

*(All fine-tuned with LoRA + selective SSM blocks for efficiency.)*

### How to Start Tomorrow Morning

**Spin up a repo with active-inference core:**

```bash
pip install jax mctx distrax flax
```

**Implement Eqns 1-5 in a single Agent class; run on the CartPole gym.**

**Swap the RNN with a Mamba block to prove linear-time scaling.**

**Layer a tiny Prolog engine; use contradiction counts as extra loss.**

**Write a one-page evaluation protocol (surprise ↓, cumulative reward ↑).**

**Give yourself 30 days to hit Phase 1 KPI; then iterate upward.**

### Remember
**y = mx + b gave us Transformers; dμ/dt = -∂F/∂μ can give us minds.**

The maths, hardware, and open-source code already exist — they just need an architect bold enough to weld them together.

---

## Implementation Guide for Cursor Users

### Big-Picture Story (Playground Version)
Imagine a curious robot kid in a playground. It has:
- **Eyes & ears** (Sensors/Encoders) to see and hear
- **A notebook** (Belief state μ_t) for remembering what just happened  
- **Two thinking brains** that argue ("gut feelings" vs "logic") - Dual Reasoner
- **Hands** (Actuators/Tools) to touch things

Its main goal is "keep the world unsurprising & fun."

### Real Terms vs Playground Analogy

| Real Term | Playground Analogy | One-liner Meaning |
|-----------|-------------------|-------------------|
| Sensors / Encoders | Eyes & ears | Turn sights/sounds/text into numbers (x_t) |
| Belief state μ_t | The notebook | A running guess of "what's really going on" |
| Free-energy F_t | "Surprise meter" | High when world does something notebook didn't expect |
| Active inference | "If I think ball is behind box, I'll peek or push box to check" | Act to shrink future surprise |
| Dual reasoner | Gut feelings vs kid-logic arguing | Fast fuzzy guesses (neural) and slow exact checks (symbolic) |
| Policy / Planner | Deciding which game to play next | Chooses action sequence that keeps surprise low and achieves goals |
| Actuators / Tools | Hands, feet, computer APIs | Carry out the chosen action |

### The Loop is Always:
1. **Predict** what will happen next (p_θ in equations)
2. **Observe** & measure surprise (F_t)  
3. **Update** the notebook (move μ_t closer to reality)
4. **Act** so tomorrow's surprise will probably be smaller

### How to Build with Cursor (Step-by-Step)

#### 2.1 One-time Setup
```bash
# 1. Create project
mkdir ai_x && cd ai_x
python -m venv .venv && source .venv/bin/activate

# 2. Install core libs
pip install "jax[cuda12_local]" flax distrax gymnasium
pip install pytest rich

# 3. Open the folder in Cursor
cursor .
```

#### 2.2 Folder Scaffold
```
ai_x/
│
├─ core/
│   ├─ sensors.py
│   ├─ belief_state.py
│   ├─ active_inference.py
│   ├─ dual_reasoner.py
│   ├─ planner.py
│   └─ actuators.py
│
├─ envs/          # tiny playgrounds (start with CartPole)
├─ tests/
│   └─ test_loop.py
└─ README.md
```

**Tip:** Keep every file ≤ 250 lines so Cursor can "see" the whole thing when you chat about it.

#### 2.3 Paste Starter "Spec" Comments
*(Cursor treats these like prompts — highlight and press ⌘ + K to "Generate code")*

**belief_state.py:**
```python
"""
BeliefState module
------------------
Goal: maintain a latent vector μ_t that tracks the hidden state of the world.

Math:
    F_t = E_q[ log q(μ_t) - log p(o_t, μ_t | μ_{t-1}, a_{t-1}) ]
    μ̇_t = -∇_{μ_t} F_t      # gradient flow

Implementation notes:
    * Represent μ_t as a float32 JAX array, shape [latent_dim]
    * Use a tiny MLP to compute the energy / free energy
    * Wrap the time loop with `jax.lax.scan`
    * Expose:
        init_latent(key)  -> μ_0
        step(μ_prev, obs, act_prev) -> (μ_next, free_energy)
"""
```

Then highlight the whole docstring and ask Cursor: **"Generate Flax code that matches this spec."**

#### 2.4 Make a Tiny Training Loop
Create **train.py** with only ~20 lines:

```python
from envs.cartpole import make_env
from core import sensors, belief_state, active_inference, actuators

env = make_env()
key = jax.random.PRNGKey(0)
obs, _ = env.reset()
mu = belief_state.init_latent(key)

total_F = 0.0
for t in range(200):
    x_t = sensors.encode(obs)
    mu, F_t = belief_state.step(mu, x_t, None)   # no prev action yet
    a = active_inference.choose_action(mu)
    obs, _, terminated, truncated, _ = env.step(actuators.to_env(a))
    total_F += F_t
    if terminated or truncated:
        break
print("Episode surprise:", float(total_F))
```

Highlight the missing bits ("sensors.encode", etc.) and ask Cursor to **"Create this function in sensors.py"**.

Run it:
```bash
python train.py
```

If it prints a number, great — the loop works! Don't worry if the surprise is huge; you'll shrink it with learning.

### Troubleshooting Cheatsheet

| Symptom | Likely Fix |
|---------|------------|
| NaN in free energy | Clip gradients (`jax.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0))`) |
| JAX "can't JIT python loop" | Wrap it in `lax.scan` or `jax.jit` |
| Cursor autogeneration stalls | Split file into smaller pieces; regenerate each part |
| Surprise never goes down | Increase latent_dim, add small Gaussian obs noise so gradients flow |

### What to Do When This Tiny Loop Works

**Phase-1 KPI:** total surprise on CartPole < 50 after 500 episodes.

Then:
1. Switch env to Crafter or MiniHack for richer tasks
2. Scale: μ_t → 4096 dims, replace MLP energy with Hyena or Mamba  
3. Introduce text & image sensors (clip-style encoders)
4. Add external vector DB as long-term memory
5. Tie in symbolic rules for planning puzzles (e.g. "don't drop key before door open")
6. Finally connect actuators to a real tool (browser, code runner, or robot arm)

### Remember
- Keep each new concept small & testable before scaling
- Use Cursor's strength: highlight-&-ask, refactor in place, fast feedback  
- The fancy Greek letters are just grown-up shorthand for: **"Guess → Compare → Learn → Try again"**

---

## Reality Check: Which Equations Are Real vs Hype

| Method | Core Equation | Real or Gimmick? | Evidence |
|--------|---------------|------------------|----------|
| **1. Variational Free Energy** | F = E_q[ln q - ln p] and μ̇ = -∇_μ F | ✅ Real (but early-stage) | Used in active inference agents solving grid worlds, sensory prediction tasks, and robotics. Used by Karl Friston & DeepMind papers. Unity & Isaac Sim demos exist. |
| **2. Active Inference Loop** | a_t = arg min_a E[F_future | a] | ✅ Real + Grounded | Used in decision-making agents in uncertain environments. Applied in robotics, prosthetics, and even pain modeling. Empirical results on Crafter and MiniHack. |
| **3. Predictive Processing Stack** | error_t = sensory_t - prediction_t | ✅ Very Real | Underlies models like Deep Predictive Coding, used in NeuroVision systems, edge detection, and robotic visual tracking. Used in MIT's neurally-inspired robotics. |
| **4. Dynamical Systems AI** | dx/dt = f(x,u,t) | ✅ Fully Proven | Fundamental to Neural ODEs, SSMs like Mamba, and spiking neural nets. Already used in commercial models: Mamba beats Transformers in language + audio, Neural ODEs used in finance, control systems. |
| **5. Dual System Reasoner** | Interleaved symbolic logic + neural learning | ⚠️ Experimental but Real | Symbolic+Neural hybrids have excelled in math, code reasoning, and causal QA. Used in OpenAI's Toolformer, DeepMind's Neural-Symbolic Machines, IBM's Neuro-Symbolic Concept Learner. Real, but needs care — symbol logic breaks if too brittle. |

### TL;DR by Trust Level

| Method | Practical Use Today | Used in Labs | Used in Prod Systems |
|--------|-------------------|--------------|---------------------|
| Variational Free Energy | ✅ | ✅ DeepMind, Friston Lab | ⚠️ Sim-level, not deployed at scale |
| Active Inference | ✅ | ✅ (Imperial, Cambridge) | ⚠️ Not yet general-purpose AGI |
| Predictive Processing | ✅✅✅ | ✅ | ✅ (Robotics, CV systems) |
| Dynamical Systems (Neural ODE, SSM) | ✅✅✅ | ✅ | ✅ (Mamba, ODE-RNNs, etc.) |
| Dual Reasoner | ✅ | ✅ | ⚠️ Limited prod use due to complexity |

**So nothing's a "stub" or vaporware.** Each equation is grounded in real math, has been implemented (often in PyTorch or JAX), and has produced publishable, verifiable results.

### If You Want to Build Something Now That Works:
1. Use **Mamba + Predictive Error signals** for your core sensory system
2. Wrap it in **Active Inference** for goal-driven action  
3. Add **Free Energy minimization** for belief updates
4. Plug in a **Symbolic Engine** later for abstract thinking

---

## 🎯 **Mission: Build AGI Using Active Inference + Optimal Value Learning**

This playbook combines **Active Inference** (curiosity-driven exploration) with **Q★** (optimal value learning) to create a goal-seeking, self-consistent AGI system.

---

## **🅰 Revised Core Math (Free-Energy ✚ Optimal Q)**

### **Stage 1: Generative World-Model**
```
p_θ(o_t, μ_t | μ_{t-1}, a_{t-1}) = p_θ(o_t | μ_t) p_θ(μ_t | μ_{t-1}, a_{t-1})
```
**Purpose:** Predict next observation `o_t` and latent state `μ_t`.

### **Stage 2: Variational Posterior**
```
q_φ(μ_t) ≈ p_θ(μ_t | o_{≤t}, a_{<t})
```
**Purpose:** Inference network that guesses the current hidden state.

### **Stage 3: Instantaneous Free-Energy**
```
F_t = E_{q_φ}[ln q_φ(μ_t) - ln p_θ(o_t, μ_t | μ_{t-1}, a_{t-1})]
```
**Purpose:** "Surprise" of today's sensory evidence.

### **Stage 4: Belief Update**
```
μ̇_t = -∇_{μ_t} F_t
```
**Purpose:** Gradient flow that nudges beliefs to reduce surprise.

### **Stage 5: Action Selection (Dual Objective)**
```
a_t = argmin_a (λ E[F_{t+1:T} | a] - (1-λ) Q★(μ_t, a))
```
**Purpose:** Trade-off between curiosity (low future F) and goal reward (high Q★).

### **Stage 6: Q★ Recursion (Bellman)**
```
Q★(μ_t, a_t) = r_t + γ E_{μ_{t+1}∼p}[max_{a'} Q★(μ_{t+1}, a')]
```
**Purpose:** Learns long-term value of an action in latent state space.

### **Hyper-parameter**
```
λ ∈ [0,1]
```
- **0 → pure reward-seeking**
- **1 → pure curiosity**

---

## **🅱 Updated "AI-X v0" Module Stack**

```
┌──────────────────────────┐
│ 1. Sensors / Encoders    │  (audio ▸ vision ▸ text)
└────────────┬─────────────┘
             │  x_t
             ▼
┌──────────────────────────┐
│ 2. Belief State  μ_t     │  (RetNet + ext. vector DB)
└────────────┬─────────────┘
             │   state repr.
             ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│ 3A Neural "intuition"    │◀─gate──▶│ 3B Symbolic KB / Z3      │
└────────────┬─────────────┘          └────────────┬─────────────┘
             ▼                                     │
┌──────────────────────────────────────────────────┐
│ 4. Policy & Planner                             │
│    • EFE head  (curiosity)                      │
│    • Q★ head   (reward)                         │
│    • Multi-objective optimiser                  │
└────────────┬─────────────────────────────────────┘
             │  a_t
             ▼
┌──────────────────────────┐
│ 5. Actuators / Tools     │  (robot arm, code exec, web API)
└──────────────────────────┘
          ▲
          └── prediction-error e_t back to step 2
```

**Note:** The Q★ head is a small MLP (or Mamba block) on top of `μ_t`, trained with DQN-style updates and sharing the encoder with the EFE head.

---

## **🅲 Road-map Tweaks (delta only)**

| Phase | New Q★-specific milestones |
|-------|---------------------------|
| **1** (weeks 0-8) | Two-layer MLP `Q(μ,a)` + vanilla Q-learning on CartPole.<br>Reward = +1 per step, penalty = -F_t on surprise spikes. |
| **2** (9-14) | Add target network & TD-error; log λ-sweep curve (curiosity ↔ score). |
| **3** (15-20) | Upgrade to Distributional Q (C51 / QR-DQN) for uncertainty. |
| **4** (21-26) | In Crafter / MiniHack, mask illegal actions via symbolic KB before `max_a`. |
| **5** (27+) | Use Dreamer-style latent roll-outs: simulate trajectories with `p_θ` and train Q★ entirely in latent space. |

---

## **🅳 Cursor-Ready Code Stubs**

### **core/q_head.py**
```python
"""
Q★ Head
-------
Neural approximator for Q*(μ, a).

API
----
init(key, latent_dim, n_actions) -> params
apply(params, mu)                -> q_values  # [n_actions]
update(params, batch, target_net, opt) -> new_params
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

def init_q_head(key: jax.random.PRNGKey, latent_dim: int, n_actions: int) -> Dict[str, Any]:
    """Initialize Q★ head parameters."""
    k1, k2 = jax.random.split(key)
    
    # Two-layer MLP: latent_dim -> hidden -> n_actions
    hidden_dim = 128
    w1 = jax.random.normal(k1, (latent_dim, hidden_dim)) * 0.01
    b1 = jnp.zeros(hidden_dim)
    w2 = jax.random.normal(k2, (hidden_dim, n_actions)) * 0.01
    b2 = jnp.zeros(n_actions)
    
    return {
        'w1': w1, 'b1': b1,
        'w2': w2, 'b2': b2
    }

def apply_q_head(params: Dict[str, Any], mu: jnp.ndarray) -> jnp.ndarray:
    """Apply Q★ head to get Q-values for all actions."""
    h = jnp.tanh(mu @ params['w1'] + params['b1'])
    q_values = h @ params['w2'] + params['b2']
    return q_values

def update_q_head(params: Dict[str, Any], batch: Dict[str, Any], 
                  target_params: Dict[str, Any], opt_state: Any) -> Tuple[Dict[str, Any], Any]:
    """Update Q★ head using DQN-style updates."""
    def loss_fn(params):
        q_pred = apply_q_head(params, batch['mu'])
        q_target = batch['r'] + 0.99 * jnp.max(apply_q_head(target_params, batch['mu_next']), axis=-1)
        q_target = jnp.where(batch['done'], batch['r'], q_target)
        
        td_error = q_pred[jnp.arange(len(batch['a'])), batch['a']] - q_target
        return jnp.mean(td_error ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_params, new_opt_state = update_optimizer(params, grads, opt_state)
    return new_params, new_opt_state
```

### **planner.py**
```python
"""
choose_action(mu_t, epsilon, params_q, params_efe, lamb)
-------------------------------------------------------
Return argmin_a (λ * EFE(mu_t, a)  -  (1-λ) * Q(mu_t, a)).
   • ε-greedy exploration.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any

def choose_action(mu_t: jnp.ndarray, epsilon: float, 
                 params_q: Dict[str, Any], params_efe: Dict[str, Any], 
                 lamb: float) -> int:
    """Choose action using dual objective: curiosity + reward."""
    
    # Get Q-values and EFE values
    q_values = apply_q_head(params_q, mu_t)
    efe_values = apply_efe_head(params_efe, mu_t)
    
    # Combine objectives
    combined_values = lamb * efe_values - (1 - lamb) * q_values
    
    # ε-greedy exploration
    if jax.random.uniform(jax.random.PRNGKey(0)) < epsilon:
        return jax.random.randint(jax.random.PRNGKey(0), (), 0, len(q_values))
    else:
        return jnp.argmin(combined_values)
```

### **Training-loop patch**
```python
from collections import deque, namedtuple
Transition = namedtuple("Transition", "mu a r mu_next done")

q_params  = q_head.init(key, latent_dim, n_actions)
target_q  = q_params
replay    = deque(maxlen=50_000)

# inside main loop
a  = planner.choose_action(mu, eps, q_params, efe_params, lamb)
obs_next, r_env, done, _ = env.step(a)
r  = r_env - float(F_t)            # combined reward
replay.append(Transition(mu, a, r, mu_next, done))

if len(replay) >= batch_size:
    batch     = random.sample(replay, batch_size)
    q_params  = q_head.update(q_params, batch, target_q, opt)

if step % 500 == 0:
    target_q = q_params            # hard target network sync
```

---

## **🅴 Quick Debug Guide**

| Symptom | Solution |
|---------|----------|
| **Surprise ↓, reward ↑** | Lower λ (rely more on Q★) |
| **Reward flat, agent "fidgets"** | Raise λ (encourage curiosity) |
| **Q-values explode / NaN** | Reduce learning rate; sync target net more often |

---

## **🎯 Bottom Line**

Adding the Q★ head turns your curious, self-consistent agent into a goal-seeking planner. Both objectives share the same latent beliefs, so progress on one generally boosts the other. 

**Start small on CartPole, watch total reward climb and surprise shrink, then scale to richer worlds.**

---

## **📋 Implementation Checklist**

### **Phase 0: Foundation (Weeks 0-4)**
- [ ] Set up JAX environment with Cursor
- [ ] Implement basic Active Inference loop in CartPole
- [ ] Add Q★ head with two-layer MLP
- [ ] Implement dual-objective action selection
- [ ] Test λ-sweep (curiosity vs reward trade-off)

### **Phase 1: Enhancement (Weeks 5-8)**
- [ ] Add target network for stable Q-learning
- [ ] Implement experience replay buffer
- [ ] Add TD-error logging and monitoring
- [ ] Test on more complex environments (Acrobot, MountainCar)

### **Phase 2: Advanced Features (Weeks 9-12)**
- [ ] Upgrade to Distributional Q (C51/QR-DQN)
- [ ] Add uncertainty quantification
- [ ] Implement symbolic action masking
- [ ] Test on Crafter/MiniHack environments

### **Phase 3: Scaling (Weeks 13+)**
- [ ] Implement Dreamer-style latent rollouts
- [ ] Add multi-modal sensor integration
- [ ] Scale to real-world robotics tasks
- [ ] Deploy in production environments

---

## **🔬 Research Directions**

1. **Adaptive λ**: Learn the curiosity-reward trade-off dynamically
2. **Hierarchical Q★**: Multi-level value functions for complex tasks
3. **Meta-learning**: Learn to learn new Q★ functions quickly
4. **Multi-agent**: Extend to cooperative/competitive scenarios
5. **Real-world deployment**: Bridge simulation-to-reality gap

---

*This playbook provides a concrete path from Active Inference theory to practical AGI implementation, combining the best of both worlds: curiosity-driven exploration and goal-directed planning.* 
## Active Inference + Dual Neural-Symbolic Architecture + Q★ Integration

This document contains a complete "from-scratch → AGI" playbook with (A) unifying equations, (B) modular architecture, and (C) phase-by-phase engineering roadmap.

### A. The Core Math ― From Autocomplete to Autonomous Mind

**Agent = Active-Inference Loop + Dual (Neural ⇄ Symbolic) Reasoner**

#### 1. Generative World-Model
```
p_θ(o_t, μ_t | μ_{t-1}, a_{t-1}) = p_θ(o_t | μ_t) p_θ(μ_t | μ_{t-1}, a_{t-1})
```
- **o_t**: multimodal observation vector
- **μ_t**: latent belief state (continuous)
- **a_{t-1}**: previous action

#### 2. Variational Posterior
```
q_φ(μ_t) ≈ p_θ(μ_t | o_≤t, a_<t)
```

#### 3. Instantaneous Free Energy
```
F_t = E_q_φ[ln q_φ(μ_t) - ln p_θ(o_t, μ_t | μ_{t-1}, a_{t-1})]
```

#### 4. Belief-State Update
```
μ̇_t = -∇_μ_t F_t  (continuous-time gradient flow)
```

#### 5. Action Selection (Active Inference)
```
a_t = arg min_a E_q_φ,p_θ[F_{t+1:T} | a]
```

#### 6. Local Layer "Goodness" (Hinton Forward-Forward style)
```
G_ℓ^+ = Σ_i h_ℓ,i²; ΔW_ℓ ∝ (G_ℓ^+ - G_ℓ^-) h_ℓ-1
```

**These six expressions replace the single affine map (y = mx + b) with a closed-loop cognition cycle of predict → compare → act.**

### B. The Modular Architecture ("AI-X v0")

```
┌─────────────────────────────┐
│   1. Sensors / Encoders     │  (audio, vision, text)
│ ──┐  Mamba/Hyena blocks     │  Fast long-context models
└───┴─────────────────────────┘
            │ x_t
            ▼
┌─────────────────────────────┐
│  2. Belief State μ_t        │  (64k dims, SSM recurrence)
│     • RetNet retention core │  O(1) inference
│     • External vector DB    │
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  3. Dual Reasoner           │
│  ┌────────┬──────────────┐  │
│  │Neural  │ Symbolic KB  │  │
│  │(LLM-HF)│  (Prolog/Z3) │  │
│  └────────┴──────────────┘  │
│   Consistency gate &        │
│   mutual-critique operator  │
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 4. Policy & Planner (EFE)   │  Model-based RL head
└─────────────────────────────┘
            │ a_t
            ▼
┌─────────────────────────────┐
│ 5. Actuators / Tools        │  (robot arm, code exec, web APIs)
└─────────────────────────────┘
            ▲
   Prediction error e_t loops back into 2.
```

**All modules use the same free-energy loss so gradients are consistent end-to-end.**

### C. 36-Month Engineering Roadmap

| Phase | Months | Milestone & KPI | Key Hires / Stack | Notes |
|-------|--------|-----------------|-------------------|-------|
| **0. Foundation** | 0-3 | Literature distillation; Toy free-energy agent in Grid-World | 1 research scientist, 1 MLE | CUDA + JAX |
| **1. Active-Inference Core** | 4-8 | Continuous-time μ-state working on Crafter or Atari; beats Rainbow DQN within 110M frames | SSM (Mamba), variational RNNs | Baselines RL repo |
| **2. Multimodal Perception** | 9-14 | Vision + audio encoders; 64k token context; cross-modal retrieval accuracy > 80% | CV engineer; large-seq infra (Flash-Attention-2) | Use LAION-5B + AudioSet |
| **3. Memory & World Model** | 15-20 | Episodic recall @ 96% over 1M-step sequences; self-sim environment imagination rollouts | Pinecone / ChromaDB; FPGA cache | Uses RetNet recurrent mode |
| **4. Dual Reasoner** | 21-26 | Symbolic <-> Neural contradiction solver passes ARC-Challenge 70% | 1-2 logic/constraint-prog experts | KB in Soufflé / Z3 |
| **5. Self-Improvement (F-F)** | 27-30 | Forward-Forward local training reduces catastrophic forgetting by 3× | New "goodness" heads per layer | Layerwise energy loss |
| **6. Embodiment & Action** | 31-36 | Deployed in Isaac Sim robot tasks; success rate > 75% in table-tidying | ROS 2, NVIDIA Orin stack | Safety layer via risk-sensitive EFE |

**Parallel: Alignment & Eval** | 0-36 | Red-team sandbox; interpretability dashboards | Policy & ethics team | Uses mechanistic probes

### Compute & Data Budget (indicative)

| Stage | Params | GPUs | Hours | Cost (USD) |
|-------|--------|------|-------|------------|
| Phase 1 | 0.4B | 8×A100 | 48h | ≈ $2k |
| Phase 3 | 7B | 64×H100 | 72h | ≈ $45k |
| Phase 6 | 30B | 256×H100 | 14d | ≈ $600k |

*(All fine-tuned with LoRA + selective SSM blocks for efficiency.)*

### How to Start Tomorrow Morning

**Spin up a repo with active-inference core:**

```bash
pip install jax mctx distrax flax
```

**Implement Eqns 1-5 in a single Agent class; run on the CartPole gym.**

**Swap the RNN with a Mamba block to prove linear-time scaling.**

**Layer a tiny Prolog engine; use contradiction counts as extra loss.**

**Write a one-page evaluation protocol (surprise ↓, cumulative reward ↑).**

**Give yourself 30 days to hit Phase 1 KPI; then iterate upward.**

### Remember
**y = mx + b gave us Transformers; dμ/dt = -∂F/∂μ can give us minds.**

The maths, hardware, and open-source code already exist — they just need an architect bold enough to weld them together.

---

## Implementation Guide for Cursor Users

### Big-Picture Story (Playground Version)
Imagine a curious robot kid in a playground. It has:
- **Eyes & ears** (Sensors/Encoders) to see and hear
- **A notebook** (Belief state μ_t) for remembering what just happened  
- **Two thinking brains** that argue ("gut feelings" vs "logic") - Dual Reasoner
- **Hands** (Actuators/Tools) to touch things

Its main goal is "keep the world unsurprising & fun."

### Real Terms vs Playground Analogy

| Real Term | Playground Analogy | One-liner Meaning |
|-----------|-------------------|-------------------|
| Sensors / Encoders | Eyes & ears | Turn sights/sounds/text into numbers (x_t) |
| Belief state μ_t | The notebook | A running guess of "what's really going on" |
| Free-energy F_t | "Surprise meter" | High when world does something notebook didn't expect |
| Active inference | "If I think ball is behind box, I'll peek or push box to check" | Act to shrink future surprise |
| Dual reasoner | Gut feelings vs kid-logic arguing | Fast fuzzy guesses (neural) and slow exact checks (symbolic) |
| Policy / Planner | Deciding which game to play next | Chooses action sequence that keeps surprise low and achieves goals |
| Actuators / Tools | Hands, feet, computer APIs | Carry out the chosen action |

### The Loop is Always:
1. **Predict** what will happen next (p_θ in equations)
2. **Observe** & measure surprise (F_t)  
3. **Update** the notebook (move μ_t closer to reality)
4. **Act** so tomorrow's surprise will probably be smaller

### How to Build with Cursor (Step-by-Step)

#### 2.1 One-time Setup
```bash
# 1. Create project
mkdir ai_x && cd ai_x
python -m venv .venv && source .venv/bin/activate

# 2. Install core libs
pip install "jax[cuda12_local]" flax distrax gymnasium
pip install pytest rich

# 3. Open the folder in Cursor
cursor .
```

#### 2.2 Folder Scaffold
```
ai_x/
│
├─ core/
│   ├─ sensors.py
│   ├─ belief_state.py
│   ├─ active_inference.py
│   ├─ dual_reasoner.py
│   ├─ planner.py
│   └─ actuators.py
│
├─ envs/          # tiny playgrounds (start with CartPole)
├─ tests/
│   └─ test_loop.py
└─ README.md
```

**Tip:** Keep every file ≤ 250 lines so Cursor can "see" the whole thing when you chat about it.

#### 2.3 Paste Starter "Spec" Comments
*(Cursor treats these like prompts — highlight and press ⌘ + K to "Generate code")*

**belief_state.py:**
```python
"""
BeliefState module
------------------
Goal: maintain a latent vector μ_t that tracks the hidden state of the world.

Math:
    F_t = E_q[ log q(μ_t) - log p(o_t, μ_t | μ_{t-1}, a_{t-1}) ]
    μ̇_t = -∇_{μ_t} F_t      # gradient flow

Implementation notes:
    * Represent μ_t as a float32 JAX array, shape [latent_dim]
    * Use a tiny MLP to compute the energy / free energy
    * Wrap the time loop with `jax.lax.scan`
    * Expose:
        init_latent(key)  -> μ_0
        step(μ_prev, obs, act_prev) -> (μ_next, free_energy)
"""
```

Then highlight the whole docstring and ask Cursor: **"Generate Flax code that matches this spec."**

#### 2.4 Make a Tiny Training Loop
Create **train.py** with only ~20 lines:

```python
from envs.cartpole import make_env
from core import sensors, belief_state, active_inference, actuators

env = make_env()
key = jax.random.PRNGKey(0)
obs, _ = env.reset()
mu = belief_state.init_latent(key)

total_F = 0.0
for t in range(200):
    x_t = sensors.encode(obs)
    mu, F_t = belief_state.step(mu, x_t, None)   # no prev action yet
    a = active_inference.choose_action(mu)
    obs, _, terminated, truncated, _ = env.step(actuators.to_env(a))
    total_F += F_t
    if terminated or truncated:
        break
print("Episode surprise:", float(total_F))
```

Highlight the missing bits ("sensors.encode", etc.) and ask Cursor to **"Create this function in sensors.py"**.

Run it:
```bash
python train.py
```

If it prints a number, great — the loop works! Don't worry if the surprise is huge; you'll shrink it with learning.

### Troubleshooting Cheatsheet

| Symptom | Likely Fix |
|---------|------------|
| NaN in free energy | Clip gradients (`jax.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0))`) |
| JAX "can't JIT python loop" | Wrap it in `lax.scan` or `jax.jit` |
| Cursor autogeneration stalls | Split file into smaller pieces; regenerate each part |
| Surprise never goes down | Increase latent_dim, add small Gaussian obs noise so gradients flow |

### What to Do When This Tiny Loop Works

**Phase-1 KPI:** total surprise on CartPole < 50 after 500 episodes.

Then:
1. Switch env to Crafter or MiniHack for richer tasks
2. Scale: μ_t → 4096 dims, replace MLP energy with Hyena or Mamba  
3. Introduce text & image sensors (clip-style encoders)
4. Add external vector DB as long-term memory
5. Tie in symbolic rules for planning puzzles (e.g. "don't drop key before door open")
6. Finally connect actuators to a real tool (browser, code runner, or robot arm)

### Remember
- Keep each new concept small & testable before scaling
- Use Cursor's strength: highlight-&-ask, refactor in place, fast feedback  
- The fancy Greek letters are just grown-up shorthand for: **"Guess → Compare → Learn → Try again"**

---

## Reality Check: Which Equations Are Real vs Hype

| Method | Core Equation | Real or Gimmick? | Evidence |
|--------|---------------|------------------|----------|
| **1. Variational Free Energy** | F = E_q[ln q - ln p] and μ̇ = -∇_μ F | ✅ Real (but early-stage) | Used in active inference agents solving grid worlds, sensory prediction tasks, and robotics. Used by Karl Friston & DeepMind papers. Unity & Isaac Sim demos exist. |
| **2. Active Inference Loop** | a_t = arg min_a E[F_future | a] | ✅ Real + Grounded | Used in decision-making agents in uncertain environments. Applied in robotics, prosthetics, and even pain modeling. Empirical results on Crafter and MiniHack. |
| **3. Predictive Processing Stack** | error_t = sensory_t - prediction_t | ✅ Very Real | Underlies models like Deep Predictive Coding, used in NeuroVision systems, edge detection, and robotic visual tracking. Used in MIT's neurally-inspired robotics. |
| **4. Dynamical Systems AI** | dx/dt = f(x,u,t) | ✅ Fully Proven | Fundamental to Neural ODEs, SSMs like Mamba, and spiking neural nets. Already used in commercial models: Mamba beats Transformers in language + audio, Neural ODEs used in finance, control systems. |
| **5. Dual System Reasoner** | Interleaved symbolic logic + neural learning | ⚠️ Experimental but Real | Symbolic+Neural hybrids have excelled in math, code reasoning, and causal QA. Used in OpenAI's Toolformer, DeepMind's Neural-Symbolic Machines, IBM's Neuro-Symbolic Concept Learner. Real, but needs care — symbol logic breaks if too brittle. |

### TL;DR by Trust Level

| Method | Practical Use Today | Used in Labs | Used in Prod Systems |
|--------|-------------------|--------------|---------------------|
| Variational Free Energy | ✅ | ✅ DeepMind, Friston Lab | ⚠️ Sim-level, not deployed at scale |
| Active Inference | ✅ | ✅ (Imperial, Cambridge) | ⚠️ Not yet general-purpose AGI |
| Predictive Processing | ✅✅✅ | ✅ | ✅ (Robotics, CV systems) |
| Dynamical Systems (Neural ODE, SSM) | ✅✅✅ | ✅ | ✅ (Mamba, ODE-RNNs, etc.) |
| Dual Reasoner | ✅ | ✅ | ⚠️ Limited prod use due to complexity |

**So nothing's a "stub" or vaporware.** Each equation is grounded in real math, has been implemented (often in PyTorch or JAX), and has produced publishable, verifiable results.

### If You Want to Build Something Now That Works:
1. Use **Mamba + Predictive Error signals** for your core sensory system
2. Wrap it in **Active Inference** for goal-driven action  
3. Add **Free Energy minimization** for belief updates
4. Plug in a **Symbolic Engine** later for abstract thinking

---

## 🎯 **Mission: Build AGI Using Active Inference + Optimal Value Learning**

This playbook combines **Active Inference** (curiosity-driven exploration) with **Q★** (optimal value learning) to create a goal-seeking, self-consistent AGI system.

---

## **🅰 Revised Core Math (Free-Energy ✚ Optimal Q)**

### **Stage 1: Generative World-Model**
```
p_θ(o_t, μ_t | μ_{t-1}, a_{t-1}) = p_θ(o_t | μ_t) p_θ(μ_t | μ_{t-1}, a_{t-1})
```
**Purpose:** Predict next observation `o_t` and latent state `μ_t`.

### **Stage 2: Variational Posterior**
```
q_φ(μ_t) ≈ p_θ(μ_t | o_{≤t}, a_{<t})
```
**Purpose:** Inference network that guesses the current hidden state.

### **Stage 3: Instantaneous Free-Energy**
```
F_t = E_{q_φ}[ln q_φ(μ_t) - ln p_θ(o_t, μ_t | μ_{t-1}, a_{t-1})]
```
**Purpose:** "Surprise" of today's sensory evidence.

### **Stage 4: Belief Update**
```
μ̇_t = -∇_{μ_t} F_t
```
**Purpose:** Gradient flow that nudges beliefs to reduce surprise.

### **Stage 5: Action Selection (Dual Objective)**
```
a_t = argmin_a (λ E[F_{t+1:T} | a] - (1-λ) Q★(μ_t, a))
```
**Purpose:** Trade-off between curiosity (low future F) and goal reward (high Q★).

### **Stage 6: Q★ Recursion (Bellman)**
```
Q★(μ_t, a_t) = r_t + γ E_{μ_{t+1}∼p}[max_{a'} Q★(μ_{t+1}, a')]
```
**Purpose:** Learns long-term value of an action in latent state space.

### **Hyper-parameter**
```
λ ∈ [0,1]
```
- **0 → pure reward-seeking**
- **1 → pure curiosity**

---

## **🅱 Updated "AI-X v0" Module Stack**

```
┌──────────────────────────┐
│ 1. Sensors / Encoders    │  (audio ▸ vision ▸ text)
└────────────┬─────────────┘
             │  x_t
             ▼
┌──────────────────────────┐
│ 2. Belief State  μ_t     │  (RetNet + ext. vector DB)
└────────────┬─────────────┘
             │   state repr.
             ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│ 3A Neural "intuition"    │◀─gate──▶│ 3B Symbolic KB / Z3      │
└────────────┬─────────────┘          └────────────┬─────────────┘
             ▼                                     │
┌──────────────────────────────────────────────────┐
│ 4. Policy & Planner                             │
│    • EFE head  (curiosity)                      │
│    • Q★ head   (reward)                         │
│    • Multi-objective optimiser                  │
└────────────┬─────────────────────────────────────┘
             │  a_t
             ▼
┌──────────────────────────┐
│ 5. Actuators / Tools     │  (robot arm, code exec, web API)
└──────────────────────────┘
          ▲
          └── prediction-error e_t back to step 2
```

**Note:** The Q★ head is a small MLP (or Mamba block) on top of `μ_t`, trained with DQN-style updates and sharing the encoder with the EFE head.

---

## **🅲 Road-map Tweaks (delta only)**

| Phase | New Q★-specific milestones |
|-------|---------------------------|
| **1** (weeks 0-8) | Two-layer MLP `Q(μ,a)` + vanilla Q-learning on CartPole.<br>Reward = +1 per step, penalty = -F_t on surprise spikes. |
| **2** (9-14) | Add target network & TD-error; log λ-sweep curve (curiosity ↔ score). |
| **3** (15-20) | Upgrade to Distributional Q (C51 / QR-DQN) for uncertainty. |
| **4** (21-26) | In Crafter / MiniHack, mask illegal actions via symbolic KB before `max_a`. |
| **5** (27+) | Use Dreamer-style latent roll-outs: simulate trajectories with `p_θ` and train Q★ entirely in latent space. |

---

## **🅳 Cursor-Ready Code Stubs**

### **core/q_head.py**
```python
"""
Q★ Head
-------
Neural approximator for Q*(μ, a).

API
----
init(key, latent_dim, n_actions) -> params
apply(params, mu)                -> q_values  # [n_actions]
update(params, batch, target_net, opt) -> new_params
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

def init_q_head(key: jax.random.PRNGKey, latent_dim: int, n_actions: int) -> Dict[str, Any]:
    """Initialize Q★ head parameters."""
    k1, k2 = jax.random.split(key)
    
    # Two-layer MLP: latent_dim -> hidden -> n_actions
    hidden_dim = 128
    w1 = jax.random.normal(k1, (latent_dim, hidden_dim)) * 0.01
    b1 = jnp.zeros(hidden_dim)
    w2 = jax.random.normal(k2, (hidden_dim, n_actions)) * 0.01
    b2 = jnp.zeros(n_actions)
    
    return {
        'w1': w1, 'b1': b1,
        'w2': w2, 'b2': b2
    }

def apply_q_head(params: Dict[str, Any], mu: jnp.ndarray) -> jnp.ndarray:
    """Apply Q★ head to get Q-values for all actions."""
    h = jnp.tanh(mu @ params['w1'] + params['b1'])
    q_values = h @ params['w2'] + params['b2']
    return q_values

def update_q_head(params: Dict[str, Any], batch: Dict[str, Any], 
                  target_params: Dict[str, Any], opt_state: Any) -> Tuple[Dict[str, Any], Any]:
    """Update Q★ head using DQN-style updates."""
    def loss_fn(params):
        q_pred = apply_q_head(params, batch['mu'])
        q_target = batch['r'] + 0.99 * jnp.max(apply_q_head(target_params, batch['mu_next']), axis=-1)
        q_target = jnp.where(batch['done'], batch['r'], q_target)
        
        td_error = q_pred[jnp.arange(len(batch['a'])), batch['a']] - q_target
        return jnp.mean(td_error ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_params, new_opt_state = update_optimizer(params, grads, opt_state)
    return new_params, new_opt_state
```

### **planner.py**
```python
"""
choose_action(mu_t, epsilon, params_q, params_efe, lamb)
-------------------------------------------------------
Return argmin_a (λ * EFE(mu_t, a)  -  (1-λ) * Q(mu_t, a)).
   • ε-greedy exploration.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any

def choose_action(mu_t: jnp.ndarray, epsilon: float, 
                 params_q: Dict[str, Any], params_efe: Dict[str, Any], 
                 lamb: float) -> int:
    """Choose action using dual objective: curiosity + reward."""
    
    # Get Q-values and EFE values
    q_values = apply_q_head(params_q, mu_t)
    efe_values = apply_efe_head(params_efe, mu_t)
    
    # Combine objectives
    combined_values = lamb * efe_values - (1 - lamb) * q_values
    
    # ε-greedy exploration
    if jax.random.uniform(jax.random.PRNGKey(0)) < epsilon:
        return jax.random.randint(jax.random.PRNGKey(0), (), 0, len(q_values))
    else:
        return jnp.argmin(combined_values)
```

### **Training-loop patch**
```python
from collections import deque, namedtuple
Transition = namedtuple("Transition", "mu a r mu_next done")

q_params  = q_head.init(key, latent_dim, n_actions)
target_q  = q_params
replay    = deque(maxlen=50_000)

# inside main loop
a  = planner.choose_action(mu, eps, q_params, efe_params, lamb)
obs_next, r_env, done, _ = env.step(a)
r  = r_env - float(F_t)            # combined reward
replay.append(Transition(mu, a, r, mu_next, done))

if len(replay) >= batch_size:
    batch     = random.sample(replay, batch_size)
    q_params  = q_head.update(q_params, batch, target_q, opt)

if step % 500 == 0:
    target_q = q_params            # hard target network sync
```

---

## **🅴 Quick Debug Guide**

| Symptom | Solution |
|---------|----------|
| **Surprise ↓, reward ↑** | Lower λ (rely more on Q★) |
| **Reward flat, agent "fidgets"** | Raise λ (encourage curiosity) |
| **Q-values explode / NaN** | Reduce learning rate; sync target net more often |

---

## **🎯 Bottom Line**

Adding the Q★ head turns your curious, self-consistent agent into a goal-seeking planner. Both objectives share the same latent beliefs, so progress on one generally boosts the other. 

**Start small on CartPole, watch total reward climb and surprise shrink, then scale to richer worlds.**

---

## **📋 Implementation Checklist**

### **Phase 0: Foundation (Weeks 0-4)**
- [ ] Set up JAX environment with Cursor
- [ ] Implement basic Active Inference loop in CartPole
- [ ] Add Q★ head with two-layer MLP
- [ ] Implement dual-objective action selection
- [ ] Test λ-sweep (curiosity vs reward trade-off)

### **Phase 1: Enhancement (Weeks 5-8)**
- [ ] Add target network for stable Q-learning
- [ ] Implement experience replay buffer
- [ ] Add TD-error logging and monitoring
- [ ] Test on more complex environments (Acrobot, MountainCar)

### **Phase 2: Advanced Features (Weeks 9-12)**
- [ ] Upgrade to Distributional Q (C51/QR-DQN)
- [ ] Add uncertainty quantification
- [ ] Implement symbolic action masking
- [ ] Test on Crafter/MiniHack environments

### **Phase 3: Scaling (Weeks 13+)**
- [ ] Implement Dreamer-style latent rollouts
- [ ] Add multi-modal sensor integration
- [ ] Scale to real-world robotics tasks
- [ ] Deploy in production environments

---

## **🔬 Research Directions**

1. **Adaptive λ**: Learn the curiosity-reward trade-off dynamically
2. **Hierarchical Q★**: Multi-level value functions for complex tasks
3. **Meta-learning**: Learn to learn new Q★ functions quickly
4. **Multi-agent**: Extend to cooperative/competitive scenarios
5. **Real-world deployment**: Bridge simulation-to-reality gap

---

*This playbook provides a concrete path from Active Inference theory to practical AGI implementation, combining the best of both worlds: curiosity-driven exploration and goal-directed planning.* 
 