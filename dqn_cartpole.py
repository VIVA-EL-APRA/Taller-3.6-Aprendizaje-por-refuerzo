"""
DQN (Deep Q-Network) — CartPole-v1
Actividad Grupal: Aprendizaje por Refuerzo

Innovaciones clave de DQN (DeepMind 2013):
  1. Experience Replay   → rompe correlación temporal
  2. Target Network      → estabiliza el aprendizaje
  3. Red neuronal        → aproxima Q(s,a) para estados continuos
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import random

# ══════════════════════════════════════════════════════════
# HIPERPARÁMETROS
# ══════════════════════════════════════════════════════════
N_EPISODIOS      = 500
GAMMA            = 0.99
LR               = 1e-3
BATCH_SIZE       = 64
REPLAY_SIZE      = 10_000   # Tamaño del Experience Replay buffer
MIN_REPLAY       = 1_000    # Mínimo de experiencias antes de entrenar
EPSILON_INI      = 1.0
EPSILON_MIN      = 0.01
EPSILON_DECAY    = 0.995
TARGET_UPDATE    = 10       # Cada cuántos episodios sincronizar Target Network
SEMILLA          = 42


# ══════════════════════════════════════════════════════════
# 1. RED Q (aproximador de Q(s,a))
# ══════════════════════════════════════════════════════════
class RedQ(nn.Module):
    """
    Recibe el estado s y devuelve Q(s,a) para TODAS las acciones a.
    Permite elegir la mejor acción con un solo forward pass.
    """
    def __init__(self, n_estados, n_acciones, hidden=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_estados, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_acciones)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.red(x)


# ══════════════════════════════════════════════════════════
# 2. EXPERIENCE REPLAY BUFFER
# ══════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Memoria circular de experiencias (s, a, r, s', done).
    Innovación 1 de DQN: rompe la correlación temporal
    muestreando transiciones aleatorias del pasado.
    """
    def __init__(self, capacidad):
        self.buffer = deque(maxlen=capacidad)

    def agregar(self, estado, accion, recompensa, sig_estado, done):
        self.buffer.append((estado, accion, recompensa, sig_estado, done))

    def muestrear(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        estados, acciones, recompensas, sig_estados, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(estados)),
            torch.LongTensor(acciones),
            torch.FloatTensor(recompensas),
            torch.FloatTensor(np.array(sig_estados)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════
# 3. AGENTE DQN
# ══════════════════════════════════════════════════════════
class AgenteDQN:
    """
    DQN = Q-Learning + Red Neuronal + Experience Replay + Target Network.

    Ecuación de Bellman para DQN:
        Q(s,a) ← r + γ · max_a' Q_target(s', a')
    La Target Network (θ⁻) se congela N episodios para estabilizar el objetivo.
    """
    def __init__(self, n_estados, n_acciones):
        self.n_acciones = n_acciones
        self.epsilon    = EPSILON_INI

        # Red principal (se actualiza en cada paso)
        self.red_q      = RedQ(n_estados, n_acciones)
        # Target Network (copia congelada, se sincroniza cada TARGET_UPDATE ep)
        self.red_target = RedQ(n_estados, n_acciones)
        self.red_target.load_state_dict(self.red_q.state_dict())
        self.red_target.eval()

        self.optimizador = optim.Adam(self.red_q.parameters(), lr=LR)
        self.replay      = ReplayBuffer(REPLAY_SIZE)
        self.perdidas    = []

    @torch.no_grad()
    def seleccionar_accion(self, estado):
        """ε-greedy: explora aleatoriamente con prob ε."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_acciones - 1)
        s = torch.FloatTensor(estado).unsqueeze(0)
        return self.red_q(s).argmax(dim=1).item()

    def almacenar(self, s, a, r, ns, done):
        self.replay.agregar(s, a, r, ns, done)

    def aprender(self):
        """Actualización DQN con mini-batch del replay buffer."""
        if len(self.replay) < MIN_REPLAY:
            return

        s, a, r, ns, dones = self.replay.muestrear(BATCH_SIZE)

        # Q valores actuales: Q(s, a_tomada)
        q_actuales = self.red_q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Objetivo: r + γ · max_a' Q_target(s', a')  (Bellman)
        with torch.no_grad():
            q_siguiente = self.red_target(ns).max(dim=1).values
            q_objetivo  = r + GAMMA * q_siguiente * (1 - dones)

        perdida = nn.functional.smooth_l1_loss(q_actuales, q_objetivo)
        self.optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.red_q.parameters(), 1.0)
        self.optimizador.step()
        self.perdidas.append(perdida.item())

    def sincronizar_target(self):
        """Innovación 2: copiar pesos de red_q → red_target."""
        self.red_target.load_state_dict(self.red_q.state_dict())

    def decaer_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ══════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO
# ══════════════════════════════════════════════════════════
def entrenar_dqn():
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    random.seed(SEMILLA)

    env = gym.make("CartPole-v1")
    n_estados  = env.observation_space.shape[0]   # 4
    n_acciones = env.action_space.n               # 2
    agente     = AgenteDQN(n_estados, n_acciones)

    historial_rewards = []
    historial_epsilon = []
    suavizado         = []
    ventana           = deque(maxlen=50)

    print("=" * 58)
    print("  DQN (Deep Q-Network) — CartPole-v1")
    print("=" * 58)
    print(f"  Estados: {n_estados} | Acciones: {n_acciones}")
    print(f"  Replay buffer: {REPLAY_SIZE} | Batch: {BATCH_SIZE}")
    print(f"  Target update cada {TARGET_UPDATE} episodios")
    print("=" * 58)

    for ep in range(1, N_EPISODIOS + 1):
        estado, _ = env.reset(seed=SEMILLA + ep)
        reward_total = 0
        done = False

        while not done:
            accion = agente.seleccionar_accion(estado)
            sig_estado, recompensa, terminado, truncado, _ = env.step(accion)
            done = terminado or truncado

            # Guardar en replay buffer
            agente.almacenar(estado, accion, recompensa, sig_estado, float(done))

            # Aprender del buffer
            agente.aprender()

            estado        = sig_estado
            reward_total += recompensa

        agente.decaer_epsilon()

        # Sincronizar Target Network cada TARGET_UPDATE episodios
        if ep % TARGET_UPDATE == 0:
            agente.sincronizar_target()

        historial_rewards.append(reward_total)
        historial_epsilon.append(agente.epsilon)
        ventana.append(reward_total)
        suavizado.append(np.mean(ventana))

        if ep % 50 == 0:
            print(f"  Ep {ep:4d} | Reward: {reward_total:6.1f} "
                  f"| Prom-50: {np.mean(ventana):6.1f} "
                  f"| ε={agente.epsilon:.3f} "
                  f"| Buffer: {len(agente.replay)}")

    env.close()
    print("=" * 58)
    print(f"  Reward final prom-50: {suavizado[-1]:.1f}")
    print("=" * 58)
    return historial_rewards, historial_epsilon, suavizado, agente.perdidas


# ══════════════════════════════════════════════════════════
# 5. GRÁFICA DE CONVERGENCIA (3 paneles)
# ══════════════════════════════════════════════════════════
def graficar(historial_rewards, historial_epsilon, suavizado, perdidas,
             guardar_en="dqn_convergencia.png"):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1A1A2E')
    C = {'raw':'#4FC3F7', 'sm':'#FF6B6B', 'eps':'#FFD700',
         'loss':'#F472B6', 'thr':'#69FF47', 'panel':'#16213E'}

    eps_range = range(1, len(historial_rewards) + 1)

    # ── Panel 1: Reward acumulado ──────────────────────────
    ax = axes[0]
    ax.set_facecolor(C['panel'])
    ax.plot(eps_range, historial_rewards, color=C['raw'],
            alpha=0.3, lw=0.8, label='Reward por episodio')
    ax.plot(eps_range, suavizado, color=C['sm'],
            lw=2.5, label='Media móvil (50 ep)')
    ax.axhline(195, color=C['thr'], ls='--', lw=1.5,
               label='Umbral resuelto (195)')
    ax.axhline(500, color='#FFD700', ls=':', lw=1.0,
               alpha=0.6, label='Máximo (500)')
    conv = next((i for i, v in enumerate(suavizado) if v >= 195), None)
    if conv:
        ax.axvspan(conv, len(historial_rewards), alpha=0.07, color=C['thr'])
        ax.annotate(f'Convergencia\nep. ~{conv}',
                    xy=(conv, 195), xytext=(conv + 30, 260),
                    color=C['thr'], fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=C['thr']))
    ax.set_title('DQN — Reward Acumulado (CartPole-v1)',
                 color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episodio', color='#B0BEC5')
    ax.set_ylabel('Reward Acumulado', color='#B0BEC5')
    ax.tick_params(colors='#B0BEC5'); ax.spines[:].set_color('#2A3A5C')
    ax.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C',
              labelcolor='white', fontsize=8)
    ax.set_ylim(0, 520)
    ax.grid(axis='y', color='#2A3A5C', alpha=0.4)

    # ── Panel 2: Pérdida de entrenamiento ─────────────────
    ax2 = axes[1]
    ax2.set_facecolor(C['panel'])
    if perdidas:
        suav_loss = np.convolve(perdidas, np.ones(200)/200, mode='valid')
        ax2.plot(range(1, len(perdidas)+1), perdidas,
                 color=C['loss'], alpha=0.15, lw=0.5, label='Pérdida (Huber Loss)')
        ax2.plot(range(100, len(suav_loss)+100), suav_loss,
                 color='#E879F9', lw=2.5, label='Media móvil (200 pasos)')
    ax2.set_title('Pérdida de la Red Q (Huber Loss)',
                  color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Paso de actualización', color='#B0BEC5')
    ax2.set_ylabel('Pérdida', color='#B0BEC5')
    ax2.tick_params(colors='#B0BEC5'); ax2.spines[:].set_color('#2A3A5C')
    ax2.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C',
               labelcolor='white', fontsize=8)
    ax2.grid(axis='y', color='#2A3A5C', alpha=0.4)

    # ── Panel 3: Decaimiento epsilon ───────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(C['panel'])
    ax3.plot(eps_range, historial_epsilon, color=C['eps'],
             lw=2.0, label='ε (epsilon)')
    ax3.fill_between(eps_range, historial_epsilon, alpha=0.15, color=C['eps'])
    ax3.axhline(EPSILON_MIN, color='#F87171', ls='--', lw=1.2,
                label=f'ε mínimo ({EPSILON_MIN})')
    ax3.set_title('Decaimiento de Epsilon (ε-greedy)',
                  color='white', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Episodio', color='#B0BEC5')
    ax3.set_ylabel('Valor de ε', color='#B0BEC5')
    ax3.tick_params(colors='#B0BEC5'); ax3.spines[:].set_color('#2A3A5C')
    ax3.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C',
               labelcolor='white', fontsize=8)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(axis='y', color='#2A3A5C', alpha=0.4)

    plt.tight_layout(pad=2.0)
    plt.savefig(guardar_en, dpi=150, bbox_inches='tight', facecolor='#1A1A2E')
    print(f"  Gráfica guardada en: {guardar_en}")
    plt.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    rewards, epsilons, suav, perdidas = entrenar_dqn()
    graficar(rewards, epsilons, suav, perdidas)
