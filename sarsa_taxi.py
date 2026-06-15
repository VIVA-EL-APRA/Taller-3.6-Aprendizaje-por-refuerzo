"""
SARSA (State-Action-Reward-State-Action) — Taxi-v4
Actividad Grupal: Aprendizaje por Refuerzo

Taxi-v4 es el entorno ideal para SARSA tabular:
  - 500 estados discretos (posición taxi × pasajero × destino)
  - 6 acciones (N, S, E, O, recoger, dejar)
  - Recompensa: +20 entrega correcta, -10 acción ilegal, -1 por paso
  - On-Policy: aprende la política que realmente ejecuta (incluyendo errores ε)
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

# ══════════════════════════════════════════════════════════
# HIPERPARÁMETROS
# ══════════════════════════════════════════════════════════
N_EPISODIOS   = 5000
ALPHA         = 0.1      # Tasa de aprendizaje
GAMMA         = 0.99     # Factor de descuento
EPSILON_INI   = 1.0      # Exploración inicial (100%)
EPSILON_MIN   = 0.01     # Exploración mínima (1%)
EPSILON_DECAY = 0.999    # Decaimiento multiplicativo por episodio
SEMILLA       = 42


# ══════════════════════════════════════════════════════════
# 1. AGENTE SARSA TABULAR
# ══════════════════════════════════════════════════════════
class AgenteSARSA:
    """
    SARSA On-Policy: actualiza Q(s,a) usando la acción REAL siguiente a'.
    Ecuación de actualización:
        Q(s,a) ← Q(s,a) + α · [r + γ·Q(s',a') - Q(s,a)]
    donde a' es la acción que el agente REALMENTE tomará en s'
    (puede ser exploración ε o explotación).
    """
    def __init__(self, n_estados, n_acciones, alpha, gamma, eps_ini, eps_min, eps_decay):
        self.n_acciones  = n_acciones
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = eps_ini
        self.eps_min     = eps_min
        self.eps_decay   = eps_decay

        # Q-Table: filas=estados, columnas=acciones, init=0
        self.Q = np.zeros((n_estados, n_acciones))

    def seleccionar_accion(self, estado):
        """Política ε-greedy: explora con prob ε, explota con prob (1-ε)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_acciones)   # Exploración
        return np.argmax(self.Q[estado])                # Explotación

    def actualizar(self, s, a, r, s_prima, a_prima, done):
        """
        Núcleo de SARSA: usa Q(s',a') donde a' YA fue elegida por ε-greedy.
        Diferencia clave vs Q-Learning: Q-Learning usa max_a Q(s',a),
        SARSA usa Q(s', a_real) → más conservador, evita acantilados.
        """
        if done:
            objetivo = r
        else:
            objetivo = r + self.gamma * self.Q[s_prima, a_prima]

        # Error TD (diferencia temporal)
        error_td = objetivo - self.Q[s, a]
        self.Q[s, a] += self.alpha * error_td

    def decaer_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ══════════════════════════════════════════════════════════
# 2. BUCLE DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════
def entrenar_sarsa():
    np.random.seed(SEMILLA)
    env = gym.make("Taxi-v4")
    n_estados  = env.observation_space.n    # 500
    n_acciones = env.action_space.n         # 6

    agente = AgenteSARSA(n_estados, n_acciones,
                         ALPHA, GAMMA, EPSILON_INI, EPSILON_MIN, EPSILON_DECAY)

    historial_rewards  = []
    historial_pasos    = []
    historial_epsilon  = []
    suavizado          = []
    ventana            = deque(maxlen=100)

    print("=" * 58)
    print("  SARSA Tabular — Taxi-v4")
    print("=" * 58)
    print(f"  Estados: {n_estados} | Acciones: {n_acciones}")
    print(f"  α={ALPHA} | γ={GAMMA} | ε_ini={EPSILON_INI} | ε_min={EPSILON_MIN}")
    print("=" * 58)

    for ep in range(1, N_EPISODIOS + 1):
        estado, _ = env.reset(seed=SEMILLA + ep)
        accion    = agente.seleccionar_accion(estado)   # a_0 elegida ANTES del loop
        reward_total = 0
        pasos        = 0
        done         = False

        while not done:
            # Ejecutar acción actual
            sig_estado, recompensa, terminado, truncado, _ = env.step(accion)
            done = terminado or truncado

            # Elegir a' ANTES de actualizar (esto es SARSA, no Q-Learning)
            sig_accion = agente.seleccionar_accion(sig_estado)

            # Actualización SARSA: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            agente.actualizar(estado, accion, recompensa, sig_estado, sig_accion, done)

            estado  = sig_estado
            accion  = sig_accion    # La siguiente acción ya elegida pasa al próximo paso
            reward_total += recompensa
            pasos        += 1

        agente.decaer_epsilon()

        historial_rewards.append(reward_total)
        historial_pasos.append(pasos)
        historial_epsilon.append(agente.epsilon)
        ventana.append(reward_total)
        suavizado.append(np.mean(ventana))

        if ep % 500 == 0:
            print(f"  Ep {ep:5d} | Reward: {reward_total:6.1f} "
                  f"| Prom-100: {np.mean(ventana):6.1f} "
                  f"| ε={agente.epsilon:.4f}")

    env.close()
    print("=" * 58)
    print(f"  Reward final prom-100: {suavizado[-1]:.2f}")
    print(f"  Celdas Q no-cero: {np.count_nonzero(agente.Q)} / {agente.Q.size}")
    print("=" * 58)

    return (historial_rewards, historial_pasos,
            historial_epsilon, suavizado, agente.Q)


# ══════════════════════════════════════════════════════════
# 3. GRÁFICA DE CONVERGENCIA (3 paneles)
# ══════════════════════════════════════════════════════════
def graficar(historial_rewards, historial_pasos,
             historial_epsilon, suavizado,
             guardar_en="sarsa_convergencia.png"):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1A1A2E')
    C = {'raw':'#4FC3F7', 'sm':'#FF6B6B', 'eps':'#FFD700',
         'steps':'#A78BFA', 'thr':'#69FF47', 'panel':'#16213E'}

    eps_range = range(1, len(historial_rewards) + 1)

    # ── Panel 1: Reward acumulado ──────────────────────────
    ax = axes[0]
    ax.set_facecolor(C['panel'])
    ax.plot(eps_range, historial_rewards, color=C['raw'],
            alpha=0.2, lw=0.6, label='Reward por episodio')
    ax.plot(eps_range, suavizado, color=C['sm'],
            lw=2.5, label='Media móvil (100 ep)')
    ax.axhline(7.0, color=C['thr'], ls='--', lw=1.5,
               label='Umbral "resuelto" (≥7)')
    conv = next((i for i, v in enumerate(suavizado) if v >= 7.0), None)
    if conv:
        ax.axvspan(conv, len(historial_rewards), alpha=0.07, color=C['thr'])
        ax.annotate(f'Convergencia\nep. ~{conv}',
                    xy=(conv, 7), xytext=(conv + 200, 3),
                    color=C['thr'], fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=C['thr']))
    ax.set_title('SARSA — Reward Acumulado', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episodio', color='#B0BEC5'); ax.set_ylabel('Reward', color='#B0BEC5')
    ax.tick_params(colors='#B0BEC5'); ax.spines[:].set_color('#2A3A5C')
    ax.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C', labelcolor='white', fontsize=8)
    ax.grid(axis='y', color='#2A3A5C', alpha=0.4)

    # ── Panel 2: Pasos por episodio ────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(C['panel'])
    pasos_suav = np.convolve(historial_pasos,
                              np.ones(100)/100, mode='valid')
    ax2.plot(range(1, len(historial_pasos)+1), historial_pasos,
             color=C['steps'], alpha=0.2, lw=0.6, label='Pasos por episodio')
    ax2.plot(range(50, len(pasos_suav)+50), pasos_suav,
             color='#F472B6', lw=2.5, label='Media móvil (100 ep)')
    ax2.set_title('Pasos por Episodio', color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episodio', color='#B0BEC5'); ax2.set_ylabel('Pasos', color='#B0BEC5')
    ax2.tick_params(colors='#B0BEC5'); ax2.spines[:].set_color('#2A3A5C')
    ax2.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C', labelcolor='white', fontsize=8)
    ax2.grid(axis='y', color='#2A3A5C', alpha=0.4)

    # ── Panel 3: Decaimiento de epsilon ────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(C['panel'])
    ax3.plot(eps_range, historial_epsilon, color=C['eps'], lw=2.0, label='ε (epsilon)')
    ax3.fill_between(eps_range, historial_epsilon, alpha=0.15, color=C['eps'])
    ax3.axhline(EPSILON_MIN, color='#F87171', ls='--', lw=1.2,
                label=f'ε mínimo ({EPSILON_MIN})')
    conv_ep = next((i for i, e in enumerate(historial_epsilon)
                    if e <= EPSILON_MIN + 0.01), None)
    if conv_ep:
        ax3.axvline(conv_ep, color='#F87171', ls=':', lw=1.0, alpha=0.7)
        ax3.annotate(f'ε≈mín\nep.{conv_ep}',
                     xy=(conv_ep, EPSILON_MIN),
                     xytext=(conv_ep + 200, 0.15),
                     color='#F87171', fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='#F87171'))
    ax3.set_title('Decaimiento de Epsilon (ε-greedy)',
                  color='white', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Episodio', color='#B0BEC5')
    ax3.set_ylabel('Valor de ε', color='#B0BEC5')
    ax3.tick_params(colors='#B0BEC5'); ax3.spines[:].set_color('#2A3A5C')
    ax3.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C', labelcolor='white', fontsize=8)
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
    rewards, pasos, epsilons, suav, Q = entrenar_sarsa()
    graficar(rewards, pasos, epsilons, suav)
