"""
A2C (Advantage Actor-Critic) — CartPole-v1
Actividad Grupal: Aprendizaje por Refuerzo

Implementación CORRECTA de A2C:
  - N_WORKERS entornos síncronos en paralelo (VectorEnv)
  - Actualización por lotes cada N pasos
  - Sin warnings de broadcasting (MSELoss con tensores del mismo shape)
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

# ══════════════════════════════════════════════════════════
# HIPERPARÁMETROS
# ══════════════════════════════════════════════════════════
N_WORKERS        = 8        # Entornos paralelos (workers síncronos)
PASOS_POR_UPDATE = 5        # Pasos que recolecta cada worker antes de actualizar
GAMMA            = 0.99
LR               = 7e-4
ENTROPY_COEF     = 0.01
VALOR_COEF       = 0.5
GRAD_CLIP        = 0.5
N_UPDATES        = 3000     # Actualizaciones totales (~N_WORKERS * PASOS * N_UPDATES pasos)
SEMILLA          = 42


# ══════════════════════════════════════════════════════════
# 1. RED ACTOR-CRÍTICO
# ══════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    """Cuerpo compartido → cabeza Actor π(a|s)  +  cabeza Crítico V(s)."""

    def __init__(self, n_estados, n_acciones, hidden=64):
        super().__init__()
        self.cuerpo  = nn.Sequential(
            nn.Linear(n_estados, hidden), nn.Tanh(),
            nn.Linear(hidden,    hidden), nn.Tanh(),
        )
        self.actor   = nn.Linear(hidden, n_acciones)
        self.critico = nn.Linear(hidden, 1)

        # Inicialización ortogonal — estabilidad en policy gradient
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,   gain=0.01)
        nn.init.orthogonal_(self.critico.weight, gain=1.0)

    def forward(self, x):
        rep    = self.cuerpo(x)
        logits = self.actor(rep)
        valor  = self.critico(rep).squeeze(-1)  # shape (batch,) — sin broadcasting
        return logits, valor


# ══════════════════════════════════════════════════════════
# 2. RETORNOS DESCONTADOS (n-step desde el último estado)
# ══════════════════════════════════════════════════════════
def calcular_retornos(recompensas, dones, ultimo_valor, gamma=0.99):
    """
    recompensas: lista [T] por worker
    dones:       lista [T] por worker
    ultimo_valor: float — bootstrap V(s_{T+1})
    """
    retornos, G = [], ultimo_valor
    for r, d in zip(reversed(recompensas), reversed(dones)):
        G = r + gamma * G * (1.0 - float(d))
        retornos.insert(0, G)
    return retornos


# ══════════════════════════════════════════════════════════
# 3. AGENTE A2C
# ══════════════════════════════════════════════════════════
class AgenteA2C:
    def __init__(self, n_estados, n_acciones):
        self.red = ActorCritic(n_estados, n_acciones)
        self.opt = optim.RMSprop(self.red.parameters(),
                                 lr=LR, alpha=0.99, eps=1e-5)

    @torch.no_grad()
    def seleccionar_acciones(self, estados):
        """Selecciona acciones para un batch de workers."""
        s      = torch.as_tensor(estados, dtype=torch.float32)
        logits, _ = self.red(s)
        dist   = torch.distributions.Categorical(logits=logits)
        return dist.sample().numpy()

    def aprender(self, estados, acciones, retornos):
        s = torch.as_tensor(np.array(estados),  dtype=torch.float32)
        a = torch.as_tensor(np.array(acciones), dtype=torch.long)
        G = torch.as_tensor(np.array(retornos), dtype=torch.float32)

        logits, valores = self.red(s)   # valores: (batch,)  ← mismo shape que G

        # Ventaja normalizada
        ventajas = G - valores.detach()
        ventajas = (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)

        dist            = torch.distributions.Categorical(logits=logits)
        log_probs       = dist.log_prob(a)
        entropia        = dist.entropy().mean()

        perdida_actor   = -(log_probs * ventajas).mean()
        perdida_critico = nn.functional.mse_loss(valores, G)   # (batch,) vs (batch,)
        perdida_total   = (perdida_actor
                           + VALOR_COEF   * perdida_critico
                           - ENTROPY_COEF * entropia)

        self.opt.zero_grad()
        perdida_total.backward()
        nn.utils.clip_grad_norm_(self.red.parameters(), GRAD_CLIP)
        self.opt.step()

        return perdida_total.item()


# ══════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO CON N WORKERS SÍNCRONOS
# ══════════════════════════════════════════════════════════
def entrenar_a2c():
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)

    # N entornos síncronos — el núcleo del "sync" en A2C
    envs = gym.vector.SyncVectorEnv(
        [lambda i=i: gym.make("CartPole-v1") for i in range(N_WORKERS)]
    )
    n_estados  = envs.single_observation_space.shape[0]  # 4
    n_acciones = envs.single_action_space.n              # 2
    agente     = AgenteA2C(n_estados, n_acciones)

    # Buffers globales de rewards por episodio (para la gráfica)
    historial_ep   = []
    suavizado_ep   = []
    ventana_ep     = deque(maxlen=50)
    rewards_actuales = np.zeros(N_WORKERS)

    estados, _  = envs.reset(seed=SEMILLA)

    print("=" * 60)
    print("  A2C (Advantage Actor-Critic) — CartPole-v1")
    print("=" * 60)
    print(f"  Workers síncronos: {N_WORKERS} | Pasos/update: {PASOS_POR_UPDATE}")
    print(f"  Actualizaciones: {N_UPDATES} | lr={LR} | γ={GAMMA}")
    print("=" * 60)

    for update in range(1, N_UPDATES + 1):

        # ── Recolectar PASOS_POR_UPDATE pasos en todos los workers ──
        buf_s, buf_a, buf_r, buf_d = [], [], [], []

        for _ in range(PASOS_POR_UPDATE):
            acciones = agente.seleccionar_acciones(estados)

            sig_estados, recompensas, terminados, truncados, _ = envs.step(acciones)
            dones = terminados | truncados

            buf_s.append(estados.copy())
            buf_a.append(acciones.copy())
            buf_r.append(recompensas.copy())
            buf_d.append(dones.copy())

            rewards_actuales += recompensas

            # Registrar episodios terminados
            for i, d in enumerate(dones):
                if d:
                    historial_ep.append(rewards_actuales[i])
                    ventana_ep.append(rewards_actuales[i])
                    suavizado_ep.append(np.mean(ventana_ep))
                    rewards_actuales[i] = 0.0

            estados = sig_estados

        # ── Calcular retornos con bootstrap ──
        with torch.no_grad():
            s_t = torch.as_tensor(estados, dtype=torch.float32)
            _, ultimo_vals = agente.red(s_t)
            ultimo_vals = ultimo_vals.numpy()

        # Aplanar: (PASOS, N_WORKERS) → (PASOS*N_WORKERS,)
        all_s, all_a, all_G = [], [], []
        for w in range(N_WORKERS):
            rews_w = [buf_r[t][w] for t in range(PASOS_POR_UPDATE)]
            dons_w = [buf_d[t][w] for t in range(PASOS_POR_UPDATE)]
            rets_w = calcular_retornos(rews_w, dons_w, ultimo_vals[w], GAMMA)
            for t in range(PASOS_POR_UPDATE):
                all_s.append(buf_s[t][w])
                all_a.append(buf_a[t][w])
                all_G.append(rets_w[t])

        agente.aprender(all_s, all_a, all_G)

        if update % 300 == 0 and suavizado_ep:
            ep_total = len(historial_ep)
            prom     = suavizado_ep[-1]
            print(f"  Update {update:5d} | Episodios: {ep_total:5d} "
                  f"| Prom-50ep: {prom:6.1f}")

    envs.close()
    print("=" * 60)
    print(f"  Episodios totales: {len(historial_ep)}")
    print(f"  Reward final prom-50: {suavizado_ep[-1]:.1f}")
    print("=" * 60)
    return historial_ep, suavizado_ep


# ══════════════════════════════════════════════════════════
# 5. GRÁFICA DE CONVERGENCIA
# ══════════════════════════════════════════════════════════
def graficar(historial, suavizado, guardar_en="a2c_convergencia.png"):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1A1A2E')
    C = {'raw':'#4FC3F7','sm':'#FF6B6B','thr':'#69FF47','panel':'#16213E'}

    eps = range(1, len(historial) + 1)

    # ── Curva de aprendizaje ───────────────────────────────
    ax.set_facecolor(C['panel'])
    ax.plot(eps, historial,  color=C['raw'], alpha=0.25, lw=0.7,
            label='Reward por episodio')
    ax.plot(eps, suavizado,  color=C['sm'],  lw=2.5,
            label='Media móvil (50 ep)')
    ax.axhline(195, color=C['thr'],  ls='--', lw=1.5, label='Umbral resuelto (195)')
    ax.axhline(500, color='#FFD700', ls=':',  lw=1.2, alpha=0.7, label='Máximo (500)')

    conv = next((i for i, v in enumerate(suavizado) if v >= 195), None)
    if conv:
        ax.axvspan(conv, len(historial), alpha=0.08, color=C['thr'])
        ax.annotate(f'Convergencia\nep. ~{conv}',
                    xy=(conv, 195), xytext=(min(conv + 80, len(historial) - 50), 260),
                    color=C['thr'], fontsize=9,
                    arrowprops=dict(arrowstyle='->', color=C['thr']))

    ax.set_title('A2C — Curva de Aprendizaje (CartPole-v1)',
                 color='white', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Episodio', color='#B0BEC5', fontsize=11)
    ax.set_ylabel('Reward Acumulado', color='#B0BEC5', fontsize=11)
    ax.tick_params(colors='#B0BEC5')
    ax.spines[:].set_color('#2A3A5C')
    ax.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C',
              labelcolor='white', fontsize=9)
    ax.set_ylim(0, 520)
    ax.grid(axis='y', color='#2A3A5C', alpha=0.4)

    # ── Distribución por fase ──────────────────────────────
    ax2.set_facecolor(C['panel'])
    m = len(historial) // 2
    ax2.hist(historial[:m], bins=30, color='#4A90D9', alpha=0.65,
             label=f'Ep 1-{m} (exploración)')
    ax2.hist(historial[m:], bins=30, color=C['sm'],   alpha=0.65,
             label=f'Ep {m+1}-{len(historial)} (explotación)')
    ax2.axvline(195, color=C['thr'], ls='--', lw=1.5, label='Umbral 195')
    ax2.set_title('Distribución de Rewards por Fase',
                  color='white', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Reward', color='#B0BEC5', fontsize=11)
    ax2.set_ylabel('Frecuencia', color='#B0BEC5', fontsize=11)
    ax2.tick_params(colors='#B0BEC5')
    ax2.spines[:].set_color('#2A3A5C')
    ax2.legend(facecolor='#0F0F2A', edgecolor='#2A3A5C',
               labelcolor='white', fontsize=9)
    ax2.grid(axis='y', color='#2A3A5C', alpha=0.4)

    plt.tight_layout(pad=2.0)
    plt.savefig(guardar_en, dpi=150, bbox_inches='tight', facecolor='#1A1A2E')
    print(f"  Gráfica guardada en: {guardar_en}")
    plt.close()


# ══════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    rewards, suavizado = entrenar_a2c()
    graficar(rewards, suavizado)

# ══════════════════════════════════════════════════════════
# NOTA: Para convergencia completa (prom > 195) aumentar
# N_UPDATES a 8000. Este ejemplo muestra la curva de
# aprendizaje progresivo real del algoritmo.
# ══════════════════════════════════════════════════════════
