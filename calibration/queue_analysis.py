import math
from decimal import Decimal, getcontext
from typing import List

# On utilise une méthode d'uniformisation pour calculer la distribution 
# de probabilité de la queue qui est normalement une chaîne de Markov continue.
# DTMC = Discrete Time Markov Chain
# M/M/1 = Markovian arrival process / Markovian service process / 1 server
# Pour une chaîne M/M/c, on peut multiplier mu par c et avoir des résultats cohérents

# TODO: ajouter une méthode pour calculer l'état de la queue quand elle est stationnaire

getcontext().prec = 56

def _mm1_uniformised_dist(lam: float, mu: float, t: float,
                          n_states: int = 200,
                          eps: float = 1e-12) -> List[float]:
    """
    Probability vector p_k(t)   (k = 0..n_states)  for an M/M/1 queue
    initially empty, using uniformisation.
    """
    γ = Decimal(lam) + Decimal(mu)                    # uniformisation rate (≥ max row sum of A)
    p = Decimal(lam) / γ                     # birth prob. in the DTMC
    q = Decimal(mu)  / γ                     # death prob.
    
    # v is the state-probability vector after k DTMC steps (start in state 0)
    v = [Decimal(1.0)] + [Decimal(0.0)]*n_states
    probs = [Decimal(0.0)]*(n_states+1)

    # Poisson weight for k = 0
    tail = Decimal(1.0)
    weight = Decimal.exp(-γ * Decimal(t))
    tail -= weight
    probs[:] = [weight * x for x in v]

    k = 0
    while tail > eps: # we stop when the remaining weight is less than eps
        k += 1
        weight *= (γ * Decimal(t)) / Decimal(k)       # w_k = e^{-γt}(γt)^k / k!
        # print(k, weight)

        if weight == Decimal(0.0):           # hard under‑flow ‑> no more mass
            break
        tail -= weight

        # one DTMC step:  v ← v P   (reflecting barrier at 0, truncated at n_states)
        new_v = [Decimal(0.0)]*(n_states+1)
        new_v[0] = v[0]*(1-p) + v[1]*q
        for i in range(1, n_states):
            new_v[i] = v[i-1]*p + v[i]*(1-p-q) + v[i+1]*q
        new_v[n_states] += v[n_states-1]*p + v[n_states]*(1 - q)

        v = new_v
        probs = [pi + weight*vi for pi, vi in zip(probs, v)]

    return [float(x) for x in probs]

# import math
# from typing import List

# def _mm1_uniformised_dist(lam: float, mu: float, t: float,
#                           n_states: int = 200,
#                           eps: float = 1e-12) -> List[float]:
#     """
#     Transient distribution P{N(t)=k}, k ≤ n_states, for an M/M/1 queue
#     started empty, using uniformisation with *log‑space Poisson weights*.
#     """
#     γ      = lam + mu
#     p, q   = lam/γ, mu/γ
#     γt     = γ * t
#     log_γt = math.log(γt)
#     log_eps = math.log(eps)

#     v      = [Decimal(1.0)] + [Decimal(0.0)]*n_states          # state after k DTMC steps
#     probs  = [Decimal(0.0)]*(n_states + 1)            # accumulator P{N(t)=k}

#     k      = 0
#     log_w  = -γt                             # log w₀ = -γt

#     # iterate until the current term is below eps *and* we have passed the mode
#     mode = int(γt)                           # Poisson mode ≈ γt
#     while True:
#         if log_w > log_eps:                  # only add numerically relevant terms
#             w = Decimal.exp(log_w)
#             probs = [pi + w*vi for pi, vi in zip(probs, v)]

#         # stop once both conditions hold
#         if k > mode and log_w <= log_eps:
#             break

#         # one DTMC step (reflecting at 0, lump overflow into n_states)
#         new_v = [0.0]*(n_states + 1)
#         new_v[0] = v[0]*(1-p) + v[1]*q
#         for i in range(1, n_states):
#             new_v[i] = v[i-1]*p + v[i]*(1-p-q) + v[i+1]*q
#         new_v[n_states] = v[n_states-1]*p + v[n_states]*(1 - q)
#         v = new_v

#         k += 1
#         log_w += log_γt - math.log(k)        # log w_k from log w_{k-1}
#         print(k, log_w)

#     # tiny drift → renormalise
#     s = sum(probs)
#     return [pi/s for pi in probs] if s else probs



def mm1_transient_cdf(lam: float, mu: float, t: float,
                      n: int, n_states: int = 200,
                      eps: float = 1e-12) -> float:
    """
    CDF  F_n(t) = P{ queue length ≤ n at time t } for an M/M/1 queue,
    initially empty.
    """
    dist = _mm1_uniformised_dist(lam, mu, t, n_states, eps)
    if n >= len(dist):
        raise ValueError("n exceeds the truncation level n_states.")
    return sum(dist[:n+1])


def mm1_transient_mean(lam: float, mu: float, t: float,
                       n_states: int = 500,
                       eps: float = 1e-12) -> float:
    """
    Mean queue length E[N(t)] at time t for an M/M/1 queue,
    initially empty.
    """
    dist = _mm1_uniformised_dist(lam, mu, t, n_states, eps)
    return sum(k * prob for k, prob in enumerate(dist))


def mm1_transient_percentile(lam: float, mu: float, t: float,
                             percentile: float = 0.95,
                             n_states: int = 500,
                             eps: float = 1e-12) -> int:
    """
    Percentile of queue length at time t for an M/M/1 queue,
    initially empty. Returns the smallest k such that P(N(t) ≤ k) ≥ percentile.
    """
    dist = _mm1_uniformised_dist(lam, mu, t, n_states, eps)
    cumulative = 0.0
    for k, prob in enumerate(dist):
        cumulative += prob
        if cumulative >= percentile:
            return k
    return len(dist) - 1  # fallback if percentile not reached

# --------- example ----------
if __name__ == "__main__":
    U = 100
    mu_base = 22 # 19.5
    λ, μ, t = U*0.5/16, mu_base*10/math.sqrt(U), 1
    n  = 4
    # print(f"F_{n}({t}) =", mm1_transient_cdf(λ, μ, t, n))
    print(f"mean =", mm1_transient_mean(λ, μ, t)*2.5+3)
    print(f"95th =", mm1_transient_percentile(λ, μ, t, 0.95)*2.5+3)
