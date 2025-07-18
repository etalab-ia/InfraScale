import math
from typing import List

# On utilise une méthode d'uniformisation pour calculer la distribution 
# de probabilité de la queue qui est normalement une chaîne de Markov continue.
# DTMC = Discrete Time Markov Chain
# M/M/1 = Markovian arrival process / Markovian service process / 1 server
# Pour une chaîne M/M/c, on peut multiplier mu par c et avoir des résultats cohérents

# TODO: ajouter une méthode pour calculer l'état de la queue quand elle est stationnaire

def _mm1_uniformised_dist(lam: float, mu: float, t: float,
                          n_states: int = 200,
                          eps: float = 1e-12) -> List[float]:
    """
    Probability vector p_k(t)   (k = 0..n_states)  for an M/M/1 queue
    initially empty, using uniformisation.
    """
    γ = lam + mu                    # uniformisation rate (≥ max row sum of A)
    p = lam / γ                     # birth prob. in the DTMC
    q = mu  / γ                     # death prob.
    
    # v is the state-probability vector after k DTMC steps (start in state 0)
    v = [1.0] + [0.0]*n_states
    probs = [0.0]*(n_states+1)

    # Poisson weight for k = 0
    tail = 1.0
    weight = math.exp(-γ * t)
    tail -= weight
    probs[:] = [weight * x for x in v]

    k = 0
    while tail > eps: # we stop when the remaining weight is less than eps
        k += 1
        weight *= (γ * t) / k       # w_k = e^{-γt}(γt)^k / k!
        tail -= weight

        # one DTMC step:  v ← v P   (reflecting barrier at 0, truncated at n_states)
        new_v = [0.0]*(n_states+1)
        new_v[0] = v[0]*(1-p) + v[1]*q
        for i in range(1, n_states):
            new_v[i] = v[i-1]*p + v[i]*(1-p-q) + v[i+1]*q
        new_v[n_states] += v[n_states-1]*p + v[n_states]*(1-p-q)  # births above cut-off

        v = new_v
        probs = [pi + weight*vi for pi, vi in zip(probs, v)]

    return probs


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
