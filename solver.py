import numpy as np
from math import floor
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from constraints import calculate_mem_constraint, calculate_speed_constraint, calculate_capacity_constraint, calculate_wait_constraint, calculate_speed_constants


class InfrascaleSolver:
    """Solveur du problème d'optimisation d'Infrascale avec [l'implémentation scipy de l'algorithme de Nelder-Mead](https://docs.scipy.org/doc/scipy/tutorial/optimize.html).
       L'algorithme est théoriquement sensible aux conditions initiales mais du fait de la forme de la fonction objectif
       (produit de fonctions monotones croissantes sur R+) on s'attend à ce qu'il converge systématiquement vers
       la solution optimale.

       Paramètres:
       - logger: logger
       - **kwargs: paramètres du problème
    """

    def __init__(self, logger, **kwargs):
        self.params = kwargs
        self.logger = logger

    def _objective_function(self, x):
        S, N, B, Q = x
        return S * N

    def _constraints_function(self, x, **kwargs):
        S, N, B, Q = x

        kwargs['batch_size'] = 2**B

        return calculate_mem_constraint(S, **kwargs), \
                calculate_speed_constraint(S, N, B, Q, **kwargs), \
                calculate_capacity_constraint(N, B, Q, **kwargs), \
                calculate_wait_constraint(Q, B, **kwargs)

    def solve(self):
        """Résolution du problème d'optimisation d'Infrascale. Voir [l'implémentation scipy de l'algorithme de Nelder-Mead](https://docs.scipy.org/doc/scipy/tutorial/optimize.html).
        
        Args:
            **params: Paramètres du problème

        Returns:
            dict: Résultats de la résolution
        """
        params = self.params
        constraints = NonlinearConstraint(lambda x: self._constraints_function(x, **params), lb=-np.inf, ub=[0, 0, 0, 0])
        bounds = Bounds([1, 1, 1, 0.1], [np.inf, np.inf, 9, np.inf]) # On pose max(B) = 9 car 2^9 = 512 et plus serait irréaliste

        intermediate_results = minimize(self._objective_function, x0=[1, 1, 1, 1], bounds=bounds, constraints=constraints)
        S, N, B, Q = intermediate_results.x

        # On veut un nombre entier pour B donc une fois qu'on a une solution, on borne B à l'entier inférieur
        bounds = Bounds([1, 1, 1, 0.1], [np.inf, np.inf, floor(B), np.inf])
        return minimize(self._objective_function, x0=[1, 1, 1, 1], bounds=bounds, constraints=constraints)
    
    
    def get_metrics(self, S, N):
        throughput_per_gpu = calculate_speed_constants(7, **self.params)[0]
        throughput_per_node = throughput_per_gpu * (1 + self.params['efficiency_factor'] * (S - 1))
        throughput_total = throughput_per_node * N
        throughput_per_user = throughput_total / self.params['users']
        tpot = 1 / throughput_per_node

        return {
            'throughput_per_gpu': throughput_per_gpu,
            'throughput_per_node': throughput_per_node,
            'throughput_total': throughput_total,
            'throughput_per_user': throughput_per_user,
            'tpot': tpot
        }

