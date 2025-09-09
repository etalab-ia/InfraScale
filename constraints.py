"""
L'ensemble des fonctions qui permettent calculer les 4 contraintes (mémoire, débit, capacité, attente)
de l'actuel problème d'optimisation d'Infrascale.

Chaque contrainte est une fonction qui prend en entrée les paramètres du problème et retourne une valeur
négative si la contrainte est satisfaite et positive sinon.
"""

from math import log10

#########################
# CONTRAINTE DE MEMOIRE #
#########################

# Voir https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm
# et https://www.jinghong-chen.net/estimate-vram-usage-in-llm-inference/
# pour les formules

def calculate_kv_cache_memory(**kwargs):
  model_layers = kwargs['model_layers']
  model_dim = kwargs['model_dim']
  batch_size = kwargs['batch_size']
  tokens_per_request = kwargs['tokens_per_request']
  bytes_per_param = kwargs['bytes_per_param']

  bytes_to_gb = 2**30

  kv_cache_bytes = (
      2 * model_layers * model_dim * batch_size *
      tokens_per_request * bytes_per_param
  )
  return kv_cache_bytes / bytes_to_gb

def calculate_memory_constant(**kwargs):
  """Calcul de la constante de la contrainte de mémoire.

  Args:
      **kwargs: Paramètres du problème

  Returns:
      float: Mémoire totale utilisée pour le modèle et l'inférence (en GB)
  """
  model_b_params = kwargs['model_b_params']
  model_layers = kwargs['model_layers']
  model_dim = kwargs['model_dim']
  gpu_vram_gb = kwargs['gpu_vram_gb']
  batch_size = kwargs['batch_size']
  tokens_per_request = kwargs['tokens_per_request']
  bytes_per_param = kwargs['bytes_per_param']
  memory_overhead_percent = kwargs['memory_overhead_percent']

  bytes_to_gb = 2**30

  model_mem_gb = model_b_params * bytes_per_param
  activations_mem_gb = (batch_size * tokens_per_request * model_dim * bytes_per_param) / bytes_to_gb
  kv_cache_mem_gb = calculate_kv_cache_memory(**kwargs)

  total_mem_gb = (model_mem_gb + activations_mem_gb + kv_cache_mem_gb) * (1 + memory_overhead_percent / 100)

  return total_mem_gb

def calculate_mem_constraint(S, **kwargs):
  """Calcul de la contrainte de mémoire.

  Args:
      S (int): Taille de node (nombre de GPUs)
      **kwargs: Paramètres du problème

  Returns:
      float: Contrainte de mémoire
  """
  E = kwargs['efficiency_factor']
  K = calculate_memory_constant(**kwargs)
  constraint = K - (1 + E * (S - 1)) * kwargs['gpu_vram_gb']

  return constraint

#######################
# CONTRAINTE DE DEBIT #
#######################

# Voir https://www.jinghong-chen.net/estimate-vram-usage-in-llm-inference/ pour les formules

def calculate_speed_constants(B, **kwargs):
  """Calcul des constantes de la contrainte de débit. Ces chiffres sont à côté de la plaque
  (voir calibration.ipynb) mais on les corrige avec calculate_scaling_with_B_and_Q.
  La correction n'est fiable que pour des modèles d'une taille entre 8b et 27b.
  
  Args:
      **kwargs: Paramètres du problème

  Returns:
      tuple: (tpot_one_gpu, time_prefill, time_decoding)
  """
  users = kwargs['users']
  batch_size = 2**B
  prompt_size = kwargs['prompt_size']
  tokens_per_request = kwargs['tokens_per_request']
  bytes_per_param = kwargs['bytes_per_param']
  model_b_params = kwargs['model_b_params']
  model_dim = kwargs['model_dim']
  model_layers = kwargs['model_layers']
  gpu_flops = kwargs['gpu_flops']
  gpu_bandwidth = kwargs['gpu_bandwidth']
  
  gpu_flops = gpu_flops * 10**12 * 0.55
  gpu_bandwidth = gpu_bandwidth * 10**12

  flops_prefill = 2 * batch_size * model_layers * prompt_size * model_dim * (2*model_dim + prompt_size)
  mm_prefill = bytes_per_param * (model_b_params * 1e9 + 2 * model_layers * batch_size * prompt_size * model_dim + batch_size * prompt_size * model_dim)
  time_prefill = max(flops_prefill / gpu_flops, mm_prefill / gpu_bandwidth)

  flops_decode_one_token = batch_size * model_b_params * 2 * 1e9
  mm_params = model_b_params*10**9
  mm_kv_read = 2 * batch_size * prompt_size * model_dim
  mm_kv_write = 2 * batch_size * model_dim
  mm_decode_one_token = 2 * (mm_params + mm_kv_read + mm_kv_write)
  tpot = max(flops_decode_one_token / gpu_flops, mm_decode_one_token / gpu_bandwidth)

  # recalibration qui marche pour les modèles de 8b à 27b, voir calibration.ipynb
  k = 24/model_b_params
  recalibration_factor = k**(-1/(k+1))
  calibrated_tpot = tpot / recalibration_factor

  time_decoding = calibrated_tpot * (tokens_per_request - prompt_size)

  return (time_prefill + time_decoding) / (tokens_per_request - prompt_size), time_prefill, time_decoding

def calculate_scaling_with_S(S, **kwargs):
  """Calcul du scaling de débit avec S. Attention, la relation avec S n'est satisfaisante que pour des valeurs
  de S pas trop grandes (< 8) et sur des interconnexions très rapides (type NVLink). 
  
  Args:
      S (int): Taille de node (nombre de GPUs)
      **kwargs: Paramètres du problème
  """
  E = kwargs['efficiency_factor']
  return 1 + (S - 1) * E

def calculate_scaling_with_B_and_Q(B, Q, **kwargs):
  """Calcul du scaling de débit avec la charge de GPU (facteur de B et Q). 
  Fiable uniquement pour des modèles d'une taille entre 8b et 27b.
  
  Args:
      B (int): Taille de batch
      Q (int): Taille de queue (nombre de requêtes parallèles / batch size)
      **kwargs: Paramètres du problème
  """
  return log10(min(Q, 1) * 2**B + 30)

def calculate_speed_constraint(S, N, B, Q, **kwargs):
  """Calcul de la contrainte de débit. Les calculs sont faits en termes de latence
  inter-token plutôt qu'en terme de throughput, mais le résultat est le même. La contrainte
  n'est actuellement fiable que pour des valeurs de S assez faibles (< 8), des interconnexions
  à très haute bande passante (type NVLink) et pour des modèles d'une taille entre 8b et 27b.
  
  Args:
      S (int): Taille de node (nombre de GPUs)
      N (int): Nombre de nodes
      B (int): Taille de batch
      Q (int): Taille de queue (nombre de requêtes parallèles / batch size)
      **kwargs: Paramètres du problème

  Returns:
      float: Contrainte de débit
  """
  E = kwargs['efficiency_factor']
  target_speed = kwargs['target_speed']
  users = kwargs['users']
  tpot_one_gpu = calculate_speed_constants(B, **kwargs)[0]
  constraint =  (Q * tpot_one_gpu * calculate_scaling_with_B_and_Q(B, Q, **kwargs)) / calculate_scaling_with_S(S, **kwargs) - (1 / target_speed)
  
  return constraint

##########################
# CONTRAINTE DE CAPACITE #
##########################

def calculate_capacity_constraint(N, B, Q, **kwargs):
  """Calcul de la contrainte de capacité : on doit pouvoir servir autant de requêtes que d'utilisateurs
  (sur le temps de traitement de Q batchs - hypothèse implicite).
  
  Args:
      N (int): Nombre de nodes
      B (int): Taille de batch
      Q (int): Taille de queue (nombre de requêtes en attente)
      **kwargs: Paramètres du problème

  Returns:
      float: Contrainte de capacité
  """
  return kwargs['users'] - N * Q * 2**B

########################
# CONTRAINTE D'ATTENTE #
########################

def calculate_wait_constraint(Q, B, **kwargs):
  """Calcul de la contrainte d'attente : le temps d'attente moyen doit être inférieur à max_wait.
  
  Args:
      Q (int): Taille de queue (nombre de requêtes en attente)
      B (int): Taille de batch
      **kwargs: Paramètres du problème

  Returns:
      float: Contrainte d'attente
  """
  _, time_prefill, time_decoding = calculate_speed_constants(B, **kwargs)
  return time_prefill + (time_prefill + time_decoding) * max(Q - 1, 0) - kwargs['max_wait']
   