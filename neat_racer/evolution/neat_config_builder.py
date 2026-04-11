"""Build neat-python config text from experiment settings."""

from __future__ import annotations

from pathlib import Path

from neat_racer.config import ExperimentConfig
from neat_racer.constants import NEAT_OUTPUT_COUNT


def observation_size(cfg: ExperimentConfig) -> int:
    extras = 3 + (1 if cfg.sensors.include_alignment_feature else 0)
    return len(cfg.sensors.angles_deg) + extras


def render_neat_config(cfg: ExperimentConfig) -> str:
    num_inputs = observation_size(cfg)
    pop_size = cfg.neat.population_size

    return f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000000
pop_size              = {pop_size}
reset_on_extinction   = {str(cfg.neat.reset_on_extinction)}
no_fitness_termination = False

[DefaultGenome]
activation_default      = {cfg.neat.activation_default}
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.5
conn_delete_prob        = 0.3

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

node_add_prob           = 0.2
node_delete_prob        = 0.1

num_hidden              = {cfg.neat.hidden_nodes}
num_inputs              = {num_inputs}
num_outputs             = {NEAT_OUTPUT_COUNT}

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

single_structural_mutation = False
structural_mutation_surer = default

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = {cfg.neat.survival_threshold}
min_species_size   = 2
"""


def write_neat_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_neat_config(cfg), encoding="utf-8")
    return out_path
