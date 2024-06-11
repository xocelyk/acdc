import os
from torch import Tensor
from transformer_lens import HookedTransformer
from data_loader import load_datasets
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from acdc import ACDC
import torch
import itertools


# Set device
torch.set_grad_enabled(False)
device = torch.device("mps") if torch.backends.mps.is_built() else "cpu"

# Load models
model = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)
model.to(device)

# Load datasets
ioi_dataset, abc_dataset = load_datasets(model, device)

# Run model and get caches
clean_logits, clean_cache = model.run_with_cache(ioi_dataset.toks, return_type='logits')
corrupted_logits, corrupted_cache = model.run_with_cache(abc_dataset.toks, return_type='logits')

# Define heads and layers
n_heads = 12
n_layers = 12

IOI_CIRCUIT = {
    "name mover": [
        (9, 9),  # ordered by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        (7, 1),
    ],
    "previous token": [
        (2, 2),
        (2, 9),
        (4, 11),
        (4, 3),
        (4, 7),
        (5, 6),
        (3, 3),
        (3, 7),
        (3, 6),
    ],
}

heads = list(itertools.chain(*IOI_CIRCUIT.values()))

acdc_config = {
    'model': model,
    'n_heads': n_heads,
    'n_layers': n_layers,
    'clean_cache': clean_cache,
    'corrupted_cache': corrupted_cache,
    'clean_dataset': ioi_dataset,
    'corrupted_dataset': abc_dataset,
    'clean_logits': clean_logits,
    'n_epochs': 1,
    'heads': heads,
    'verbose': True
}

acdc = ACDC(acdc_config)
acdc.run()

