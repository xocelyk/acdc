# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

t.set_grad_enabled(False)
torch = t

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
# assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part3_indirect_object_identification', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, scatter, bar
import tests as tests

device = t.device("mps") if torch.backends.mps.is_built() else "cpu"

MAIN = __name__ == "__main__"

# %% 1️⃣ MODEL & TASK SETUP


if MAIN:
	model = HookedTransformer.from_pretrained(
		"gpt2-small",
		center_unembed=True,
		center_writing_weights=True,
		fold_ln=True,
		refactor_factored_attn_matrices=True,
		device=device,
	)
	model.set_use_attn_result(True)
	model.set_use_split_qkv_input(True)
	# cfg = HookedTransformerConfig(
	# 	n_layers=2,  # Set the number of layers to 2
	# 	d_model=512,
	# 	d_head=64,
	# 	n_heads=8,
	# 	d_mlp=2048,
	# 	d_vocab=61,
	# 	n_ctx=59,
	# 	act_fn="gelu",
	# 	normalization_type="LNPre",
	# 	device=device,
	# )
	# model = HookedTransformer(cfg)

# Load the state dict as before, assuming the state dict is compatible with a two-layer model
# sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
# model.load_state_dict(sd)

# %%

# Here is where we test on a single prompt
# Result: 70% probability on Mary, as we expect


if MAIN:
	example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
	example_answer = " Mary"
	# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# %%


if MAIN:
	prompt_format = [
		"When John and Mary went to the shops,{} gave the bag to",
		"When Tom and James went to the park,{} gave the ball to",
		"When Dan and Sid went to the shops,{} gave an apple to",
		"After Martin and Amy went to the park,{} gave a drink to",
	]
	name_pairs = [
		(" John", " Mary"),
		(" Tom", " James"),
		(" Dan", " Sid"),
		(" Martin", " Amy"),
	]
	
	# Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
	prompts = [
		prompt.format(name) 
		for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1] 
	]
	# Define the answers for each prompt, in the form (correct, incorrect)
	answers = [names[::i] for names in name_pairs for i in (1, -1)]
	# Define the answer tokens (same shape as the answers)
	answer_tokens = t.concat([
		model.to_tokens(names, prepend_bos=False).T for names in answers
	])
	
	rprint(prompts)
	rprint(answers)
	rprint(answer_tokens)

# %%


if MAIN:
	table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")
	
	for prompt, answer in zip(prompts, answers):
		table.add_row(prompt, repr(answer[0]), repr(answer[1]))
	
	rprint(table)

# %%


if MAIN:
	tokens = model.to_tokens(prompts, prepend_bos=True)
	# Move the tokens to the GPU
	tokens = tokens.to(device)
	# Run the model and cache all activations
	original_logits, cache = model.run_with_cache(tokens)

# %%

if MAIN:
	def logits_to_ave_logit_diff(
		logits: Float[Tensor, "batch seq d_vocab"],
		answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
		per_prompt: bool = False
	) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
		'''
		Returns logit difference between the correct and incorrect answer.

		If per_prompt=True, return the array of differences rather than the average.
		'''
		# Only the final logits are relevant for the answer
		final_logits = logits[:, -1, :] # [batch d_vocab]
		# Get the logits corresponding to the indirect object / subject tokens respectively
		answer_logits = final_logits.gather(dim=-1, index=answer_tokens) # [batch 2]
		# Find logit difference
		correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
		answer_logit_diff = correct_logits - incorrect_logits
		return answer_logit_diff if per_prompt else answer_logit_diff.mean()



if MAIN:
	# tests.test_logits_to_ave_logit_diff(logits_to_ave_logit_diff)
	
	original_per_prompt_diff = logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
	print("Per prompt logit difference:", original_per_prompt_diff)
	original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
	print("Average logit difference:", original_average_logit_diff)
	
	cols = [
		"Prompt", 
		Column("Correct", style="rgb(0,200,0) bold"), 
		Column("Incorrect", style="rgb(255,0,0) bold"), 
		Column("Logit Difference", style="bold")
	]
	table = Table(*cols, title="Logit differences")
	
	for prompt, answer, logit_diff in zip(prompts, answers, original_per_prompt_diff):
		table.add_row(prompt, repr(answer[0]), repr(answer[1]), f"{logit_diff.item():.3f}")
	
	rprint(table)

# %% 2️⃣ LOGIT ATTRIBUTION


if MAIN:
	answer_residual_directions = model.tokens_to_residual_directions(answer_tokens) # [batch 2 d_model]
	print("Answer residual directions shape:", answer_residual_directions.shape)
	
	correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
	logit_diff_directions = correct_residual_directions - incorrect_residual_directions # [batch d_model]
	print(f"Logit difference directions shape:", logit_diff_directions.shape)

# %%

# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 

if MAIN:
	final_residual_stream = cache["resid_post", -1] # [batch seq d_model]
	print(f"Final residual stream shape: {final_residual_stream.shape}")
	final_token_residual_stream = final_residual_stream[:, -1, :] # [batch d_model]
	
	# Apply LayerNorm scaling (to just the final sequence position)
	# pos_slice is the subset of the positions we take - here the final token of each prompt
	scaled_final_token_residual_stream = cache.apply_ln_to_stack(final_token_residual_stream, layer=-1, pos_slice=-1)
	
	average_logit_diff = einops.einsum(
		scaled_final_token_residual_stream, logit_diff_directions,
		"batch d_model, batch d_model ->"
	) / len(prompts)
	
	print(f"Calculated average logit diff: {average_logit_diff:.10f}")
	print(f"Original logit difference:     {original_average_logit_diff:.10f}")
	
	# t.testing.assert_close(average_logit_diff, original_average_logit_diff)

# %%

if MAIN:
	def residual_stack_to_logit_diff(
		residual_stack: Float[Tensor, "... batch d_model"], 
		cache: ActivationCache,
		logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
	) -> Float[Tensor, "..."]:
		'''
		Gets the avg logit difference between the correct and incorrect answer for a given 
		stack of components in the residual stream.
		'''
		batch_size = residual_stack.size(-2)
		scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
		return einops.einsum(
			scaled_residual_stack, logit_diff_directions,
			"... batch d_model, batch d_model -> ..."
		) / batch_size


# Test function by checking that it gives the same result as the original logit difference

# if MAIN:
# 	t.testing.assert_close(
# 		residual_stack_to_logit_diff(final_token_residual_stream, cache),
# 		original_average_logit_diff
# 	)

# %%


if MAIN:
	accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
	# accumulated_residual has shape (component, batch, d_model)
	
	logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, cache) # [components]
	
	line(
		logit_lens_logit_diffs, 
		hovermode="x unified",
		title="Logit Difference From Accumulated Residual Stream",
		labels={"x": "Layer", "y": "Logit Diff"},
		xaxis_tickvals=labels,
		width=800
	)

# %%


if MAIN:
	per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
	per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)
	
	line(
		per_layer_logit_diffs, 
		hovermode="x unified",
		title="Logit Difference From Each Layer",
		labels={"x": "Layer", "y": "Logit Diff"},
		xaxis_tickvals=labels,
		width=800
	)

# %%


if MAIN:
	per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
	per_head_residual = einops.rearrange(
		per_head_residual, 
		"(layer head) ... -> layer head ...", 
		layer=model.cfg.n_layers
	)
	per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)
	
	imshow(
		per_head_logit_diffs, 
		labels={"x":"Head", "y":"Layer"}, 
		title="Logit Difference From Each Head",
		width=600
	)

# %%

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
	'''
	Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
	Returns a list of indices, i.e. shape [k, tensor.ndim].

	Example: if tensor is 2D array of values for each head in each layer, this will
	return a list of heads.
	'''
	i = t.topk(tensor.flatten(), k).indices
	return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()



if MAIN:
	k = 3
	
	for head_type in ["Positive", "Negative"]:
	
		# Get the heads with largest (or smallest) contribution to the logit difference
		top_heads = topk_of_Nd_tensor(per_head_logit_diffs * (1 if head_type=="Positive" else -1), k)
	
		# Get all their attention patterns
	attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
		cache["pattern", layer][:, head].mean(0)
		for layer, head in top_heads
	])

	# # Display results
	# display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))
	# display(cv.attention.attention_patterns(
	# 	attention = attn_patterns_for_important_heads,
	# 	tokens = model.to_str_tokens(tokens[0]),
	# 	attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
	# ))

# %% 3️⃣ ACTIVATION PATCHING

from transformer_lens import patching

# %%


if MAIN:
	clean_tokens = tokens
	# Swap each adjacent pair to get corrupted tokens
	indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(tokens))]
	corrupted_tokens = clean_tokens[indices]
	
	print(
		"Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
		"Corrupted string 0:", model.to_string(corrupted_tokens[0])
	)
	
	clean_logits, clean_cache = model.run_with_cache(clean_tokens)
	corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
	
	clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
	print(f"Clean logit diff: {clean_logit_diff:.4f}")
	
	corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
	print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

# %%

if MAIN:
	def ioi_metric(
		logits: Float[Tensor, "batch seq d_vocab"], 
		answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
		corrupted_logit_diff: float = corrupted_logit_diff,
		clean_logit_diff: float = clean_logit_diff,
	) -> Float[Tensor, ""]:
		'''
		Linear function of logit diff, calibrated so that it equals 0 when performance is 
		same as on corrupted input, and 1 when performance is same as on clean input.
		'''
		patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
		return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)



# if MAIN:
# 	t.testing.assert_close(ioi_metric(clean_logits).item(), 1.0)
# 	t.testing.assert_close(ioi_metric(corrupted_logits).item(), 0.0)
# 	t.testing.assert_close(ioi_metric((clean_logits + corrupted_logits) / 2).item(), 0.5)

# %%


if MAIN:
	act_patch_resid_pre = patching.get_act_patch_resid_pre(
		model = model,
		corrupted_tokens = corrupted_tokens,
		clean_cache = clean_cache,
		patching_metric = ioi_metric
	)
	
	labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
	
	imshow(
		act_patch_resid_pre, 
		labels={"x": "Position", "y": "Layer"},
		x=labels,
		title="resid_pre Activation Patching",
		width=600
	)

# %%

def patch_residual_component(
	corrupted_residual_component: Float[Tensor, "batch pos d_model"],
	hook: HookPoint, 
	pos: int, 
	clean_cache: ActivationCache
) -> Float[Tensor, "batch pos d_model"]:
	'''
	Patches a given sequence position in the residual stream, using the value
	from the clean cache.
	'''
	corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
	return corrupted_residual_component


def get_act_patch_resid_pre(
	model: HookedTransformer, 
	corrupted_tokens: Float[Tensor, "batch pos"], 
	clean_cache: ActivationCache, 
	patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "layer pos"]:
	'''
	Returns an array of results of patching each position at each layer in the residual
	stream, using the value from the clean cache.

	The results are calculated using the patching_metric function, which should be
	called on the model's logit output.
	'''
	model.reset_hooks()
	seq_len = corrupted_tokens.size(1)
	results = t.zeros(model.cfg.n_layers, seq_len, device=device, dtype=t.float32)

	for layer in tqdm(range(model.cfg.n_layers)):
		for position in range(seq_len):
			hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
			patched_logits = model.run_with_hooks(
				corrupted_tokens, 
				fwd_hooks = [(utils.get_act_name("resid_pre", layer), hook_fn)], 
			)
			results[layer, position] = patching_metric(patched_logits)

	return results




if MAIN:
	act_patch_resid_pre_own = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)
	
	# t.testing.assert_close(act_patch_resid_pre, act_patch_resid_pre_own)

# %%


if MAIN:
	imshow(
		act_patch_resid_pre_own, 
		x=labels, 
		title="Logit Difference From Patched Residual Stream", 
		labels={"x":"Sequence Position", "y":"Layer"},
		width=600 # If you remove this argument, the plot will usually fill the available space
	)

# %%


if MAIN:
	act_patch_block_every = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)
	
	imshow(
		act_patch_block_every,
		x=labels, 
		facet_col=0, # This argument tells plotly which dimension to split into separate plots
		facet_labels=["Residual Stream", "Attn Output", "MLP Output"], # Subtitles of separate plots
		title="Logit Difference From Patched Attn Head Output", 
		labels={"x": "Sequence Position", "y": "Layer"},
		width=1000,
	)

# %%

def get_act_patch_block_every(
	model: HookedTransformer, 
	corrupted_tokens: Float[Tensor, "batch pos"], 
	clean_cache: ActivationCache, 
	patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "layer pos"]:
	'''
	Returns an array of results of patching each position at each layer in the residual
	stream, using the value from the clean cache.

	The results are calculated using the patching_metric function, which should be
	called on the model's logit output.
	'''
	model.reset_hooks()
	results = t.zeros(3, model.cfg.n_layers, tokens.size(1), device=device, dtype=t.float32)

	for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
		for layer in tqdm(range(model.cfg.n_layers)):
			for position in range(corrupted_tokens.shape[1]):
				hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
				patched_logits = model.run_with_hooks(
					corrupted_tokens, 
					fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)], 
				)
				results[component_idx, layer, position] = patching_metric(patched_logits)

	return results

# %%


if MAIN:
	act_patch_block_every_own = get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)
	
	# t.testing.assert_close(act_patch_block_every, act_patch_block_every_own)
	
	imshow(
		act_patch_block_every_own,
		x=labels, 
		facet_col=0,
		facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
		title="Logit Difference From Patched Attn Head Output", 
		labels={"x": "Sequence Position", "y": "Layer"},
		width=1000
	)

# %%


if MAIN:
	act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
		model, 
		corrupted_tokens, 
		clean_cache, 
		ioi_metric
	)
	
	imshow(
		act_patch_attn_head_out_all_pos, 
		labels={"y": "Layer", "x": "Head"}, 
		title="attn_head_out Activation Patching (All Pos)",
		width=600
	)

# %%

def patch_head_vector(
	corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
	hook: HookPoint, 
	head_index: int, 
	clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
	'''
	Patches the output of a given head (before it's added to the residual stream) at
	every sequence position, using the value from the clean cache.
	'''
	corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
	return corrupted_head_vector


def get_act_patch_attn_head_out_all_pos(
	model: HookedTransformer, 
	corrupted_tokens: Float[Tensor, "batch pos"], 
	clean_cache: ActivationCache, 
	patching_metric: Callable
) -> Float[Tensor, "layer head"]:
	'''
	Returns an array of results of patching at all positions for each head in each
	layer, using the value from the clean cache.

	The results are calculated using the patching_metric function, which should be
	called on the model's logit output.
	'''
	model.reset_hooks()
	results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

	for layer in tqdm(range(model.cfg.n_layers)):
		for head in range(model.cfg.n_heads):
			hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
			patched_logits = model.run_with_hooks(
				corrupted_tokens, 
				fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)], 
				return_type="logits"
			)
			results[layer, head] = patching_metric(patched_logits)

	return results



if MAIN:
	act_patch_attn_head_out_all_pos_own = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, ioi_metric)
	
	# t.testing.assert_close(act_patch_attn_head_out_all_pos, act_patch_attn_head_out_all_pos_own)
	
	imshow(
		act_patch_attn_head_out_all_pos_own,
		title="Logit Difference From Patched Attn Head Output", 
		labels={"x":"Head", "y":"Layer"},
		width=600
	)

# %%


if MAIN:
	act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
		model, 
		corrupted_tokens, 
		clean_cache, 
		ioi_metric
	)
	
	imshow(
		act_patch_attn_head_all_pos_every, 
		facet_col=0, 
		facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
		title="Activation Patching Per Head (All Pos)", 
		labels={"x": "Head", "y": "Layer"},
	)

# %%

def patch_attn_patterns(
	corrupted_head_vector: Float[Tensor, "batch head_index pos_q pos_k"],
	hook: HookPoint, 
	head_index: int, 
	clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
	'''
	Patches the attn patterns of a given head at every sequence position, using 
	the value from the clean cache.
	'''
	corrupted_head_vector[:, head_index] = clean_cache[hook.name][:, head_index]
	return corrupted_head_vector


def get_act_patch_attn_head_all_pos_every(
	model: HookedTransformer,
	corrupted_tokens: Float[Tensor, "batch pos"],
	clean_cache: ActivationCache,
	patching_metric: Callable
) -> Float[Tensor, "layer head"]:
	'''
	Returns an array of results of patching at all positions for each head in each
	layer (using the value from the clean cache) for output, queries, keys, values
	and attn pattern in turn.

	The results are calculated using the patching_metric function, which should be
	called on the model's logit output.
	'''
	results = t.zeros(5, model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)
	# Loop over each component in turn
	for component_idx, component in enumerate(["z", "q", "k", "v", "pattern"]):
		for layer in tqdm(range(model.cfg.n_layers)):
			for head in range(model.cfg.n_heads):
				# Get different hook function if we're doing attention probs
				hook_fn_general = patch_attn_patterns if component == "pattern" else patch_head_vector
				hook_fn = partial(hook_fn_general, head_index=head, clean_cache=clean_cache)
				# Get patched logits
				patched_logits = model.run_with_hooks(
					corrupted_tokens,
					fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
					return_type="logits"
				)
				results[component_idx, layer, head] = patching_metric(patched_logits)

	return results



# if MAIN:
# 	act_patch_attn_head_all_pos_every_own = get_act_patch_attn_head_all_pos_every(
# 		model,
# 		corrupted_tokens,
# 		clean_cache,
# 		ioi_metric
# 	)
	
# 	# t.testing.assert_close(act_patch_attn_head_all_pos_every, act_patch_attn_head_all_pos_every_own)
	
# 	imshow(
# 		act_patch_attn_head_all_pos_every_own,
# 		facet_col=0,
# 		facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
# 		title="Activation Patching Per Head (All Pos)",
# 		labels={"x": "Head", "y": "Layer"},
# 		width=1200
# 	)

# %% 4️⃣ PATH PATCHING

from ioi_dataset import NAMES, IOIDataset

# %%


if MAIN:
	N = 25
	ioi_dataset = IOIDataset(
		prompt_type="mixed",
		N=N,
		tokenizer=model.tokenizer,
		prepend_bos=False,
		seed=1,
		device=str(device)
	)

# %%


if MAIN:
	abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

# %%

def format_prompt(sentence: str) -> str:
	'''Format a prompt by underlining names (for rich print)'''
	return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
	'''Makes and displays a table, from cols rather than rows (using rich print)'''
	table = Table(*colnames, title=title)
	rows = list(zip(*cols))
	f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
	for row in rows[:n_rows]:
		table.add_row(*list(map(f, row)))
	rprint(table)

# %%


if MAIN:
	make_table(
		colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
		cols = [
			map(format_prompt, ioi_dataset.sentences), 
			model.to_string(ioi_dataset.s_tokenIDs).split(), 
			model.to_string(ioi_dataset.io_tokenIDs).split(), 
			map(format_prompt, abc_dataset.sentences), 
		],
		title = "Sentences from IOI vs ABC distribution",
	)

# %%

if MAIN:
	def logits_to_ave_logit_diff_2(
		logits: Float[Tensor, "batch seq d_vocab"],
		ioi_dataset: IOIDataset = ioi_dataset,
		per_prompt=False
	) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
		'''
		Returns logit difference between the correct and incorrect answer.

		If per_prompt=True, return the array of differences rather than the average.
		'''
		
		# Only the final logits are relevant for the answer
		# Get the logits corresponding to the indirect object / subject tokens respectively
		io_logits = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs] # [batch]
		s_logits = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs] # [batch]
		# Find logit difference
		answer_logit_diff = io_logits - s_logits
		return answer_logit_diff if per_prompt else answer_logit_diff.mean()



if MAIN:
	model.reset_hooks(including_permanent=True)
	
	ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
	abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)
	
	ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
	abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)
	
	ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
	abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

# %%


if MAIN:
	print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
	print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")
	
	make_table(
		colnames = ["IOI prompt", "IOI logit diff", "ABC prompt", "ABC logit diff"],
		cols = [
			map(format_prompt, ioi_dataset.sentences), 
			ioi_per_prompt_diff,
			map(format_prompt, abc_dataset.sentences), 
			abc_per_prompt_diff,
		],
		title = "Sentences from IOI vs ABC distribution",
	)

# %%

if MAIN:
	def ioi_metric_2(
		logits: Float[Tensor, "batch seq d_vocab"],
		clean_logit_diff: float = ioi_average_logit_diff,
		corrupted_logit_diff: float = abc_average_logit_diff,
		ioi_dataset: IOIDataset = ioi_dataset,
	) -> float:
		'''
		We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
		and -1 when performance has been destroyed (i.e. is same as ABC dataset).
		'''
		patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
		return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)



if MAIN:
	print(f"IOI metric (IOI dataset): {ioi_metric_2(ioi_logits_original):.4f}")
	print(f"IOI metric (ABC dataset): {ioi_metric_2(abc_logits_original):.4f}")

# %%


def patch_or_freeze_head_vectors(
	orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
	hook: HookPoint, 
	new_cache: ActivationCache,
	orig_cache: ActivationCache,
	head_to_patch: Tuple[int, int], 
) -> Float[Tensor, "batch pos head_index d_head"]:
	'''
	This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them
	to their values in orig_cache), except for head_to_patch (if it's in this layer) which
	we patch with the value from new_cache.

	head_to_patch: tuple of (layer, head)
		we can use hook.layer() to check if the head to patch is in this layer
	'''
	# Setting using ..., otherwise changing orig_head_vector will edit cache value too
	orig_head_vector[...] = orig_cache[hook.name][...]
	if head_to_patch[0] == hook.layer():
		orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
	return orig_head_vector


if MAIN:
	def get_path_patch_head_to_final_resid_post(
		model: HookedTransformer,
		patching_metric: Callable,
		new_dataset: IOIDataset = abc_dataset,
		orig_dataset: IOIDataset = ioi_dataset,
		new_cache: Optional[ActivationCache] = abc_cache,
		orig_cache: Optional[ActivationCache] = ioi_cache,
	) -> Float[Tensor, "layer head"]:
		'''
		Performs path patching (see algorithm in appendix B of IOI paper), with:

			sender head = (each head, looped through, one at a time)
			receiver node = final value of residual stream

		Returns:
			tensor of metric values for every possible sender head
		'''
		model.reset_hooks()
		results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

		resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
		resid_post_name_filter = lambda name: name == resid_post_hook_name


		# ========== Step 1 ==========
		# Gather activations on x_orig and x_new

		# Note the use of names_filter for the run_with_cache function. Using it means we 
		# only cache the things we need (in this case, just attn head outputs).
		z_name_filter = lambda name: name.endswith("z")
		if new_cache is None:
			_, new_cache = model.run_with_cache(
				new_dataset.toks, 
				names_filter=z_name_filter, 
				return_type=None
			)
		if orig_cache is None:
			_, orig_cache = model.run_with_cache(
				orig_dataset.toks, 
				names_filter=z_name_filter, 
				return_type=None
			)


		# Looping over every possible sender head (the receiver is always the final resid_post)
		# Note use of itertools (gives us a smoother progress bar)
		for (sender_layer, sender_head) in tqdm(list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):

			# ========== Step 2 ==========
			# Run on x_orig, with sender head patched from x_new, every other head frozen
			
			hook_fn = partial(
				patch_or_freeze_head_vectors,
				new_cache=new_cache, 
				orig_cache=orig_cache,
				head_to_patch=(sender_layer, sender_head),
			)
			model.add_hook(z_name_filter, hook_fn)
		
			_, patched_cache = model.run_with_cache(
				orig_dataset.toks, 
				names_filter=resid_post_name_filter, 
				return_type=None
			)

			assert set(patched_cache.keys()) == {resid_post_hook_name}

			# ========== Step 3 ==========
			# Unembed the final residual stream value, to get our patched logits

			patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

			# Save the results
			results[sender_layer, sender_head] = patching_metric(patched_logits)

		return results



if MAIN:
	path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric_2)
	
	imshow(
		100 * path_patch_head_to_final_resid_post,
		title="Direct effect on logit difference",
		labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
		coloraxis=dict(colorbar_ticksuffix = "%"),
		width=600,
	)

# %%


def patch_head_input(
	orig_activation: Float[Tensor, "batch pos head_idx d_head"],
	hook: HookPoint,
	patched_cache: ActivationCache,
	head_list: List[Tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
	'''
	Function which can patch any combination of heads in layers,
	according to the heads in head_list.
	'''
	heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
	orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
	return orig_activation


if MAIN:
	def get_path_patch_head_to_heads(
		receiver_heads: List[Tuple[int, int]],
		receiver_input: str,
		model: HookedTransformer,
		patching_metric: Callable,
		new_dataset: IOIDataset = abc_dataset,
		orig_dataset: IOIDataset = ioi_dataset,
		new_cache: Optional[ActivationCache] = None,
		orig_cache: Optional[ActivationCache] = None,
	) -> Float[Tensor, "layer head"]:
		'''
		Performs path patching (see algorithm in appendix B of IOI paper), with:

			sender head = (each head, looped through, one at a time)
			receiver node = input to a later head (or set of heads)

		The receiver node is specified by receiver_heads and receiver_input.
		Example (for S-inhibition path patching the queries):
			receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
			receiver_input = "v"

		Returns:
			tensor of metric values for every possible sender head
		'''
		model.reset_hooks()

		assert receiver_input in ("k", "q", "v")
		receiver_layers = set(next(zip(*receiver_heads)))
		receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
		receiver_hook_names_filter = lambda name: name in receiver_hook_names

		results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=device, dtype=t.float32)
		
		# ========== Step 1 ==========
		# Gather activations on x_orig and x_new

		# Note the use of names_filter for the run_with_cache function. Using it means we 
		# only cache the things we need (in this case, just attn head outputs).
		z_name_filter = lambda name: name.endswith("z")
		if new_cache is None:
			_, new_cache = model.run_with_cache(
				new_dataset.toks, 
				names_filter=z_name_filter, 
				return_type=None
			)
		if orig_cache is None:
			_, orig_cache = model.run_with_cache(
				orig_dataset.toks, 
				names_filter=z_name_filter, 
				return_type=None
			)

		# Note, the sender layer will always be before the final receiver layer, otherwise there will
		# be no causal effect from sender -> receiver. So we only need to loop this far.
		for (sender_layer, sender_head) in tqdm(list(itertools.product(
			range(max(receiver_layers)),
			range(model.cfg.n_heads)
		))):

			# ========== Step 2 ==========
			# Run on x_orig, with sender head patched from x_new, every other head frozen

			hook_fn = partial(
				patch_or_freeze_head_vectors,
				new_cache=new_cache, 
				orig_cache=orig_cache,
				head_to_patch=(sender_layer, sender_head),
			)
			model.add_hook(z_name_filter, hook_fn, level=1)
			
			_, patched_cache = model.run_with_cache(
				orig_dataset.toks, 
				names_filter=receiver_hook_names_filter,  
				return_type=None
			)
			# model.reset_hooks(including_permanent=True)
			assert set(patched_cache.keys()) == set(receiver_hook_names)

			# ========== Step 3 ==========
			# Run on x_orig, patching in the receiver node(s) from the previously cached value
			
			hook_fn = partial(
				patch_head_input, 
				patched_cache=patched_cache, 
				head_list=receiver_heads,
			)
			patched_logits = model.run_with_hooks(
				orig_dataset.toks,
				fwd_hooks = [(receiver_hook_names_filter, hook_fn)], 
				return_type="logits"
			)

			# Save the results
			results[sender_layer, sender_head] = patching_metric(patched_logits)

		return results

# %%


if MAIN:
	model.reset_hooks()
	
	s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
		receiver_heads = [(8, 6)],
		receiver_input = "v",
		model = model,
		patching_metric = ioi_metric_2
		# patching_metric=kl_divergence,
	)
	
	imshow(
		100 * s_inhibition_value_path_patching_results,
		title="Direct effect on S-Inhibition Heads' values", 
		labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
		width=600,
		coloraxis=dict(colorbar_ticksuffix = "%"),
	)
else:
	pass
# %%

# load a simple two layer transformer model from HookedTransformer
from transformer_lens import HookedTransformer

n_layers = 12
n_heads = 12


# %%


def build_ioi_computational_graph():
	g = {} # receiver -> sender
	heads = [(i, j) for i in range(n_layers) for j in range(n_heads)]
	g['resid_pre'] = []
	g['resid_post'] = []
	for head in heads:
		g[head] = []
		g[head].append('resid_pre')
		g['resid_post'].append(head)
	for send in heads:
		for rec in heads:
			if send[0] < rec[0]:
				g[rec].append(send)
	return g

from collections import OrderedDict


def topological_sort(graph):
	visted = set()
	stack = []
	def dfs(node):
		if node not in visted:
			visted.add(node)
			for neighbor in graph[node]:
				dfs(neighbor)
			stack.append(node)
	for node in graph:
		dfs(node)
	return OrderedDict((k, graph[k]) for k in reversed(stack))


g = build_ioi_computational_graph()
g = topological_sort(g)
# show topological sort
print(g)




# %%

def patch_head_input(
	orig_activation: Float[Tensor, "batch pos head_idx d_head"],
	hook: HookPoint,
	patched_cache: ActivationCache,
	head_list: List[Tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
	'''
	Function which can patch any combination of heads in layers,
	according to the heads in head_list.
	'''
	heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
	orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
	return orig_activation

def patch_path(
	activation: Float[Tensor, "batch pos head_idx d_vocab"],
	hook: HookPoint,
	corrupted_src_activation: Float[Tensor, "batch pos d_vocab"],
	clean_src_activation: Float[Tensor, "batch pos d_vocab"],
	src: Tuple[int, int],
	receiver: Tuple[int, int],
):
	'''
	Performs path patching algorithm from the ACDC paper
	For each receiver head, we add (corrupted sender output - clean sender output) to the receiver input
	'''
	print(hook.name)
	if hook.layer() == receiver[0]:
		# We only need to patch the receiver layer
		# We patch the receiver layer with the cache activations from the src layer
		activation[:, :, receiver[1], :] += (corrupted_src_activation - clean_src_activation)
	return activation

orig_logits, orig_cache = model.run_with_cache(
	ioi_dataset.toks, 
	return_type='logits'
)

def kl_divergence(patched_logits, orig_logits=orig_logits) -> float:
	'''
	Returns the KL divergence between two probability distributions averaged over batches.
	'''
    # Select the last logits from each sequence for comparison
	p_logits = orig_logits[:, -1, :]
	q_logits = patched_logits[:, -1, :]

    # Compute log-probabilities for p
	p_log_probs = torch.nn.functional.log_softmax(p_logits, dim=-1)
    # Compute probabilities for q to use in manual KL calculation
	q_log_probs = torch.nn.functional.log_softmax(q_logits, dim=-1)
	q_probs = torch.exp(q_log_probs)

	# why am I getting negative values??? TODO

    # Calculate the KL divergence manually
	kl_div = (q_probs * (torch.log(q_probs) - p_log_probs)).sum(dim=-1)

    # Return the average KL divergence over all batches
	return kl_div.mean().item()
# %%
def remove_path_head_to_head(
	src: Tuple[int, int],
	receiver: Tuple[int, int],
	receiver_input: str,
	model: HookedTransformer,
	corrupted_dataset: IOIDataset = abc_dataset,
	orig_dataset: IOIDataset = ioi_dataset,
	new_cache: Optional[ActivationCache] = None,
	orig_cache: Optional[ActivationCache] = None,
	orig_logits: Optional[Float[Tensor, "batch seq d_vocab"]] = None,
) -> float:
	'''
	Performs path patching on edge from src to receiver heads and returns the KL divergence
	'''
	if src[0] >= receiver[0]:
		return 0
	
	model.reset_hooks()

	# we go over all receiver inputs for now
	# receiver_inputs = ["k", "q", "v"]
	receiver_inputs = ["q_input", "k_input", "v_input"]


	assert receiver_input in ("k", "q", "v")
	receiver_layer = receiver[0]
	receiver_hook_names = [utils.get_act_name(receiver_input, receiver_layer) for receiver_input in receiver_inputs]
	receiver_hook_names_filter = lambda name: name in receiver_hook_names

	src_layer, src_head = src

	# ========== Step 1 ==========
	# Gather activations on x_orig and x_new

	# Note the use of names_filter for the run_with_cache function. Using it means we 
	# only cache the things we need (in this case, just attn head outputs).
	# z_name_filter = lambda name: name.endswith("z")
	z_name_filter = lambda name: name.endswith("result")

	if new_cache is None:
		_, new_cache = model.run_with_cache(
			corrupted_dataset.toks, 
			names_filter=z_name_filter, 
			return_type=None
		)
	if orig_cache is None: # we need to keep the logits to compare to the patched logits
		orig_logits, orig_cache = model.run_with_cache(
			orig_dataset.toks, 
			return_type='logits'
		)

	model.reset_hooks()

	clean_src_activation = orig_cache[utils.get_act_name("result", src_layer)][:, :, src_head, :]
	corrupted_src_activation = new_cache[utils.get_act_name("result", src_layer)][:, :, src_head, :]
	# clean_src_activation = orig_cache[utils.get_act_name("z", src_layer)][:, :, src_head, :]
	# corrupted_src_activation = new_cache[utils.get_act_name("z", src_layer)][:, :, src_head, :]
	
	# Note, the sender layer will always be before the final receiver layer, otherwise there will
	# be no causal effect from sender -> receiver. So we only need to loop this far.


	# ========== Step 2 ==========
	# Run on x_orig, with sender head patched from x_new, every other head frozen

	hook_fn = partial(
		patch_path,
		corrupted_src_activation=corrupted_src_activation,
		clean_src_activation=clean_src_activation,
		src=src,
		receiver=receiver,
	)
		
	model.add_hook(receiver_hook_names_filter, hook_fn, level=1)
	
	patched_logits, _ = model.run_with_cache(
		orig_dataset.toks, 
		return_type='logits'
	)
	model.reset_hooks()
	# ========== Step 3 ==========
	# Calculate KL divergence between the patched logits and the original logits
	kl_div = kl_divergence(orig_logits, patched_logits)
	return kl_div
		


# %%

result_name_filter = lambda name: name.endswith("result")

orig_logits, orig_cache = model.run_with_cache(
	ioi_dataset.toks, 
	names_filter=result_name_filter,
	return_type='logits'
)
print(orig_cache.keys())


results = [[0 for _ in range(n_heads)] for _ in range(n_layers)]
srcs = [(i, j) for i in range(n_layers) for j in range(n_heads)]
receivers = [(8, 6)]

for i, src in enumerate(srcs):
	for j, rec in enumerate(receivers):
		i_results = i // n_heads
		j_results = i % n_heads
		print()
		print(src)
		results[i_results][j_results] += remove_path_head_to_head(src, rec, "v", model, orig_cache=orig_cache)
		print(results[i_results][j_results])
results = t.tensor(results, dtype=t.float32, device=device)
# reshape
results = results.view(n_layers, n_heads)
# # normalize
# results = (results - results.min()) / (results.max() - results.min())


# %%
if MAIN:
	# path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric_2)
	
	imshow(
		100 * results,
		title="Direct effect on logit difference",
		labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
		coloraxis=dict(colorbar_ticksuffix = "%"),
		width=600,
	)
# %%

# def patch_or_freeze_head_vectors(
# 	orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
# 	hook: HookPoint, 
# 	new_cache: ActivationCache,
# 	orig_cache: ActivationCache,
# 	head_to_patch: Tuple[int, int], 
# ) -> Float[Tensor, "batch pos head_index d_head"]:
# 	'''
# 	This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them
# 	to their values in orig_cache), except for head_to_patch (if it's in this layer) which
# 	we patch with the value from new_cache.

# 	head_to_patch: tuple of (layer, head)
# 		we can use hook.layer() to check if the head to patch is in this layer
# 	'''
# 	# Setting using ..., otherwise changing orig_head_vector will edit cache value too
# 	orig_head_vector[...] = orig_cache[hook.name][...]
# 	if head_to_patch[0] == hook.layer():
# 		orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
# 	return orig_head_vector

# def patch_path_to_resid_post(
# 	activation: Float[Tensor, "batch pos head_idx d_vocab"],
# 	hook: HookPoint,
# 	corrupted_src_activation: Float[Tensor, "batch pos d_vocab"],
# 	clean_src_activation: Float[Tensor, "batch pos d_vocab"],
# 	src: Tuple[int, int],
# ):
# 	'''
# 	Performs path patching algorithm from the ACDC paper
# 	For each receiver head, we add (corrupted sender output - clean sender output) to the receiver input
# 	'''
# 	new_act = activation[...]
# 	new_act[:, :, src[1], :] += corrupted_src_activation - clean_src_activation
# 	return new_act

# def remove_path_head_to_final_resid_post(
# 	src: Tuple[int, int],
# 	model: HookedTransformer,
# 	corrupted_dataset: IOIDataset = abc_dataset,
# 	orig_dataset: IOIDataset = ioi_dataset,
# 	new_cache: Optional[ActivationCache] = None,
# 	orig_cache: Optional[ActivationCache] = None,
# ) -> float:
# 	'''
# 	Performs path patching (see algorithm in appendix B of IOI paper), with:

# 		sender head = (each head, looped through, one at a time)
# 		receiver node = final value of residual stream

# 	Returns:
# 		tensor of metric values for every possible sender head
# 	'''
	
# 	model.reset_hooks()

# 	resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
# 	resid_post_name_filter = lambda name: name == resid_post_hook_name
# 	print(resid_post_hook_name)

# 	src_layer, src_head = src

# 	# ========== Step 1 ==========
# 	# Gather activations on x_orig and x_new

# 	# Note the use of names_filter for the run_with_cache function. Using it means we 
# 	# only cache the things we need (in this case, just attn head outputs).
# 	z_name_filter = lambda name: name.endswith("z")
# 	z_and_resid_post_name_filter = lambda name: name.endswith("z") or name == resid_post_hook_name
# 	if new_cache is None:
# 		_, new_cache = model.run_with_cache(
# 			corrupted_dataset.toks, 
# 			names_filter=z_name_filter, 
# 			return_type=None
# 		)
# 	if orig_cache is None:
# 		orig_logits, orig_cache = model.run_with_cache(
# 			orig_dataset.toks, 
# 			names_filter=z_and_resid_post_name_filter, 
# 			return_type='logits'
# 		)
# 	model.reset_hooks()
# 	clean_src_activation = orig_cache[utils.get_act_name("z", src_layer)][:, :, src_head, :]
# 	corrupted_src_activation = new_cache[utils.get_act_name("z", src_layer)][:, :, src_head, :]
	
# 	resid_post = orig_cache[resid_post_hook_name]

# 	W_O = model.W_O[-1, src_head, :, :]
# 	activation_difference_out = (corrupted_src_activation - clean_src_activation) @ W_O
# 	print(activation_difference_out.sum())
# 	resid_post[:, src_head, :] += activation_difference_out[:, src_head, :]
# 	patched_logits = model.unembed(model.ln_final(resid_post))

# 	# ========== Step 3 ==========
# 	# Calculate KL divergence between the patched logits and the original logits
# 	kl_div = kl_divergence(orig_logits, patched_logits)
# 	return kl_div


# # %%
# results = [[0 for _ in range(n_heads)] for _ in range(n_layers)]
# srcs = [(i, j) for i in range(n_layers) for j in range(n_heads)]

# for i, src in enumerate(srcs):
# 	i_results = i // n_heads
# 	j_results = i % n_heads
# 	results[i_results][j_results] += remove_path_head_to_final_resid_post(src, model)
# 	print(src)
# 	print(results[i_results][j_results])
# 	print()
# results = t.tensor(results, dtype=t.float32, device=device)
# # reshape
# results = results.view(n_layers, n_heads)
# # normalize
# results = (results - results.min()) / (results.max() - results.min())
# # %%
# imshow(
# 	100 * results,
# 	title="Direct effect on logit difference",
# 	labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
# 	coloraxis=dict(colorbar_ticksuffix = "%"),
# 	width=600,
# )
# # %%
# results
# %%
	

