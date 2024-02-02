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
# %%
from utils import *

# %%

t.set_grad_enabled(False)
torch = t
from models import Config, DemoTransformer
device = torch.device("mps") if torch.backends.mps.is_built() else "cpu"

# %%

def load_demo_gpt2():
	reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
	demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
	demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
	return demo_gpt2

# %%
# model = HookedTransformer.from_pretrained("solu-2l", fold_ln=False, center_unembed=False, center_writing_weights=False)
model = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
model.tokenizer = reference_gpt2.tokenizer
model.to(device)
reference_gpt2.to(device)
# %%
from ioi_dataset import NAMES, IOIDataset
N = 25
ioi_dataset = IOIDataset(
	prompt_type="mixed",
	N=N,
	tokenizer=reference_gpt2.tokenizer,
	prepend_bos=False,
	seed=1,
	device=str(device)
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
ioi_dataset.to(device)
abc_dataset.to(device)
n_heads = 12
n_layers = 12


# %%

clean_logits, clean_cache = model.run_with_cache(
	ioi_dataset.toks, 
	return_type='logits'
)
_, corrupted_cache = model.run_with_cache(
	abc_dataset.toks, 
	return_type='logits'
)


# %%


from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from graphviz import Digraph


class HookNode():
	def __init__(self, name, index, dir='IN'):
		self.name = name
		self.index = index
		self.dir = dir

	def __repr__(self):
		if 'resid' in self.name:
			idx_val = ''
		else:
			idx_val = f'head {self.index[2]}'
		return f'{self.name} {idx_val}'
	
	def __eq__(self, other):
		return self.name == other.name and self.index == other.index
	
	def __hash__(self):
		# we need a hash function to use the HookNode as a key in a dictionary
		idx_val = 0
		for i, val in enumerate(self.index):
			if isinstance(val, int):
				idx_val += val * 10 ** i
		return hash((self.name, idx_val))

class Edge():
	def __init__(self, sender, receiver, type='ADD'):
		self.sender = sender
		self.receiver = receiver
		self.type = type

	def __repr__(self):
		return f"{self.sender} -> {self.receiver}"

	def __eq__(self, other):
		return self.sender == other.sender and self.receiver == other.receiver

	def __hash__(self):
		return hash((self.sender, self.receiver))
	

class ComputationalGraph():
	def __init__(self, n_heads, n_layers, heads, empty=False):
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.heads = heads
		if empty:
			self.graph = OrderedDict()
		else:
			self.graph = self.build_ioi_computational_graph()
			self.graph = self.topological_sort()	
		
	def build_ioi_computational_graph(self):
		# TODO: add MLPs?
		g = defaultdict(list) # receiver -> senders
		# create heads subgraph
		heads = self.heads
		for send in heads:
			for rec in heads:
				if send[0] < rec[0]:
					send_layer, send_head = send
					rec_layer, rec_head = rec
					send_hook_name = utils.get_act_name("result", send_layer)
					send_hook_index = (slice(None), slice(None), send_head, slice(None))
					rec_hook_name_q = utils.get_act_name("q_input", rec_layer)
					rec_hook_name_k = utils.get_act_name("k_input", rec_layer)
					rec_hook_name_v = utils.get_act_name("v_input", rec_layer)
					rec_hook_index = (slice(None), slice(None), rec_head, slice(None))
					sender = HookNode(send_hook_name, send_hook_index, dir='OUT')
					receiver_q = HookNode(rec_hook_name_q, rec_hook_index, dir='IN')
					receiver_k = HookNode(rec_hook_name_k, rec_hook_index, dir='IN')
					receiver_v = HookNode(rec_hook_name_v, rec_hook_index, dir='IN')
					g[receiver_q].append(sender)
					g[receiver_k].append(sender)
					g[receiver_v].append(sender)

		# add resid pre and resid post
		resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
		resid_pre_hook_index = (slice(None), slice(None), slice(None))
		resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, dir='OUT')
		g[resid_pre] = []
		for rec in heads:
			rec_layer, rec_head = rec
			rec_hook_name_q = utils.get_act_name("q_input", rec_layer)
			rec_hook_name_k = utils.get_act_name("k_input", rec_layer)
			rec_hook_name_v = utils.get_act_name("v_input", rec_layer)
			rec_hook_index = (slice(None), slice(None), rec_head, slice(None))
			receiver_q = HookNode(rec_hook_name_q, rec_hook_index, dir='IN')
			receiver_k = HookNode(rec_hook_name_k, rec_hook_index, dir='IN')
			receiver_v = HookNode(rec_hook_name_v, rec_hook_index, dir='IN')
			g[receiver_q].append(resid_pre)
			g[receiver_k].append(resid_pre)
			g[receiver_v].append(resid_pre)

		resid_post_hook_name = utils.get_act_name("resid_post", self.n_layers - 1)
		resid_post_hook_index = (slice(None), slice(None), slice(None))
		resid_post = HookNode(resid_post_hook_name, resid_post_hook_index, dir='IN')
		g[resid_post] = [resid_pre]
		for head in heads:
			head_layer, head_head = head
			head_hook_name = utils.get_act_name("result", head_layer)
			head_hook_index = (slice(None), slice(None), head_head, slice(None))
			head = HookNode(head_hook_name, head_hook_index, dir='OUT')
			g[resid_post].append(head)

		# now create direct computation edges between head inputs and outputs
		for head in heads:
			head_layer, head_head = head
			head_out_hook_name = utils.get_act_name("result", head_layer)
			head_out_hook_index = (slice(None), slice(None), head_head, slice(None))
			head_out = HookNode(head_out_hook_name, head_out_hook_index, dir='OUT')
			head_q_input_hook_name = utils.get_act_name("q_input", head_layer)
			head_q_input_hook_index = (slice(None), slice(None), head_head, slice(None))
			head_q_input = HookNode(head_q_input_hook_name, head_q_input_hook_index, dir='IN')
			head_k_input_hook_name = utils.get_act_name("k_input", head_layer)
			head_k_input_hook_index = (slice(None), slice(None), head_head, slice(None))
			head_k_input = HookNode(head_k_input_hook_name, head_k_input_hook_index, dir='IN')
			head_v_input_hook_name = utils.get_act_name("v_input", head_layer)
			head_v_input_hook_index = (slice(None), slice(None), head_head, slice(None))
			head_v_input = HookNode(head_v_input_hook_name, head_v_input_hook_index, dir='IN')
			g[head_out].append(head_q_input)
			g[head_out].append(head_k_input)
			g[head_out].append(head_v_input)
		return g

	def topological_sort(self):
		visted = set()
		stack = []
		def dfs(node):
			if node not in visted:
				visted.add(node)
				for neighbor in self.graph[node]:
					dfs(neighbor)
				stack.append(node)
		for node in self.graph:
			dfs(node)
		return OrderedDict((k, self.graph[k]) for k in reversed(stack))

	def get_incoming_edges(self, node):
		edges = []
		for rec, senders in self.graph.items():
			if node == rec:
				for sender in senders:
					direct_computation_edge = rec.dir == 'OUT' and sender.dir == 'IN'
					if not direct_computation_edge:
						edges.append(Edge(sender, rec, type='ADD'))
		return edges
	
	def remove_edge(self, edge):
		self.graph[edge.receiver].remove(edge.sender)
	
	def add_edge(self, edge):
		if edge.receiver not in self.graph:
			self.graph[edge.receiver] = []
		self.graph[edge.receiver].append(edge.sender)

	def clean_dead_edges(self):
		# TODO: I don't really use the edge type at the moment, just the node type
		# remove direct computation edges that are not in the graph
		# copy self.graph
		result_graph = self.graph.copy()
		# remove all receiver nodes that only have direct computation edges
		# where direct computation is defined as OUT -> IN
		for rec, senders in self.graph.items():
			rec_type = rec.dir
			if rec_type == 'OUT':
				# check if all senders are IN
				all_senders_in = all(sender.dir == 'IN' for sender in senders)
				if all_senders_in:
					del result_graph[rec]
		self.graph = result_graph
		result_graph = self.graph.copy()
		# remove all receiver nodes that have no senders
		for rec, senders in self.graph.items():
			if len(senders) == 0:
				del result_graph[rec]
		self.graph = result_graph

	def get_all_edges(self):
		edges = []
		for k, v in self.graph.items():
			for sender in v:
				direct_computation_edge = k.dir == 'OUT' and sender.dir == 'IN'
				type = 'ADD' if not direct_computation_edge else 'DIRECT'
				edges.append(Edge(sender, k, type=type))
		return edges
	
	from graphviz import Digraph

	def visualize(self, filename='out'):
		'''
		Visualizes the computational graph.
		:param filename: Name of the file to save the visualization.
		'''
		dot = Digraph(comment='Computational Graph', engine='dot')

		# Graph attributes
		dot.attr('graph', size='12,12', ranksep='1', nodesep='0.5', dpi='300')
		
		# Node attributes
		dot.attr('node', shape='circle', style='filled', fillcolor='lightgrey', fontcolor='black', fontsize='12')

		# Edge attributes
		dot.attr('edge', color='black', arrowsize='0.5')

		# Add nodes and edges to the graph
		for node, edges in self.graph.items():
			dot.node(str(node), str(node))
			for edge in edges:
				dot.edge(str(node), str(edge))

		dot.render(filename, view=True)

	def __repr__(self):
		return f"{self.graph}"
		
class ACDC():
	def __init__(self, model, n_heads, n_layers, clean_cache, corrupted_cache, clean_dataset, corrupted_dataset, clean_logits, heads=None):
		self.model = model
		self.clean_cache = clean_cache
		self.corrupted_cache = corrupted_cache
		self.n_heads = n_heads
		self.n_layers = n_layers
		if heads is None:
			self.heads = [(i, j) for i in range(n_layers) for j in range(n_heads)]
		else:
			self.heads = heads
		self.clean_dataset = clean_dataset
		self.corrupted_dataset: IOIDataset = corrupted_dataset
		self.clean_logits: Optional[Float[Tensor, "batch seq d_vocab"]] = clean_logits
		self.G = ComputationalGraph(n_heads, n_layers, self.heads)
		self.H = ComputationalGraph(n_heads, n_layers, self.heads)
		# self.kl_threshold = .0575
		self.kl_threshold = 1e-4
		self.bad_edges = []
		self.num_epochs = 1
		
		
	def run(self):
		'''
		Rather than imagining pruning edges from the graph,
		we consider adding accumulating "bad edges" which are corrupted at runtime
		'''
		# remember, graph is a topologically sorted ordered dict
		for epoch in range(self.num_epochs):
			for receiver, senders in self.G.graph.items():
				print('Receiver: {}'.format(receiver))
				print()
				incoming_edges = self.G.get_incoming_edges(receiver)
				bad_edges = self.bad_edges[:]
				for edge in incoming_edges:
					kl_div_prev = self.ablate_paths(bad_edges)
					print('Testing edge from {} to {}'.format(edge.sender, edge.receiver))
					bad_edges.append(edge)
					kl_div_new = self.ablate_paths(bad_edges)
					kl_div_diff = kl_div_new - kl_div_prev
					print('KL Divergence Diff: {}'.format(kl_div_diff))
					print('Threshold: {}'.format(self.kl_threshold))
					if kl_div_diff < self.kl_threshold:
						print('Edge from {} to {} is bad'.format(edge.sender, edge.receiver))
						self.bad_edges.append(edge)
					else:
						print('Edge from {} to {} is good!'.format(edge.sender, edge.receiver))
					print()
				print('-'*20)
		print('Edges at start: {}'.format(len(self.G.get_all_edges())))
		print('Edges to remove: {}'.format(len(self.bad_edges)))
		for edge in self.bad_edges:
			self.H.remove_edge(edge)
		print('Edges at end: {}'.format(len(self.H.get_all_edges())))
		self.H.clean_dead_edges()
		good_edges = self.H.get_all_edges()
		print('Final edges: {}'.format(len(good_edges)))
		# return the good edges
		self.H.visualize()
		return good_edges		

	def patch_path(
		self,
		activation: Float[Tensor, "batch pos head_idx d_model"],
		hook: HookPoint,
		edges : List[Edge]
	):
		'''
		edges should look like tuples (src, rec)
		where src = (src_act_name, src_act_index_tuple)
		and rec = (rec_act_name, rec_act_index_tuple)
		'''
		for edge in edges:
			sender = edge.sender
			receiver = edge.receiver
			sender_hook_name = sender.name
			sender_hook_index = sender.index
			receiver_act_name = receiver.name
			receiver_act_index = receiver.index
			if hook.name == receiver_act_name:
				activation[receiver_act_index] += (self.corrupted_cache[sender_hook_name][sender_hook_index] - self.clean_cache[sender_hook_name][sender_hook_index])
		return activation

	@staticmethod
	def kl_divergence(patched_logits, orig_logits) -> float:
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

		# why am I getting negative values??? numerical instability??? TODO

		# Calculate the KL divergence manually
		kl_div = (q_probs * (torch.log(q_probs) - p_log_probs)).sum(dim=-1)

		# Return the average KL divergence over all batches
		return kl_div.mean().item()

	def ablate_paths(
		self,
		edges : List[Edge]
	) -> float:
		'''
		Performs path patching on edge from src to receiver heads and returns the KL divergence
		edges should look like tuples (src, rec)
		where src = (src_act_name, src_act_index_tuple)
		and rec = (rec_act_name, rec_act_index_tuple)
		'''

		senders = [edge.sender for edge in edges]
		sender_hook_names = [sender.name for sender in senders]
		sender_hook_names_filter = lambda name: name in sender_hook_names
		
		receivers = [edge.receiver for edge in edges]
		receiver_hook_names = [receiver.name for receiver in receivers]
		receiver_hook_names_filter = lambda name: name in receiver_hook_names

		model.reset_hooks()

		# ========== Step 2 ==========
		# Run on x_orig, with sender head patched from x_new, every other head frozen

		hook_fn = partial(
			self.patch_path,
			edges=edges
		)
			
		model.add_hook(receiver_hook_names_filter, hook_fn, level=1)
		patched_logits, _ = model.run_with_cache(
			self.clean_dataset.toks, 
			return_type='logits'
		)
		model.reset_hooks()
		# ========== Step 3 ==========
		# Calculate KL divergence between the patched logits and the original logits
		kl_div = self.kl_divergence(clean_logits, patched_logits)
		return kl_div
# %%


acdc = ACDC(
	model, 
	n_heads=n_heads,
	n_layers=n_layers,
	clean_cache=clean_cache,
	corrupted_cache=corrupted_cache,
	clean_dataset=ioi_dataset,
	corrupted_dataset=abc_dataset,
	clean_logits=clean_logits,
)

# good_edges = acdc.run()
# %%

results = torch.zeros((n_layers, n_heads))
# let's test on a single receiver
for rec in [(7, 3), (7, 9), (8, 6), (8, 10)]:
	for send in [(i, j) for i in range(7) for j in range(12)]:
		if send[0] < rec[0]:
			rec_q_name = utils.get_act_name("q_input", rec[0])
			rec_q_index = (slice(None), slice(None), rec[1], slice(None))
			rec_k_name = utils.get_act_name("k_input", rec[0])
			rec_k_index = (slice(None), slice(None), rec[1], slice(None))
			rec_v_name = utils.get_act_name("v_input", rec[0])
			rec_v_index = (slice(None), slice(None), rec[1], slice(None))
			send_res = utils.get_act_name("result", send[0])
			send_res_index = (slice(None), slice(None), send[1], slice(None))

			edges = [
				Edge(HookNode(send_res, send_res_index), HookNode(rec_q_name, rec_q_index)),
				Edge(HookNode(send_res, send_res_index), HookNode(rec_k_name, rec_k_index)),
				Edge(HookNode(send_res, send_res_index), HookNode(rec_v_name, rec_v_index))
			]
			for edge in edges:
				kl_div = acdc.ablate_paths([edge])
				print()
				print(edge)
				print(kl_div)
				results[send[0], send[1]] += kl_div
results /= (3 * 4)

# %%
imshow(
	100 * results,
	title="Direct effect on logit difference",
	labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
	coloraxis=dict(colorbar_ticksuffix = "%"),
	width=600,
)
# %%

# Now test on just the subset of 26 heads mentioned in the paper

IOI_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
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
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}

heads = list(itertools.chain(*IOI_CIRCUIT.values()))
acdc = ACDC(
	model, 
	n_heads=n_heads,
	n_layers=n_layers,
	clean_cache=clean_cache,
	corrupted_cache=corrupted_cache,
	clean_dataset=ioi_dataset,
	corrupted_dataset=abc_dataset,
	clean_logits=clean_logits,
	heads=heads
)
acdc.run()
# %%
