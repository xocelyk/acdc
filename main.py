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
corrupted_logits, corrupted_cache = model.run_with_cache(
	abc_dataset.toks, 
	return_type='logits'
)

orig_logits = clean_logits
patched_logits = corrupted_logits

last_word_indices = ioi_dataset.word_idx['end']
orig_logits = orig_logits[torch.arange(ioi_dataset.N), last_word_indices]
patched_logits = patched_logits[torch.arange(ioi_dataset.N), last_word_indices]
# p_log_probs = torch.nn.functional.log_softmax(orig_logits, dim=-1)
# q_log_probs = torch.nn.functional.log_softmax(patched_logits, dim=-1)
# q_probs = t.exp(q_log_probs)
# kl_div = (q_probs * (t.log(q_probs) - p_log_probs)).sum(dim=-1)
# return kl_div.mean()

orig_logprobs = torch.nn.functional.log_softmax(orig_logits, dim=-1)
orig_probs = torch.exp(orig_logprobs)
patched_logprobs = torch.nn.functional.log_softmax(patched_logits, dim=-1)
patched_probs = torch.exp(patched_logprobs)

# debugging
# 1. Get the top 5 predicted tokens in orig_logits for each batch
_, top5_orig_indices = torch.topk(orig_logits, 5, dim=-1)
print('Top 5 original tokens')
for i in range(5):
	print(model.to_str_tokens(top5_orig_indices[:, i].squeeze()))

# 2. Get the top 5 predicted tokens in patched_logits for each batch
_, top5_patched_indices = torch.topk(patched_logits, 5, dim=-1)
print('Top 5 patched tokens')
for i in range(5):
	print(model.to_str_tokens(top5_patched_indices[:, i].squeeze()))

# 3. Take the top token in orig_logits and find the probability in patched_logits for each batch
top1_orig_indices = torch.argmax(orig_logits, dim=-1, keepdim=True)

top1_orig_probs = torch.gather(orig_probs, -1, top1_orig_indices)
top1_patched_probs = torch.gather(patched_probs, -1, top1_orig_indices)
print('Top 1 original token probs in orig logits')
print(model.to_str_tokens(top1_orig_indices))
print(top1_orig_probs)
print()
print('Top 1 original token probs in patched logits')
print(model.to_str_tokens(top1_orig_indices))
print(top1_patched_probs)
print()

# %%

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

# debug the cache
# 
# %%
print(clean_logits.shape)
print(ioi_dataset.sentences)
last_word_indices = ioi_dataset.word_idx['end']
last_token_logits = clean_logits[torch.arange(N), last_word_indices]
print(last_token_logits.shape)
# print(model.to_str_tokens(end_words))
# get the top 5 predicted tokens for each sentence
_, top5_indices = torch.topk(last_token_logits, 5, dim=-1)
# get the top 5 predicted tokens for each sentence
print('Top 5 predicted tokens for each sentence')
for i in range(N):
	print(model.to_str_tokens(top5_indices[i].squeeze()))


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
			# assert not self.check_for_cycle()
			self.graph = self.topological_sort()
			print(self.graph.keys())

	def is_cyclic_util(self, v, visited, rec_stack):
		print()
		print('Visiting: {}'.format(v))
		visited.add(v)
		rec_stack.add(v)
		for neighbor in self.graph[v]:
			# print(len(self.graph))
			if neighbor not in visited:
				print('Neighbor: {}'.format(neighbor))
				# print(len(self.graph))
				if self.is_cyclic_util(neighbor, visited, rec_stack):
					return True
			elif neighbor in rec_stack:
				print('Node: {}'.format(v))
				print('Bad neighbor: {}'.format(neighbor))
				return True
		rec_stack.remove(v)
		return False

	def check_for_cycle(self):
		
		# return True if there is a cycle in the graph
		# else return False
		visited = set()
		rec_stack = set()
		for node in self.graph:
			print('Node: {}'.format(node))
			if node not in visited:
				if self.is_cyclic_util(node, visited, rec_stack):
					return True
		return False
		
	def get_head_hook_node(self, layer, head, hook_type, dir):
		if hook_type == 'result':
			name = utils.get_act_name("result", layer)
			idx = (slice(None), slice(None), head, slice(None))
		elif hook_type in ['q_input', 'k_input', 'v_input']:
			name = f'blocks.{layer}.hook_{hook_type}'
			idx = (slice(None), slice(None), head, slice(None))
		else:
			raise ValueError('Invalid hook type')
		return HookNode(name, idx, dir)
	
	def get_mlp_hook_node(self, layer, hook_type, dir):
		if hook_type in ['mlp_in', 'mlp_out']:
			name = f'blocks.{layer}.hook_{hook_type}'
			idx = (slice(None), slice(None), slice(None))
		else:
			raise ValueError('Invalid hook type')
		return HookNode(name, idx, dir)

	def build_ioi_computational_graph(self):
		# TODO: add MLPs?
		g = defaultdict(list) # receiver -> senders

		heads = self.heads

		# receiver heads
		for rec in heads:
			rec_layer, rec_head = rec
			rec_q_input = self.get_head_hook_node(rec_layer, rec_head, 'q_input', 'IN')
			rec_k_input = self.get_head_hook_node(rec_layer, rec_head, 'k_input', 'IN')
			rec_v_input = self.get_head_hook_node(rec_layer, rec_head, 'v_input', 'IN')

			# first get all of the sender heads
			for send in heads:
				send_layer, send_head = send
				if send_layer < rec_layer:
					send_result = self.get_head_hook_node(send_layer, send_head, 'result', 'OUT')
					g[rec_q_input].append(send_result)
					g[rec_k_input].append(send_result)
					g[rec_v_input].append(send_result)
			
			# sender MLPs
			for layer in range(self.n_layers):
				if layer < rec_layer:
					mlp_out = self.get_mlp_hook_node(layer, 'mlp_out', 'OUT')
					g[rec_q_input].append(mlp_out)
					g[rec_k_input].append(mlp_out)
					g[rec_v_input].append(mlp_out)
			
			# sender resid pre
			resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
			resid_pre_hook_index = (slice(None), slice(None), slice(None))
			resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, dir='OUT')
			g[rec_q_input].append(resid_pre)
			g[rec_k_input].append(resid_pre)
			g[rec_v_input].append(resid_pre)

		# receiver MLPs
		for layer in range(self.n_layers):
			mlp_in = self.get_mlp_hook_node(layer, 'mlp_in', 'IN')

			# sender heads
			for send in heads:
				send_layer, send_head = send
				if send_layer <= layer:
					send_result = self.get_head_hook_node(send_layer, send_head, 'result', 'OUT')
					g[mlp_in].append(send_result)
			

			# sender MLPs
			for send_layer in range(self.n_layers):
				if send_layer < layer:
					mlp_out = self.get_mlp_hook_node(send_layer, 'mlp_out', 'OUT')
					g[mlp_in].append(mlp_out)
			
			# sender resid pre
			resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
			resid_pre_hook_index = (slice(None), slice(None), slice(None))
			resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, dir='OUT')
			g[mlp_in].append(resid_pre)

		# receiver resid post
		resid_post_hook_name = utils.get_act_name("resid_post", self.n_layers - 1)
		resid_post_hook_index = (slice(None), slice(None), slice(None))
		resid_post = HookNode(resid_post_hook_name, resid_post_hook_index, dir='IN')

		# sender heads
		for send in heads:
			send_layer, send_head = send
			send_result = self.get_head_hook_node(send_layer, send_head, 'result', 'OUT')
			g[resid_post].append(send_result)

		# sender MLPs
		for send_layer in range(self.n_layers):
			mlp_out = self.get_mlp_hook_node(send_layer, 'mlp_out', 'OUT')
			g[resid_post].append(mlp_out)

		# sender resid pre
		resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
		resid_pre_hook_index = (slice(None), slice(None), slice(None))
		resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, dir='OUT')
		g[resid_post].append(resid_pre)

		# create direct computation edges between head inputs and outputs
		for head in heads:
			head_layer, head_head = head
			head_out = self.get_head_hook_node(head_layer, head_head, 'result', 'OUT')
			head_q_input = self.get_head_hook_node(head_layer, head_head, 'q_input', 'IN')
			head_k_input = self.get_head_hook_node(head_layer, head_head, 'k_input', 'IN')
			head_v_input = self.get_head_hook_node(head_layer, head_head, 'v_input', 'IN')
			g[head_out].append(head_q_input)
			g[head_out].append(head_k_input)
			g[head_out].append(head_v_input)

		# turn g into regular dict
		g = {k: v for k, v in g.items()}
		# now go through all of the nodes and add the ones that are not in the graph
		new_g = {}
		for k, v in g.items():
			for sender in v:
				if sender not in g:
					new_g[sender] = []
				else:
					new_g[sender] = g[sender]
		return new_g





		# # create heads subgraph
		# heads = self.heads
		# for send in heads:
		# 	for rec in heads:
		# 		if send[0] < rec[0]:
		# 			send_layer, send_head = send
		# 			rec_layer, rec_head = rec
		# 			send_hook_name = utils.get_act_name("result", send_layer)
		# 			send_hook_index = (slice(None), slice(None), send_head, slice(None))
		# 			rec_hook_name_q = utils.get_act_name("q_input", rec_layer)
		# 			rec_hook_name_k = utils.get_act_name("k_input", rec_layer)
		# 			rec_hook_name_v = utils.get_act_name("v_input", rec_layer)
		# 			rec_hook_index = (slice(None), slice(None), rec_head, slice(None))
		# 			sender = HookNode(send_hook_name, send_hook_index, dir='OUT')
		# 			receiver_q = HookNode(rec_hook_name_q, rec_hook_index, dir='IN')
		# 			receiver_k = HookNode(rec_hook_name_k, rec_hook_index, dir='IN')
		# 			receiver_v = HookNode(rec_hook_name_v, rec_hook_index, dir='IN')
		# 			g[receiver_q].append(sender)
		# 			g[receiver_k].append(sender)
		# 			g[receiver_v].append(sender)

		# # add MLPs
		# mlps_in = []
		# mlps_out = []
		# for layer in range(self.n_layers):
		# 	mlp_in_hook_name = f'blocks.{layer}.hook_mlp_in'
		# 	mlp_in_hook_index = (slice(None), slice(None), slice(None))
		# 	mlp_in = HookNode(mlp_in_hook_name, mlp_in_hook_index, dir='IN')
		# 	mlps_in.append(mlp_in)

		# 	mlp_out_hook_name = f'blocks.{layer}.hook_mlp_out'
		# 	mlp_out_hook_index = (slice(None), slice(None), slice(None))
		# 	mlp_out = HookNode(mlp_out_hook_name, mlp_out_hook_index, dir='OUT')
		# 	mlps_out.append(mlp_out)


		# for mlp_out_layer in range(self.n_layers - 1):
		# 	mlp_out_hook_name = utils.get_act_name("mlp_out", mlp_out_layer)
		# 	mlp_out_hook_index = (slice(None), slice(None), slice(None))
		# 	mlp_out = HookNode(mlp_out_hook_name, mlp_out_hook_index, dir='OUT')
		# 	# connect mlp out to all heads in downstream layers
		# 	for rec in heads:
		# 		rec_layer, rec_head = rec
		# 		if rec_layer > mlp_out_layer:
		# 			rec_hook_name_q = utils.get_act_name("q_input", rec_layer)
		# 			rec_hook_name_k = utils.get_act_name("k_input", rec_layer)
		# 			rec_hook_name_v = utils.get_act_name("v_input", rec_layer)
		# 			rec_hook_index = (slice(None), slice(None), rec_head, slice(None))
		# 			receiver_q = HookNode(rec_hook_name_q, rec_hook_index, dir='IN')
		# 			receiver_k = HookNode(rec_hook_name_k, rec_hook_index, dir='IN')
		# 			receiver_v = HookNode(rec_hook_name_v, rec_hook_index, dir='IN')
		# 			g[receiver_q].append(mlp_out)
		# 			g[receiver_k].append(mlp_out)
		# 			g[receiver_v].append(mlp_out)

		
		# for mlp_in_layer in range(1, self.n_layers):
		# 	for send in heads:
		# 		send_layer, send_head = send
		# 		if send_layer <= mlp_in_layer:
		# 			mlp_in_hook_name = utils.get_act_name("mlp_in", mlp_in_layer)
		# 			mlp_in_hook_index = (slice(None), slice(None), slice(None))
		# 			mlp_in = HookNode(mlp_in_hook_name, mlp_in_hook_index, dir='IN')
		# 			send_hook_name = utils.get_act_name("result", send_layer)
		# 			send_hook_index = (slice(None), slice(None), send_head, slice(None))
		# 			sender = HookNode(send_hook_name, send_hook_index, dir='OUT')
		# 			print()
		# 			print(mlp_in)
		# 			print(sender)
		# 			g[mlp_in].append(sender)

		# # add resid pre and resid post
		# resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
		# resid_pre_hook_index = (slice(None), slice(None), slice(None))
		# resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, dir='OUT')
		# g[resid_pre] = []
		# for rec in heads:
		# 	rec_layer, rec_head = rec
		# 	rec_hook_name_q = utils.get_act_name("q_input", rec_layer)
		# 	rec_hook_name_k = utils.get_act_name("k_input", rec_layer)
		# 	rec_hook_name_v = utils.get_act_name("v_input", rec_layer)
		# 	rec_hook_index = (slice(None), slice(None), rec_head, slice(None))
		# 	receiver_q = HookNode(rec_hook_name_q, rec_hook_index, dir='IN')
		# 	receiver_k = HookNode(rec_hook_name_k, rec_hook_index, dir='IN')
		# 	receiver_v = HookNode(rec_hook_name_v, rec_hook_index, dir='IN')
		# 	g[receiver_q].append(resid_pre)
		# 	g[receiver_k].append(resid_pre)
		# 	g[receiver_v].append(resid_pre)
		
		# for layer in range(self.n_layers):
		# 	mlp_in_hook_name = utils.get_act_name("mlp_in", layer)
		# 	mlp_in_hook_index = (slice(None), slice(None), slice(None))
		# 	mlp_in = HookNode(mlp_in_hook_name, mlp_in_hook_index, dir='IN')
		# 	g[mlp_in].append(resid_pre)

		# for rec_layer in range(1, self.n_layers):
		# 	mlp_in_hook_name = utils.get_act_name("mlp_in", rec_layer)
		# 	mlp_in_hook_index = (slice(None), slice(None), slice(None))
		# 	mlp_in = HookNode(mlp_in_hook_name, mlp_in_hook_index, dir='IN')			
		# 	for send_layer in range(rec_layer):
		# 		mlp_out_hook_name = utils.get_act_name("mlp_out", send_layer)
		# 		mlp_out_hook_index = (slice(None), slice(None), slice(None))
		# 		mlp_out = HookNode(mlp_out_hook_name, mlp_out_hook_index, dir='OUT')
		# 		g[mlp_in].append(mlp_out)

		# resid_post_hook_name = utils.get_act_name("resid_post", self.n_layers - 1)
		# resid_post_hook_index = (slice(None), slice(None), slice(None))
		# resid_post = HookNode(resid_post_hook_name, resid_post_hook_index, dir='IN')
		# g[resid_post].append(resid_pre)
		# for head in heads:
		# 	head_layer, head_head = head
		# 	head_hook_name = utils.get_act_name("result", head_layer)
		# 	head_hook_index = (slice(None), slice(None), head_head, slice(None))
		# 	head = HookNode(head_hook_name, head_hook_index, dir='OUT')
		# 	g[resid_post].append(head)

		# for i in range(self.n_layers):
		# 	mlp_out_hook_name = utils.get_act_name("mlp_out", i)
		# 	mlp_out_hook_index = (slice(None), slice(None), slice(None))
		# 	mlp_out = HookNode(mlp_out_hook_name, mlp_out_hook_index, dir='OUT')
		# 	g[resid_post].append(mlp_out)

		# # now create direct computation edges between head inputs and outputs
		# for head in heads:
		# 	head_layer, head_head = head
		# 	head_out_hook_name = utils.get_act_name("result", head_layer)
		# 	head_out_hook_index = (slice(None), slice(None), head_head, slice(None))
		# 	head_out = HookNode(head_out_hook_name, head_out_hook_index, dir='OUT')
		# 	head_q_input_hook_name = utils.get_act_name("q_input", head_layer)
		# 	head_q_input_hook_index = (slice(None), slice(None), head_head, slice(None))
		# 	head_q_input = HookNode(head_q_input_hook_name, head_q_input_hook_index, dir='IN')
		# 	head_k_input_hook_name = utils.get_act_name("k_input", head_layer)
		# 	head_k_input_hook_index = (slice(None), slice(None), head_head, slice(None))
		# 	head_k_input = HookNode(head_k_input_hook_name, head_k_input_hook_index, dir='IN')
		# 	head_v_input_hook_name = utils.get_act_name("v_input", head_layer)
		# 	head_v_input_hook_index = (slice(None), slice(None), head_head, slice(None))
		# 	head_v_input = HookNode(head_v_input_hook_name, head_v_input_hook_index, dir='IN')
		# 	g[head_out].append(head_q_input)
		# 	g[head_out].append(head_k_input)
		# 	g[head_out].append(head_v_input)


		# out_filename = 'g.txt'
		# # write out graph
		# for k, v in g.items():
		# 	with open(out_filename, 'a') as f:
		# 		f.write('\n')
		# 	for sender in v:
		# 		sender_name = sender.__repr__()
		# 		key_name = k.__repr__()
		# 		sender_name = sender_name.split(' ')[0]
		# 		key_name = key_name.split(' ')[0]
		# 		with open(out_filename, 'a') as f:
		# 			f.write(f'{sender_name} -> {key_name}\n')
	
		# # turn g into regular dict
		# g = {k: v for k, v in g.items()}

		# #hack to make sure all of the nodes are in the graph
		# new_g = {}
		# for k, v in g.items():
		# 	for sender in v:
		# 		if sender not in g:
		# 			new_g[sender] = []
		# 		else:
		# 			new_g[sender] = g[sender]
		# return new_g

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
			self.heads = sorted(heads, key=lambda x: (x[0], x[1]))
		self.clean_dataset = clean_dataset
		self.corrupted_dataset: IOIDataset = corrupted_dataset
		self.clean_logits: Optional[Float[Tensor, "batch seq d_vocab"]] = clean_logits
		self.G = ComputationalGraph(n_heads, n_layers, self.heads)
		self.H = ComputationalGraph(n_heads, n_layers, self.heads)
		self.kl_threshold = .0575
		# self.kl_threshold = 1e-4
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
				# print difference between clean and corrupted cache
				# act_diff = (self.corrupted_cache[receiver_act_name][receiver_act_index] - self.clean_cache[receiver_act_name][receiver_act_index]).mean()
				# print('Activation difference: {}'.format(act_diff))
				activation[receiver_act_index] += (self.corrupted_cache[sender_hook_name][sender_hook_index] - self.clean_cache[sender_hook_name][sender_hook_index])
		return activation

	def kl_divergence(self, patched_logits, orig_logits) -> float:
		'''
		Returns the KL divergence between two probability distributions averaged over batches.
		'''
		# test the ave logits diff
		logit_diff = logits_to_ave_logit_diff_2(patched_logits, self.clean_dataset, per_prompt=False)
		# print('Logit diff: {}'.format(logit_diff))

		# Select the last logits from each sequence for comparison
		last_word_indices = self.clean_dataset.word_idx['end']
		orig_logits = orig_logits[torch.arange(self.clean_dataset.N), last_word_indices]
		patched_logits = patched_logits[torch.arange(self.clean_dataset.N), last_word_indices]
		# p_log_probs = torch.nn.functional.log_softmax(orig_logits, dim=-1)
		# q_log_probs = torch.nn.functional.log_softmax(patched_logits, dim=-1)
		# q_probs = t.exp(q_log_probs)
		# kl_div = (q_probs * (t.log(q_probs) - p_log_probs)).sum(dim=-1)
		# return kl_div.mean()

		orig_logprobs = torch.nn.functional.log_softmax(orig_logits, dim=-1)
		orig_probs = torch.exp(orig_logprobs)
		patched_logprobs = torch.nn.functional.log_softmax(patched_logits, dim=-1)
		patched_probs = torch.exp(patched_logprobs)

		# debugging
		# 1. Get the top 5 predicted tokens in orig_logits for each batch
		# _, top5_orig_indices = torch.topk(orig_logits, 5, dim=-1)
		# print('Top 5 original tokens')
		# for i in range(5):
		# 	print(model.to_str_tokens(top5_orig_indices[:, i].squeeze()))
		
		# # 2. Get the top 5 predicted tokens in patched_logits for each batch
		# _, top5_patched_indices = torch.topk(patched_logits, 5, dim=-1)
		# print('Top 5 patched tokens')
		# for i in range(5):
		# 	print(model.to_str_tokens(top5_patched_indices[:, i].squeeze()))
		
		# # 3. Take the top token in orig_logits and find the probability in patched_logits for each batch
		# top1_orig_indices = torch.argmax(orig_logits, dim=-1, keepdim=True)
	
		# top1_orig_probs = torch.gather(orig_probs, -1, top1_orig_indices)
		# top1_patched_probs = torch.gather(patched_probs, -1, top1_orig_indices)
		# print('Top 1 original token probs in orig logits')
		# print(model.to_str_tokens(top1_orig_indices))
		# print(top1_orig_probs)
		# print()
		# print('Top 1 original token probs in patched logits')
		# print(model.to_str_tokens(top1_orig_indices))
		# print(top1_patched_probs)
		# print()


		# Calculate the KL divergence manually
		kl_div = (patched_probs * (torch.log(patched_probs) - orig_logprobs)).sum(dim=-1)

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


# acdc = ACDC(
# 	model, 
# 	n_heads=n_heads,
# 	n_layers=n_layers,
# 	clean_cache=clean_cache,
# 	corrupted_cache=corrupted_cache,
# 	clean_dataset=ioi_dataset,
# 	corrupted_dataset=abc_dataset,
# 	clean_logits=clean_logits,
# )

# good_edges = acdc.run()
# %%
# acdc = ACDC(
# 	model,
# 	n_heads=n_heads,
# 	n_layers=n_layers,
# 	clean_cache=clean_cache,
# 	corrupted_cache=corrupted_cache,
# 	clean_dataset=ioi_dataset,
# 	corrupted_dataset=abc_dataset,
# 	clean_logits=clean_logits,
# )

# results = torch.zeros((n_layers, n_heads))
# # let's test on a few receivers
# for rec in [(7, 3), (7, 9), (8, 6), (8, 10)]:
# 	for send in [(i, j) for i in range(7) for j in range(12)]:
# 		if send[0] < rec[0]:
# 			rec_q_name = utils.get_act_name("q_input", rec[0])
# 			rec_q_index = (slice(None), slice(None), rec[1], slice(None))
# 			rec_k_name = utils.get_act_name("k_input", rec[0])
# 			rec_k_index = (slice(None), slice(None), rec[1], slice(None))
# 			rec_v_name = utils.get_act_name("v_input", rec[0])
# 			rec_v_index = (slice(None), slice(None), rec[1], slice(None))
# 			send_res = utils.get_act_name("result", send[0])
# 			send_res_index = (slice(None), slice(None), send[1], slice(None))

# 			edges = [
# 				Edge(HookNode(send_res, send_res_index), HookNode(rec_q_name, rec_q_index)),
# 				Edge(HookNode(send_res, send_res_index), HookNode(rec_k_name, rec_k_index)),
# 				Edge(HookNode(send_res, send_res_index), HookNode(rec_v_name, rec_v_index))
# 			]
# 			for edge in edges:
# 				kl_div = acdc.ablate_paths([edge])
# 				print()
# 				print(edge)
# 				print(kl_div)
# 				results[send[0], send[1]] += kl_div
# results /= (3 * 4)

# %%
# imshow(
# 	100 * results,
# 	title="Direct effect on logit difference",
# 	labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
# 	coloraxis=dict(colorbar_ticksuffix = "%"),
# 	width=600,
# )
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
