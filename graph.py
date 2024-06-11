from collections import OrderedDict, defaultdict
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from transformer_lens import utils
from graphviz import Digraph

class HookNode:
	'''Represents a node in the computational graph'''
	def __init__(self, name, index, direction='IN'):
		'''
		Initialize a HookNode.

		:param name: Name of the node.
		:param index: Index of the node. Tuple of shape (batch, pos, head, d_model) or (batch, pos, head) used to index into activations.
		:param direction: Direction of the computation flowing through the node ('IN' or 'OUT').
		'''

		self.name = name
		self.index = index
		self.direction = direction

	def __repr__(self):
		'''Return a string representation of the node.'''
		idx_val = '' if ('resid' in self.name or 'mlp' in self.name) else f'head {self.index[2]}'
		return f'{self.name} {idx_val}'
	
	def __eq__(self, other):
		'''Check if two nodes are equal.'''
		return self.name == other.name and self.index == other.index
	
	def __hash__(self):
		'''Generate hash value based on the name and index of the node.'''
		idx_val = sum(val * 10 ** i for i, val in enumerate(self.index) if isinstance(val, int))
		return hash((self.name, idx_val))

class Edge:
	'''Represents an edge in the graph, connecting two nodes.'''
	def __init__(self, sender, receiver, mode='ADD'):
		'''
		Initialize an Edge.

		:param sender: Node sending the computation.
		:param receiver: Node receiving the computation.
		:param mode: Mode of the edge ('ADD' or 'DIRECT'). ADD edges represent live computation, while DIRECT edges represent direct computation.
		'''
		self.sender = sender
		self.receiver = receiver
		self.mode = mode

	def __repr__(self):
		'''Return a string representation of the edge.'''
		return f"{self.sender} -> {self.receiver}"

	def __eq__(self, other):
		'''Check if two edges are equal.'''
		return self.sender == other.sender and self.receiver == other.receiver

	def __hash__(self):
		'''Generate hash value based on the sender and receiver nodes of the edge.'''
		return hash((self.sender, self.receiver))

class ComputationalGraph:
	def __init__(self, n_heads, n_layers, heads, empty=False):
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.heads = heads
		if empty:
			self.graph = OrderedDict()
		else:
			self.graph = self.build_ioi_computational_graph()
			assert not self.check_for_cycle(), 'The computational graph contains a cycle.'
			self.graph = self.topological_sort()
			assert any('resid_post' in k.name for k in self.graph.keys()), "No 'resid_post' node found in the graph."

	def is_cyclic_util(self, v, visited, rec_stack):
		'''Recursive utility function to check for cycles in the graph.'''
		visited.add(v)
		rec_stack.add(v)
		for neighbor in self.graph[v]:
			if neighbor not in visited:
				if self.is_cyclic_util(neighbor, visited, rec_stack):
					return True
			elif neighbor in rec_stack:
				return True
		rec_stack.remove(v)
		return False

	def check_for_cycle(self):
		'''Check if the graph contains a cycle.'''
		visited = set()
		rec_stack = set()
		for node in self.graph:
			if node not in visited:
				if self.is_cyclic_util(node, visited, rec_stack):
					return True
		return False

	def get_head_hook_node(self, layer, head, hook_type, direction):
		'''Get a head hook node based on the layer, head, hook type, and direction.'''
		if hook_type == 'result':
			name = utils.get_act_name("result", layer)
			# Hook indices are (batch, pos, head, d_model) so that they can be used to index into activations
			idx = (slice(None), slice(None), head, slice(None))
		elif hook_type in ['q_input', 'k_input', 'v_input']:
			name = f'blocks.{layer}.hook_{hook_type}'
			idx = (slice(None), slice(None), head, slice(None))
		else:
			raise ValueError(f"Invalid hook type: {hook_type}")
		return HookNode(name, idx, direction)

	def get_mlp_hook_node(self, layer, hook_type, direction):
		'''Get an MLP hook node based on the layer, hook type, and direction.'''
		if hook_type in ['mlp_in', 'mlp_out']:
			name = f'blocks.{layer}.hook_{hook_type}'
			idx = (slice(None), slice(None), slice(None))
		else:
			raise ValueError(f"Invalid hook type: {hook_type}")
		return HookNode(name, idx, direction)

	
	def _add_receiver_head_edges(self, graph):
		'''Add edges for receiver heads in the computational graph.'''
		for receiver_layer, receiver_head in self.heads:
			rec_q_input, rec_k_input, rec_v_input = self._get_receiver_head_input_nodes(receiver_layer, receiver_head)

			# Connect sender attention result nodes to each receiver input node
			self._connect_sender_heads_to_receiver(graph, receiver_layer, rec_q_input, rec_k_input, rec_v_input)

			# Connect sender MLP output nodes to each receiver input node
			self._connect_sender_mlps_to_receiver(graph, receiver_layer, rec_q_input, rec_k_input, rec_v_input)

			# Connect resid_pre node to each receiver input node
			self._connect_resid_pre_to_receiver(graph, rec_q_input, rec_k_input, rec_v_input)


	def _get_receiver_head_input_nodes(self, receiver_layer, receiver_head):
		'''Get the input nodes for a receiver head.'''
		rec_q_input = self.get_head_hook_node(receiver_layer, receiver_head, 'q_input', 'IN')
		rec_k_input = self.get_head_hook_node(receiver_layer, receiver_head, 'k_input', 'IN')
		rec_v_input = self.get_head_hook_node(receiver_layer, receiver_head, 'v_input', 'IN')
		return rec_q_input, rec_k_input, rec_v_input
	
	def _connect_sender_heads_to_receiver(self, graph, receiver_layer, rec_q_input, rec_k_input, rec_v_input):
		for send_head_layer, send_head_index in self.heads:
			# Connect sender attention result hooks to each receiver input hook
			if send_head_layer < receiver_layer:
				send_result = self.get_head_hook_node(send_head_layer, send_head_index, 'result', 'OUT')
				graph[rec_q_input].append(send_result)
				graph[rec_k_input].append(send_result)
				graph[rec_v_input].append(send_result)

	def _connect_sender_mlps_to_receiver(self, graph, receiver_layer, rec_q_input, rec_k_input, rec_v_input):	
		for layer in range(self.n_layers):
			# Only add edges from lower layers to higher layers
			if layer < receiver_layer:
				mlp_out = self.get_mlp_hook_node(layer, 'mlp_out', 'OUT')
				graph[rec_q_input].append(mlp_out)
				graph[rec_k_input].append(mlp_out)
				graph[rec_v_input].append(mlp_out)
		
	def _connect_resid_pre_to_receiver(self, graph, rec_q_input, rec_k_input, rec_v_input):
		resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
		resid_pre_hook_index = (slice(None), slice(None), slice(None))
		resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, direction='OUT')
		graph[rec_q_input].append(resid_pre)
		graph[rec_k_input].append(resid_pre)
		graph[rec_v_input].append(resid_pre)

	def _add_receiver_mlp_edges(self, graph):
		for layer in range(self.n_layers):
			mlp_in_node = self.get_mlp_hook_node(layer, 'mlp_in', 'IN')

			# Connect sender head result nodes to receiver MLP input node
			self._connect_sender_heads_to_mlp(graph, layer, mlp_in_node)

			# Connect sender MLP output nodes to receiver MLP input node
			self._connect_sender_mlps_to_mlp(graph, layer, mlp_in_node)

			# Connect resid_pre node to receiver MLP input node
			self._connect_resid_pre_to_mlp(graph, mlp_in_node)

	def _connect_sender_heads_to_mlp(self, graph, layer, mlp_in_node):
		for send_head_layer, send_head_index in self.heads:
			# MLPs are connected to all heads from lower or equal layers (MLP follows heads in computation order)
			if send_head_layer <= layer:
				send_result = self.get_head_hook_node(send_head_layer, send_head_index, 'result', 'OUT')
				graph[mlp_in_node].append(send_result)

	def _connect_sender_mlps_to_mlp(self, graph, layer, mlp_in_node):
		for send_head_layer in range(self.n_layers):
			# Only add edges from lower layer to higher layer MLPs
			if send_head_layer < layer:
				mlp_out = self.get_mlp_hook_node(send_head_layer, 'mlp_out', 'OUT')
				graph[mlp_in_node].append(mlp_out)

	def _connect_resid_pre_to_mlp(self, graph, mlp_in_node):
		resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
		resid_pre_hook_index = (slice(None), slice(None), slice(None))
		resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, direction='OUT')
		graph[mlp_in_node].append(resid_pre)

	def _add_receiver_resid_post_edges(self, graph):
		'''Add edges for receiver resid_post in the computational graph.'''
		resid_post_node = self._get_resid_post_node()

		# Connect sender head result nodes to receiver resid_post node
		self._connect_sender_heads_to_resid_post(graph, resid_post_node)

		# Connect sender MLP output nodes to receiver resid_post node
		self._connect_sender_mlps_to_resid_post(graph, resid_post_node)

		# Connect resid_pre node to receiver resid_post node
		self._connect_resid_pre_to_resid_post(graph, resid_post_node)

	def _get_resid_post_node(self):
		resid_post_hook_name = utils.get_act_name("resid_post", self.n_layers - 1)
		resid_post_hook_index = (slice(None), slice(None), slice(None))
		resid_post_node = HookNode(resid_post_hook_name, resid_post_hook_index, direction='IN')
		return resid_post_node
	
	def _connect_sender_heads_to_resid_post(self, graph, resid_post_node):
		for send_head_layer, send_head_index in self.heads:
			send_result = self.get_head_hook_node(send_head_layer, send_head_index, 'result', 'OUT')
			graph[resid_post_node].append(send_result)

	def _connect_sender_mlps_to_resid_post(self, graph, resid_post_node):
		for send_head_layer in range(self.n_layers):
			mlp_out = self.get_mlp_hook_node(send_head_layer, 'mlp_out', 'OUT')
			graph[resid_post_node].append(mlp_out)
	
	def _connect_resid_pre_to_resid_post(self, graph, resid_post_node):
		resid_pre_hook_name = utils.get_act_name("resid_pre", 0)
		resid_pre_hook_index = (slice(None), slice(None), slice(None))
		resid_pre = HookNode(resid_pre_hook_name, resid_pre_hook_index, direction='OUT')
		graph[resid_post_node].append(resid_pre)

	def _create_direct_computation_edges(self, graph):
		for head_layer, head_index in self.heads:
			head_out = self.get_head_hook_node(head_layer, head_index, 'result', 'OUT')
			head_q_input = self.get_head_hook_node(head_layer, head_index, 'q_input', 'IN')
			head_k_input = self.get_head_hook_node(head_layer, head_index, 'k_input', 'IN')
			head_v_input = self.get_head_hook_node(head_layer, head_index, 'v_input', 'IN')
			graph[head_out].append(head_q_input)
			graph[head_out].append(head_k_input)
			graph[head_out].append(head_v_input)
		
		for layer in range(self.n_layers):
			mlp_out = self.get_mlp_hook_node(layer, 'mlp_out', 'OUT')
			mlp_in = self.get_mlp_hook_node(layer, 'mlp_in', 'IN')
			graph[mlp_out].append(mlp_in)

	def _add_empty_sender_lists(self, graph):
		graph = {k: v for k, v in graph.items()}
		result_graph = {}
		for k, v in graph.items():
			result_graph[k] = v
			for sender in v:
				if sender not in graph.keys():
					result_graph[sender] = []
				else:
					result_graph[sender] = graph[sender]
		return result_graph

	def build_ioi_computational_graph(self):
		graph = defaultdict(list) # receiver -> senders

		# Add edges for receiver heads
		self._add_receiver_head_edges(graph)

		# Add edges for receiver MLPs
		self._add_receiver_mlp_edges(graph)

		# Add edges for receiver resid_post
		self._add_receiver_resid_post_edges(graph)

		# Create direct computation edges from q, k, v head inputs and head outputs in same layer
		self._create_direct_computation_edges(graph)

		# STEP 5: Add empty sender lists for nodes that only receive computation
		final_graph = self._add_empty_sender_lists(graph)
		return final_graph

	def topological_sort(self):
		'''
		Perform a topological sort on the computational graph.
		It is important that we traverse the graph in the order of computation.
		'''
		visited = set()
		stack = []

		def dfs(node):
			if node not in visited:
				visited.add(node)
				for neighbor in self.graph[node]:
					dfs(neighbor)
				stack.append(node)

		for node in self.graph:
			dfs(node)

		return OrderedDict((k, self.graph[k]) for k in reversed(stack))

	def get_incoming_edges(self, node):
		'''Get the incoming edges for a given node.'''
		edges = []
		for rec, senders in self.graph.items():
			if node == rec:
				for sender in senders:
					direct_computation_edge = rec.direction == 'OUT' and sender.direction == 'IN'
					if not direct_computation_edge:
						edges.append(Edge(sender, rec, mode='ADD'))
		# Order of edges does not matter, but usually the later ones (closer to the receiver) are more important
		return edges[::-1]

	def remove_edge(self, edge):
		'''Remove an edge from the computational graph.'''
		self.graph[edge.receiver].remove(edge.sender)

	def add_edge(self, edge):
		'''Add an edge to the computational graph.'''
		if edge.receiver not in self.graph:
			self.graph[edge.receiver] = []
		self.graph[edge.receiver].append(edge.sender)

	def clean_dead_edges(self):
		'''
		Clean up dead edges from the computational graph.

		We have two types of dead edges:
		1. Receiver nodes that only have direct computation edges
		2. Receiver nodes that have no senders (disconnect from the graph)
		'''

		result_graph = self.graph.copy()

		# Step 1. Remove receiver nodes that only have direct computation edges.
		for rec, senders in self.graph.items():
			if rec.direction == 'OUT' and all(sender.direction == 'IN' for sender in senders):
				del result_graph[rec]

		self.graph = result_graph.copy()
		# Step 2. Remove receiver nodes that have no senders.
		for rec, senders in self.graph.items():
			if len(senders) == 0:
				del result_graph[rec]

		self.graph = result_graph

	# def merge_mlps(self):
	# 	'''Merge MLP nodes in the computational graph.'''
	# 	merged_graph = {}
	# 	for receiver, senders in self.graph.items():
	# 		new_receiver_name = self.rename_node_fn(receiver)
	# 		new_receiver_node = HookNode(new_receiver_name, receiver.index, receiver.direction)
	# 		if new_receiver_node not in merged_graph:
	# 			merged_graph[new_receiver_node] = []
	# 		merged_graph[HookNode(new_receiver_name, receiver.index, receiver.direction)].extend([HookNode(self.rename_node_fn(sender), sender.index, sender.direction) for sender in senders])

	# 	# check that self loops are not added
	# 	for node, senders in merged_graph.items():
	# 		if node in senders:
	# 			senders.remove(node)
	# 	self.graph = merged_graph

	def rename_node_fn(self, x):
		print(repr(x))
		if x.name.startswith('h') or x.name.startswith('m'):
			return x.name
		if 'head' in repr(x):
			return 'h' + str(repr(x).split('.')[1]) + '.' + str(repr(x).split(' ')[-1])
		elif 'mlp' in x.name:
			return 'm' + str(x.name.split('.')[1])
		elif x.name == 'blocks.11.hook_resid_post':
			return 'resid_post'
		elif x.name == 'blocks.0.hook_resid_pre':
			return 'tok_embs'
		else:
			return x.name

	def rename_nodes(self):
		merged_graph = {}
		for receiver, senders in self.graph.items():
			new_receiver_name = self.rename_node_fn(receiver)
			new_receiver_node = HookNode(new_receiver_name, receiver.index, receiver.direction)
			if new_receiver_node not in merged_graph:
				merged_graph[new_receiver_node] = []
			merged_graph[HookNode(new_receiver_name, receiver.index, receiver.direction)].extend([HookNode(self.rename_node_fn(sender), sender.index, sender.direction) for sender in senders])

		# check that self loops are not added
		for node, senders in merged_graph.items():
			new_senders = senders.copy()
			for sender in senders:
				if sender.name == node.name:
					new_senders.remove(sender)
			merged_graph[node] = new_senders

		self.graph = merged_graph

	def get_all_edges(self):
		'''Get all edges in the computational graph.'''
		edges = []
		for receiver, senders in self.graph.items():
			for sender in senders:
				direct_computation_edge = receiver.direction == 'OUT' and sender.direction == 'IN'
				mode = 'ADD' if not direct_computation_edge else 'DIRECT'
				edges.append(Edge(sender, receiver, mode=mode))
		return edges

	def visualize(self, filename='out'):
		'''Visualize the computational graph.'''
		dot = Digraph(comment='Computational Graph', engine='dot')
		dot.attr('graph', size='12,12', ranksep='1', nodesep='0.5', dpi='300')
		dot.attr('node', shape='circle', style='filled', fillcolor='lightgrey', fontcolor='black', fontsize='12')
		dot.attr('edge', color='black', arrowsize='0.5', dir='back')

		# Add nodes and edges to the graph
		for receiver, senders in self.graph.items():
			dot.node(str(receiver.name), str(receiver.name))
			for sender in senders:
				dot.edge(str(receiver.name), str(sender.name))

		dot.render(filename, view=True)

	def __repr__(self):
		return f"{self.graph}"

