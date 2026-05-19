"""
This file contains single agent planner functions  that can be used by other planners.
Consider functions in this file as supporting functions.
"""

import heapq
import networkx as nx
import math

def calc_heuristics(graph, nodes_dict):
    """
    Calculates the exact heuristic dict (shortest distance between two nodes) to be used in A* search.
    INPUT:
        - graph = networkX graph object
        - nodes_dict = dictionary with nodes and node properties
    RETURNS:
        - heuristics = dict with shortest path distance between nodes. Dictionary in a dictionary. Key of first dict is fromnode and key in second dict is tonode.
    """
    node_ids = [nodes_dict[i]["id"] for i in nodes_dict]
    heuristics = {}
    for node_id in node_ids:
        lengths = nx.single_source_dijkstra_path_length(graph, node_id, weight="weight")
        if len(lengths) != len(node_ids):
            missing = sorted(set(node_ids) - set(lengths))
            raise Exception(
                "Heuristic cannot be calculated: No connection between "
                + str(node_id)
                + " and "
                + ", ".join(str(node) for node in missing)
            )
        heuristics[node_id] = dict(lengths)
    return heuristics



def heuristicFinder(graph, start_node, goal_node):
    """
    Finds exact distance between start_node and goal_node using the NetworkX graph.
    INPUT:
        - graph = networkX graph object
        - start_node, goal_node [int] = node_ids of start and goal node
    RETURNS:
        - path = list with node_ids that are part of the shortest path
        - path_length = length of the shortest path
    """
    try:
        path = nx.dijkstra_path(graph, start_node, goal_node, weight="weight")
        path_length = nx.dijkstra_path_length(graph, start_node, goal_node)
    except:
        path = False
        path_length = False
        raise Exception('Heuristic cannot be calculated: No connection between', start_node, "and", goal_node)
    return path, path_length



def simple_single_agent_astar(nodes_dict, from_node, goal_node, heuristics, time_start, forbidden_nodes=None, speed=1.0):
    # def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """
    Single agent A* search. Time start can only be the time that an agent is at a node.
    INPUT:
        - nodes_dict = [dict] dictionary with nodes and node properties
        - from_node = [int] node_id of node from which planning is done
        - goal_node = [int] node_id of node to which planning is done
        - heuristics = [dict] dict with shortest path distance between nodes. Dictionary in a dictionary. Key of first dict is fromnode and key in second dict is tonode.
        - time_start = [float] planning start time.
        - forbidden_nodes = [set|None] node_ids that may not be visited (used for dynamic rerouting).
        - Hint: do you need more inputs?
    RETURNS:
        - success = True/False. True if path is found and False is no path is found
        - path = list of tuples with (loc, timestep) pairs -> example [(37, 1), (101, 2)]. Empty list if success == False.
    """
    
    from_node_id = from_node
    goal_node_id = goal_node
    time_start = time_start
    
    if from_node_id == goal_node_id:
        return True, [(from_node_id, time_start)]

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = time_start
    # Normalise heuristics to time by dividing distance by speed
    if speed <= 0:
        speed = 1e-6
    h_value = heuristics[from_node_id][goal_node_id] / speed
    root = {'loc': from_node_id, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': time_start}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root
    # Depth limit: a simple path can visit at most len(nodes_dict) nodes.
    # Without this, the time-expanded closed list (keyed on (loc, timestep)) never
    # terminates when forbidden_nodes make the goal unreachable — each cycle generates
    # a unique (loc, t+k) key forever.
    max_depth = len(nodes_dict)
    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['g_val'] > max_depth:
            continue
        if curr['loc'] == goal_node_id and curr['timestep'] >= earliest_goal_timestep:
            return True, get_path(curr)

        for neighbor in nodes_dict[curr['loc']]["neighbors"]:
            if forbidden_nodes and neighbor in forbidden_nodes:
                continue
            # Compute physical traversal time based on geometric distance and agent speed
            try:
                pos_curr = nodes_dict[curr['loc']]['xy_pos']
                pos_nei = nodes_dict[neighbor]['xy_pos']
                edge_len = math.dist(pos_curr, pos_nei)
            except Exception:
                # Fallback to heuristic direct distance if xy positions missing
                edge_len = heuristics[curr['loc']].get(neighbor, 1.0)
            travel_time = edge_len / speed
            child = {
                'loc': neighbor,
                'g_val': curr['g_val'] + travel_time,
                'h_val': heuristics[neighbor][goal_node_id] / speed,
                'parent': curr,
                'timestep': curr['timestep'] + travel_time,
            }
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)
    if not forbidden_nodes:
        print("No path found, "+str(len(closed_list))+" nodes visited")
    return False, [] # Failed to find solutions

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))

def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr

def compare_nodes(n1, n2):
    """Return true if n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def get_path(goal_node):
    """Construct path if goal node is reached"""
    path = []
    curr = goal_node
    while curr is not None:
        path.append((curr['loc'], curr['timestep']))
        curr = curr['parent']
    path.reverse()
    #print(path)
    return path
