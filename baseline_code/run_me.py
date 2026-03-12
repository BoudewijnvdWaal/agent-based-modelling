"""
Run-me.py is the main file of the simulation. Run this file to run the simulation.
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as timer
import pygame as pg
from single_agent_planner import calc_heuristics
from visualization import map_initialization, map_running
from Aircraft import Aircraft
from independent import run_independent_planner
from prioritized import run_prioritized_planner
from cbs import run_CBS

#%% SET SIMULATION PARAMETERS
#Input file names (used in import_layout) -> Do not change those unless you want to specify a new layout.
nodes_file = "Data/nodes_EHAM.xlsx" #xlsx file with for each node: id, x_pos, y_pos, type
edges_file = "Data/edges_EHAM.xlsx" #xlsx file with for each edge: from  (node), to (node), length
plane_data_file = "Data/Plane_data.xlsx"

#Parameters that can be changed:
#Time scaling: 1 simulation time unit = 1 real minute.
#The simulation loop uses dt = 1, so each while-loop tick advances 1 real minute.
simulation_time = 12 * 60  # = 720 time units (covers 12 real hours)
planner = "Independent" #choose which planner to use (currently only Independent is implemented)

#Aircraft spawn schedule: list of tuples (spawn_time, flight_id, type, start_node, goal_node)
#Add or modify entries to change when/where aircraft appear.
spawn_schedule = [
    (10, 1, 'A', 1, 9),
    (10, 2, 'D', 11, 3),
    (30, 3, 'A', 4, 8),
]

#Gate stand occupancy: loaded from Plane_data.xlsx (columns: Gate, SIBT).
#Each entry becomes (spawn_time_minutes, gate_node_id)
gate_plane_schedule = []

#How long (in the same time units as t) a plane remains parked at a gate
gate_turnaround_time = 45.0

#Visualization (can also be changed)
plot_graph = False    #show graph representation in NetworkX
visualization = True        #pygame visualization
visualization_speed = 0.05 #set at 0.1 as default

#%%Function definitions
def import_layout(nodes_file, edges_file):
    """
    Imports layout information from xlsx files and converts this into dictionaries.
    INPUT:
        - nodes_file = xlsx file with node input data
        - edges_file = xlsx file with edge input data
    RETURNS:
        - nodes_dict = dictionary with nodes and node properties
        - edges_dict = dictionary with edges annd edge properties
        - start_and_goal_locations = dictionary with node ids for arrival runways, departure runways and gates 
    """
    gates_xy = []   #lst with (x,y) positions of gates
    cargoep_xy = [] #lst with (x,y) positions of entry points of departure runways
    chargingrr_xy = [] #lst with (x,y) positions of exit points of arrival runways
    
    df_nodes = pd.read_excel(os.getcwd() + "/" + nodes_file)
    df_edges = pd.read_excel(os.getcwd() + "/" + edges_file)
    
    #Create nodes_dict from df_nodes
    nodes_dict = {}
    for i, row in df_nodes.iterrows():
        node_properties = {"id": row["id"],
                           "x_pos": row["x_pos"],
                           "y_pos": row["y_pos"],
                           "xy_pos": (row["x_pos"],row["y_pos"]),
                           "type": row["type"],
                           "neighbors": set()
                           }
        node_id = row["id"]
        nodes_dict[node_id] = node_properties
        
        #Add node type
        if row["type"] == "cargo":
            cargoep_xy.append((row["x_pos"],row["y_pos"]))
        elif row["type"] == "charging":
            chargingrr_xy.append((row["x_pos"],row["y_pos"]))
        elif row["type"] == "gate":
            gates_xy.append((row["x_pos"],row["y_pos"]))

    #Specify node ids of gates, departure runways and arrival runways in a dict
    start_and_goal_locations = {"gates": gates_xy, 
                                "dep_rwy": cargoep_xy,
                                "arr_rwy": chargingrr_xy}
    
    #Create edges_dict from df_edges
    edges_dict = {}
    for i, row in df_edges.iterrows():
        edge_id = (row["from"],row["to"])
        from_node =  edge_id[0]
        to_node = edge_id[1]
        start_end_pos = (nodes_dict[from_node]["xy_pos"], nodes_dict[to_node]["xy_pos"])
        edge_properties = {"id": edge_id,
                           "from": row["from"],
                           "to": row["to"],
                           "length": row["length"],
                           "weight": row["length"],
                           "start_end_pos": start_end_pos
                           }
        edges_dict[edge_id] = edge_properties
   
    #Add neighbor nodes to nodes_dict based on edges between nodes
    for edge in edges_dict:
        from_node = edge[0]
        to_node = edge[1]
        nodes_dict[from_node]["neighbors"].add(to_node)  
    
    return nodes_dict, edges_dict, start_and_goal_locations

def create_graph(nodes_dict, edges_dict, plot_graph = True):
    """
    Creates networkX graph based on nodes and edges and plots 
    INPUT:
        - nodes_dict = dictionary with nodes and node properties
        - edges_dict = dictionary with edges annd edge properties
        - plot_graph = boolean (True/False) If True, function plots NetworkX graph. True by default.
    RETURNS:
        - graph = networkX graph object
    """
    
    graph = nx.DiGraph() #create directed graph in NetworkX
    
    #Add nodes and edges to networkX graph
    for node in nodes_dict.keys():
        graph.add_node(node, 
                       node_id = nodes_dict[node]["id"],
                       xy_pos = nodes_dict[node]["xy_pos"],
                       node_type = nodes_dict[node]["type"])
        
    for edge in edges_dict.keys():
        graph.add_edge(edge[0], edge[1], 
                       edge_id = edge,
                       from_node =  edges_dict[edge]["from"],
                       to_node = edges_dict[edge]["to"],
                       weight = edges_dict[edge]["length"])
    
    #Plot networkX graph
    if plot_graph:
        plt.figure()
        node_locations = nx.get_node_attributes(graph, 'xy_pos')
        nx.draw(graph, node_locations, with_labels=True, node_size=100, font_size=10)
        
    return graph

def spawn_aircrafts(t, nodes_dict, schedule):
    """Create aircraft whose spawn time matches the current timestep."""
    new_aircraft = []
    for spawn_time, flight_id, a_d, start_node, goal_node in schedule:
        if abs(spawn_time - t) < 1e-9:
            new_aircraft.append(Aircraft(flight_id, a_d, start_node, goal_node, spawn_time, nodes_dict))
    return new_aircraft

def load_gate_plane_schedule(filepath):
    """
    Load gate-plane spawn schedule from an Excel file with columns:
    - Gate: node id where the plane is parked
    - SIBT: spawn time in minutes from simulation start
    """
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    df = pd.read_excel(filepath)
    required_cols = {"Gate", "SIBT"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Plane data file {filepath} must contain columns {required_cols}")
    schedule = []
    for _, row in df.iterrows():
        try:
            gate_id = int(row["Gate"])
            spawn_time = float(row["SIBT"])
        except Exception as exc:
            raise ValueError(f"Invalid row in {filepath}: {row}") from exc
        schedule.append((spawn_time, gate_id))
    # sort by time to make behavior deterministic
    schedule.sort(key=lambda x: x[0])
    return schedule

def spawn_gate_planes(t, nodes_dict, schedule, turnaround_time, next_id_ref):
    """
    Create static gate planes whose spawn time matches the current timestep.
    Each plane remains visible until spawn_time + turnaround_time.
    """
    new_gate_planes = []
    for spawn_time, gate_node_id in schedule:
        if abs(spawn_time - t) < 1e-9:
            if gate_node_id not in nodes_dict:
                raise ValueError(f"Gate node {gate_node_id} not found in nodes_dict; update gate_plane_schedule.")
            position = nodes_dict[gate_node_id]["xy_pos"]
            new_gate_planes.append({
                "id": next_id_ref[0],
                "node_id": gate_node_id,
                "xy_pos": position,
                "despawn_time": spawn_time + turnaround_time
            })
            next_id_ref[0] += 1
    return new_gate_planes

#%% RUN SIMULATION
# =============================================================================
# 0. Initialization
# =============================================================================
nodes_dict, edges_dict, start_and_goal_locations = import_layout(nodes_file, edges_file)
graph = create_graph(nodes_dict, edges_dict, plot_graph)
heuristics = calc_heuristics(graph, nodes_dict)

aircraft_lst = []   #List which can contain aircraft agents
gate_planes = []    #List of static gate planes
gate_plane_next_id = [1]  #mutable ref so we can increment inside helper
gate_plane_schedule = load_gate_plane_schedule(plane_data_file)

if visualization:
    map_properties = map_initialization(nodes_dict, edges_dict) #visualization properties

# =============================================================================
# 1. While loop and visualization
# =============================================================================
 
#Start of while loop    
running=True
escape_pressed = False
time_end = simulation_time
dt = 1  # one simulation timestep equals 1 minute
t= 0

print("Simulation Started")
while running:
    t= round(t,2)    
       
    #Check conditions for termination
    if t >= time_end or escape_pressed: 
        running = False
        pg.quit()
        print("Simulation Stopped")
        break 

    #Spawn/remove static gate planes (parked aircraft that do not move)
    new_gate_planes = spawn_gate_planes(t, nodes_dict, gate_plane_schedule, gate_turnaround_time, gate_plane_next_id)
    if new_gate_planes:
        gate_planes.extend(new_gate_planes)
    gate_planes = [gp for gp in gate_planes if t < gp["despawn_time"] - 1e-9]
    
    #Visualization: Update map if visualization is true
    if visualization:
        current_states = {} #Collect current states of all aircraft
        for ac in aircraft_lst:
            if ac.status == "taxiing":
                current_states[ac.id] = {"ac_id": ac.id,
                                         "xy_pos": ac.position,
                                         "heading": ac.heading}
        gate_states = {gp["id"]: {"id": gp["id"], "node_id": gp["node_id"], "xy_pos": gp["xy_pos"]} for gp in gate_planes}
        escape_pressed = map_running(map_properties, current_states, gate_states, t)
        timer.sleep(visualization_speed) 
      
    #Spawn aircraft whose scheduled spawn time matches this timestep
    new_aircraft = spawn_aircrafts(t, nodes_dict, spawn_schedule)
    if new_aircraft:
        aircraft_lst.extend(new_aircraft)
         
    #Do planning 
    if planner == "Independent":     
        run_independent_planner(aircraft_lst, nodes_dict, edges_dict, heuristics, t)
    elif planner == "Prioritized":
        run_prioritized_planner()
    elif planner == "CBS":
        run_CBS()
    #elif planner == -> you may introduce other planners here
    else:
        raise Exception("Planner:", planner, "is not defined.")
                       
    #Move the aircraft that are taxiing
    for ac in aircraft_lst: 
        if ac.status == "taxiing": 
            ac.move(dt, t)
                           
    t = t + dt
          
# =============================================================================
# 2. Implement analysis of output data here
# =============================================================================
#what data do you want to show?
