import math
from single_agent_planner import simple_single_agent_astar

class GSE:
    def __init__(self, gse_id, start_node, nodes_dict):
        self.id = gse_id
        self.current_node = start_node
        self.position = nodes_dict[start_node]["xy_pos"]
        self.nodes_dict = nodes_dict
        
        # Status en Doelen
        self.status = "available"  # available, taxiing, charging, working
        self.goal = None
        self.assigned_gate_plane_id = None
        
        # Batterij logica
        self.soc = 100.0           # State of Charge in %
        self.consumption_rate = 0.5 # Verbruik per tijdstap
        self.charging_rate = 2.0    # Oplaadsnelheid in het depot
        
        # Planning
        self.path_to_goal = []
        self.speed = 1.0

    def calculate_bid(self, gate_node_id, heuristics):
        """Hoe lager het getal, hoe beter het bod."""
        if self.status != "available" or self.soc < 25:
            return float('inf') # Kan niet bieden
        
        # Afstand van huidige plek naar de gate
        distance = heuristics[self.current_node][gate_node_id]
        
        # Bod = Afstand + 'straf' voor een lagere batterij
        # Een GSE met 100% SoC krijgt voorrang op een GSE met 40% bij gelijke afstand
        bid = distance + (100 - self.soc) * 0.2
        return bid

    def update_soc(self, dt):
        if self.status == "taxiing":
            self.soc -= self.consumption_rate * dt
        elif self.status == "charging":
            self.soc += self.charging_rate * dt
            if self.soc >= 100:
                self.soc = 100
                self.status = "available"
        
        # Forceer opladen als de accu echt leeg is
        if self.soc < 15 and self.status == "available":
            self.status = "needs_charging"

    def move(self, dt):
        """Simpele beweging over het pad (vergelijkbaar met Aircraft.py)"""
        if self.path_to_goal:
            # Hier komt je logica om de xy_pos te updaten richting de volgende node in het pad
            # Voor nu zetten we hem simpelweg op de volgende node voor de demo
            next_step = self.path_to_goal.pop(0)
            self.current_node = next_step[0]
            self.position = self.nodes_dict[self.current_node]["xy_pos"]