class AuctionSystem:
    def __init__(self, gse_list, alpha=1.0, beta=1.0, max_shortest_path_distance=1.0):
        """
        Initialiseert het veiling systeem.
        INPUT:
            - gse_list: Lijst met beschikbare GSE objecten
            - alpha: gewicht voor genormaliseerde afstand
            - beta: gewicht voor genormaliseerde SoC-penalty
            - max_shortest_path_distance: vaste normalisatieschaal voor afstand
        """
        self.gse_list = gse_list
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_shortest_path_distance = max(float(max_shortest_path_distance), 1e-9)

    def allocate_tasks(self, unassigned_tasks, heuristics):
        """
        Verdeelt taken onder de GSE's op basis van hun biedingen.
        INPUT:
            - unassigned_tasks: Lijst met plane-objecten of gate_node_ids die service nodig hebben
            - heuristics: De voorgecalculeerde afstanden tussen alle nodes
        RETURNS:
            - assignments: Een lijst met tuples (gse_agent, task)
        """
        assignments = []
        
        # We veilen elke taak één voor één (Sequential Single-Item Auction)
        for task in unassigned_tasks:
            task_node_id = getattr(task, "node_id", task)
            task_id = getattr(task, "id", task_node_id)
            task_type = getattr(task, "next_service_type", None)
            second_node_id = None
            if task_type == "load":
                task_node_id = task.cargo_from
                second_node_id = task.node_id
            elif task_type == "unload":
                task_node_id = task.node_id
                second_node_id = task.cargo_to
            best_bid = float('inf')
            winner = None
            
            # Vraag elke beschikbare GSE om een bod
            for gse in self.gse_list:
                if gse.status == "available":
                    bid = gse.calculate_bid(
                        task_node_id,
                        heuristics,
                        second_node_id=second_node_id,
                        alpha=self.alpha,
                        beta=self.beta,
                        max_shortest_path_distance=self.max_shortest_path_distance,
                    )
                    
                    if bid < best_bid:
                        best_bid = bid
                        winner = gse
            
            # Als we een winnaar hebben, wijs de taak toe
            if winner:
                assignments.append((winner, task))
                # Zet de status van de winnaar op taxiing zodat hij niet op de volgende taak biedt in deze ronde
                winner.status = "taxiing"
                winner.assigned_plane_id = task_id
                winner.assigned_service_type = task_type
                service_label = f" ({task_type})" if task_type else ""
                print(f"[Auction] Taak bij gate {task_node_id}{service_label} toegewezen aan GSE {winner.id} met bod {best_bid:.2f}")
        
        return assignments
