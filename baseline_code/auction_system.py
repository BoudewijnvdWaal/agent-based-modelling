class AuctionSystem:
    def __init__(self, gse_list, alpha=1.0, beta=1.0, max_shortest_path_distance=1.0):
        """
        Initialize the auction system.
        INPUT:
            - gse_list: list of available GSE objects
            - alpha: weight for normalized distance
            - beta: weight for normalized SoC penalty
            - max_shortest_path_distance: fixed normalization scale for distance
        """
        self.gse_list = gse_list
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_shortest_path_distance = max(float(max_shortest_path_distance), 1e-9)

    def allocate_tasks(self, unassigned_tasks, heuristics):
        """
        Assign tasks to GSEs based on bids.
        INPUT:
            - unassigned_tasks: list of Plane objects or gate_node_ids needing service
            - heuristics: precomputed distances between all nodes
        RETURNS:
            - assignments: a list of tuples (gse_agent, task)
        """
        assignments = []
        
        # Auction each task one by one (Sequential Single-Item Auction)
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
            
            # Request a bid from each available GSE
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
            
            # If a winner exists, assign the task
            if winner:
                assignments.append((winner, task))
                # Set winner status to taxiing so it cannot bid again in this round
                winner.status = "taxiing"
                winner.assigned_plane_id = task_id
                winner.assigned_service_type = task_type
                service_label = f" ({task_type})" if task_type else ""
                print(f"[Auction] Task at gate {task_node_id}{service_label} assigned to GSE {winner.id} with bid {best_bid:.2f}")
        
        return assignments
