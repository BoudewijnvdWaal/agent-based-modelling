class AuctionSystem:
    def __init__(self, gse_list):
        """
        Initialiseert het veiling systeem.
        INPUT:
            - gse_list: Lijst met beschikbare GSE objecten
        """
        self.gse_list = gse_list

    def allocate_tasks(self, unassigned_tasks, heuristics):
        """
        Verdeelt taken onder de GSE's op basis van hun biedingen.
        INPUT:
            - unassigned_tasks: Lijst met gate_node_ids die service nodig hebben
            - heuristics: De voorgecalculeerde afstanden tussen alle nodes
        RETURNS:
            - assignments: Een lijst met tuples (gse_agent, task_id)
        """
        assignments = []
        
        # We veilen elke taak één voor één (Sequential Single-Item Auction)
        for task_node_id in unassigned_tasks:
            best_bid = float('inf')
            winner = None
            
            # Vraag elke beschikbare GSE om een bod
            for gse in self.gse_list:
                if gse.status == "available":
                    bid = gse.calculate_bid(task_node_id, heuristics)
                    
                    if bid < best_bid:
                        best_bid = bid
                        winner = gse
            
            # Als we een winnaar hebben, wijs de taak toe
            if winner:
                assignments.append((winner, task_node_id))
                # Zet de status van de winnaar op taxiing zodat hij niet op de volgende taak biedt in deze ronde
                winner.status = "taxiing"
                winner.assigned_gate_plane_id = task_node_id
                print(f"[Auction] Taak bij gate {task_node_id} toegewezen aan GSE {winner.id} met bod {best_bid:.2f}")
        
        return assignments
