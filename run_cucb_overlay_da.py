import copy

class CombinatorialUCB:
    def __init__(self, server_profiles, models, max_copies_per_model=1, alpha=1.0, k_select=None):
        self.server_profiles = server_profiles
        self.models = models
        self.alpha = alpha
        self.k_select = k_select
        self.max_copies_per_model = max_copies_per_model
        self.instances = self._init_instances()

    def _init_instances(self):
        """Initialize multiple instances of each model on servers."""
        instances = {}
        for server, resources in self.server_profiles.items():
            instances[server] = {
                model_id: self.max_copies_per_model
                for model_id in self.models.keys()
            }
        return instances

    def arm_resource(self, arm):
        """Update the logic to handle multiple instances."""
        server, model_id, replica_id = arm  # Each arm now includes a replica ID.
        return self.instances[server][model_id] > 0

    def update_instance_usage(self, arm):
        """Reduce the available instance for the selected model."""
        server, model_id, replica_id = arm
        if self.instances[server][model_id] > 0:
            self.instances[server][model_id] -= 1

    def select(self):
        """Select arms while considering per-model per-server instance limits."""
        selected = []
        for server, model_map in self.instances.items():
            for model_id, instance_count in model_map.items():
                for replica_id in range(self.max_copies_per_model):
                    if instance_count > 0:
                        selected.append((server, model_id, replica_id))
        return selected[: self.k_select] if self.k_select else selected

    def update(self, chosen, observed_rewards):
        """Update the result of the chosen arms."""
        for arm in chosen:
            self.update_instance_usage(arm)


# Add deployment tracking at the end of each frame.
def run_one_seed(graphml, models_json, tasks_json, out_dir, seed=0, frames=200, steps_per_frame=5, base_b=3, alpha=1.0, k_select=None, overlay_hop_penalty=0.5, verbose=False):
    # Initialization code (unchanged)
    # [...]

    for frame in range(frames):
        print(f"\n[Frame {frame}] Starting frame simulation...")

        # CUCB algorithm for the frame
        cucb = CombinatorialUCB(
            server_profiles=server_profiles,
            models=models,
            max_copies_per_model=2,  # Allow up to 2 copies of each model per server
            alpha=alpha,
            k_select=k_select,
        )

        chosen_arms = cucb.select()
        rewards_observed = []

        # Simulate task processing and observe rewards
        for step in range(steps_per_frame):
            rewards = simulate_tasks_and_observe_rewards(chosen_arms)  # Defined elsewhere
            rewards_observed.extend(rewards)

        cucb.update(chosen_arms, rewards_observed)

        # Log deployment tracking at the end of each frame
        print("[Deployment Tracking]")
        for server, model_map in cucb.instances.items():
            for model_id, remaining_instances in model_map.items():
                print(f"Server {server} - Model {model_id}: {remaining_instances} instances remaining")

        print(f"[Frame {frame}] Rewards: {rewards_observed}")