import wandb

class Logger:
    def __init__(
        self, 
        group: str, 
        name: str, 
        project: str, 
        method: str, 
        dataset: str,
        smoketest: bool = False,
    ) -> None:
        
        wandb.init(
            mode="disabled" if smoketest else None,
            group=group,
            name=name,
            project=project,
            config={
                "method": method,
                "dataset": dataset,
            }
        )

        # custom metric https://docs.wandb.ai/guides/track/log/customize-logging-axes
        wandb.define_metric("custom_step")
        wandb.define_metric("baseline", step_metric="custom_step")
        wandb.define_metric("found", step_metric="custom_step")

    def close(self) -> None:
        wandb.finish()
