import os
import wandb
from omegaconf import OmegaConf


class Logger:
    """
    Class to initialize logger object for writing experiment results to wandb.
    """
    def __init__(self, config):
        self.verbose = config["logger"].get("verbose", False)
        self.run = wandb.init(
            project=config["logger"]["project"],
            entity=config["logger"]["entity"],
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            tags=config["logger"].get("tags", None),
            notes=config["logger"].get("notes", None),
            group=config["logger"].get("group", None),
            mode=config["logger"].get("mode", None),
            reinit=True,
            )
        self.define_metrics()

    def log(self, data, step=None, commit=False):
        wandb.log(data, step=step, commit=commit)

    def log_item(self, tag, val, step=None, commit=True, **kwargs):
        self.log({tag: val, **kwargs}, step=step, commit=commit)
        if self.verbose:
            print(f"{tag}: {val}")

    def commit(self):
        self.log({}, commit=True)

    def log_xp_matrix(self, tag, mat, step=None, columns=None, rows=None, commit=True, **kwargs):
        if rows is None:
            rows = [str(i) for i in range(mat.shape[0])]
        if columns is None:
            columns = [str(i) for i in range(mat.shape[1])]
        tab = wandb.Table(
                columns=columns,
                data=mat,
                rows=rows
                )
        wandb.log({tag: tab, **kwargs}, step=step, commit=commit)

    def define_metrics(self):
        wandb.define_metric("train_step")
        wandb.define_metric("checkpoint")
        wandb.define_metric("Train/*", step_metric="train_step")
        wandb.define_metric("Losses/*", step_metric="train_step")
        wandb.define_metric("Eval/*", step_metric="train_step")
        wandb.define_metric("Returns/*", step_metric="train_step")
        wandb.define_metric("HeldoutEval/*", step_metric="iter")
    
    def log_artifact(self, name, path, type_name):
        artifact = wandb.Artifact(name, type=type_name)
        # check if path is a directory or a file
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self.run.log_artifact(artifact)
    
    def log_video(self, tag, path, commit=True):
        wandb.log({tag: wandb.Video(path)}, commit=commit)
    
    def close(self):
        wandb.finish()
