import os

import wandb


def init_wandb(args):
    run_name = args.wandb_run_name or os.path.basename(args.save_path)
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
