import yaml
from src.pipelines.train_pipeline import TrainingPipelineConfig, run_training

with open("configs/paths.yaml") as f:
    paths = yaml.safe_load(f)

with open("configs/model.yaml") as f:
    model = yaml.safe_load(f)

conf = TrainingPipelineConfig(
    gated_train=paths["datasets"]["gate"]["train"],
    gated_val=paths["datasets"]["gate"]["val"],
    dir_train=paths["datasets"]["dir"]["train"],
    dir_val=paths["datasets"]["dir"]["val"],
    traj_train_aug_3cls=paths["datasets"]["trajectories"]["train_aug_3cls"],
    traj_val_3cls=paths["datasets"]["trajectories"]["val_3cls"],
    ckpt_dir=paths["outputs"]["checkpoints_dir"],
    seed=model["runtime"]["seed"],
    device=model["runtime"]["device"],
    do_gate=model["stages"]["gate"],
    do_dir=model["stages"]["dir"],
    do_mag_cli=model["stages"]["mag_cli"],
    do_mag_cld=model["stages"]["mag_cld"],
)

run_training(conf)
