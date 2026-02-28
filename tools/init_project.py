import yaml
import os

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.env import collect_env_info
from utils.path import ensure_dir


def main():

    with open("configs/dpseg_tgs.yaml") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["project"]["output_dir"]
    ensure_dir(output_dir)

    logger = setup_logger(os.path.join(output_dir, "init.log"))

    logger.info("Initializing DpSSeg project")
    logger.info("Setting random seed")

    set_seed(cfg["runtime"]["seed"])

    logger.info("Collecting environment information")

    env_info = collect_env_info()
    for k, v in env_info.items():
        logger.info(f"{k}: {v}")

    logger.info("Project initialization complete.")


if __name__ == "__main__":
    main()