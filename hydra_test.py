import argparse
import hydra
from omegaconf import DictConfig, OmegaConf


# @hydra.main(version_base=None, config_path="config", config_name="config")
def my_app() -> None:
    cfg = OmegaConf.load("config/config.yaml")

    driver_name = cfg.db["driver"]
    file_path = cfg["file_path"]
    model_base_score = cfg.model["base_score"]

    print(driver_name, file_path, type(model_base_score))
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
# def run_all_sessions(
#     file_path="fraud.csv",
#     test_size_1=0.2,
#     test_size_2=0.5,
#     random_state=35,
# ):
#     artifacts = dict()
#     artifacts.update(
#         run_session_including_load_data(
#             file_path, test_size_1, test_size_2, random_state
#         )
#     )
#     return artifacts
# @hydra.main(version_base=None, config_path="config", config_name="config")
# def run_all_sessions (cfg : DictConfig) -> None:
#     """Run all sessions."""
#     file_path=cfg["file_path"]
#     test_size_1=cfg["test_size_1"]
#     test_size_2=cfg["test_size_2"]
#     random_state=cfg["random_state"]
#     artifacts = dict()
#     artifacts.update(
#         run_session_including_load_data(
#             file_path, test_size_1, test_size_2, random_state
#         )
#     )
#     return artifacts


# @hydra.main(version_base=None, config_path="config", config_name="config")
# def main(cfg : DictConfig) -> None:
#     """Run all sessions."""

#     # Edit this section to customize the behavior of artifacts
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--file_path",
#         type=str,
#         default=cfg["file_path"],
#     )
#     parser.add_argument("--test_size_1", type=float, default=cfg["test_size_1"])
#     parser.add_argument("--test_size_2", type=float, default=cfg["test_size_2"])
#     parser.add_argument("--random_state", type=int, default=cfg["random_state"])
#     args = parser.parse_args()

#     artifacts = run_all_sessions(
#         file_path=args.file_path,
#         test_size_1=args.test_size_1,
#         test_size_2=args.test_size_2,
#         random_state=args.random_state,
#     )
#     print(artifacts)


# if __name__ == "__main__":
#     main()
