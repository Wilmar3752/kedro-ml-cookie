from kedro.config import OmegaConfigLoader

CONF_SOURCE = "conf"

CONFIG_LOADER_CLASS = OmegaConfigLoader

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "base",
    "config_patterns": {
        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
        "catalog": ["catalog*", "catalog*/**", "**/catalog*"],
        "logging": ["logging*", "logging*/**", "**/logging*"],
        "credentials": ["credentials*", "credentials*/**", "**/credentials*"],
        "mlflow": ["mlflow*"],
    },
}
