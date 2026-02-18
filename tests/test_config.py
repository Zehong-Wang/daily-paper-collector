import logging
import os
from pathlib import Path

import pytest
import yaml

from src.config import get_env, get_project_root, load_config, setup_logging


class TestGetProjectRoot:
    def test_returns_path_with_pyproject_toml(self):
        root = get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_returns_absolute_path(self):
        root = get_project_root()
        assert root.is_absolute()


class TestLoadConfig:
    def test_loads_default_config(self):
        config = load_config()
        assert "arxiv" in config
        assert "matching" in config
        assert "llm" in config
        assert "email" in config
        assert "scheduler" in config
        assert "database" in config

    def test_database_path_is_absolute(self):
        config = load_config()
        db_path = config["database"]["path"]
        assert Path(db_path).is_absolute()

    def test_loads_custom_config(self, tmp_path):
        custom_config = {
            "arxiv": {"categories": ["cs.AI"]},
            "database": {"path": "data/test.db"},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(custom_config, f)

        config = load_config(str(config_file))
        assert config["arxiv"]["categories"] == ["cs.AI"]

    def test_custom_config_resolves_db_path(self, tmp_path):
        custom_config = {
            "database": {"path": "data/custom.db"},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(custom_config, f)

        config = load_config(str(config_file))
        assert Path(config["database"]["path"]).is_absolute()


class TestGetEnv:
    def test_raises_on_missing_key(self):
        with pytest.raises(ValueError, match="NONEXISTENT_KEY_ABC"):
            get_env("NONEXISTENT_KEY_ABC")

    def test_returns_existing_key(self):
        os.environ["TEST_KEY_XYZ"] = "hello"
        try:
            assert get_env("TEST_KEY_XYZ") == "hello"
        finally:
            del os.environ["TEST_KEY_XYZ"]


class TestSetupLogging:
    def test_can_be_called_without_error(self):
        setup_logging()

    def test_sets_root_logger_level(self):
        setup_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        # Reset to INFO
        setup_logging("INFO")
        assert logging.getLogger().level == logging.INFO
