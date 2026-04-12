"""Tests for environment versioning — what we just built."""

from server import __version__
from server.tasks import TASK_CONFIGS
from server.environment import StockTradingEnvironment


class TestVersionConsistency:
    """Version string exists and is well-formed."""

    def test_version_is_semver(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_version_matches_expected(self):
        assert __version__ == "1.1.0"


class TestTaskVersions:
    """Each task config has a version field."""

    def test_all_tasks_have_version(self):
        for task_id, config in TASK_CONFIGS.items():
            assert "version" in config, f"{task_id} missing version"

    def test_task_versions_are_semver(self):
        for task_id, config in TASK_CONFIGS.items():
            parts = config["version"].split(".")
            assert len(parts) == 3, f"{task_id} version not semver"
            assert all(p.isdigit() for p in parts), f"{task_id} version not numeric"


class TestVersionInObservation:
    """Version fields appear in observations returned to agents."""

    def test_fallback_observation_has_version(self):
        """Even without an active episode, version should be present."""
        env = StockTradingEnvironment()
        obs = env._build_observation(0.0)
        assert obs.env_version == __version__
        assert obs.task_version == ""
