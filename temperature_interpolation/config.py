
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

"""Central configuration and the predeclared experiment matrix."""

@dataclass
class ExperimentConfig:
    """Describe one experiment without changing the training implementation.

    Context is measured on each side of the gap. Samples per epoch are random
    windows drawn from the entire training period, rather than a chronological
    pass over every possible (heavily overlapping) window.
    """

    name: str = "A_multivariate_context_12h"
    target: str = "T (degC)"
    scenario: str = "A"
    sensor_features: list[str] = field(default_factory=lambda: [
        "T (degC)", "p (mbar)", "rh (%)", "wind_x", "wind_y",
        "max_wind_x", "max_wind_y",
    ])
    train_years: list[int] = field(default_factory=lambda: list(range(2009, 2015)))
    validation_years: list[int] = field(default_factory=lambda: [2015])
    test_years: list[int] = field(default_factory=lambda: [2016])
    gap_lengths: list[int] = field(default_factory=lambda: [1, 3, 6, 12, 24, 36])
    context_steps: int = 72
    sampling_context_steps: int = 144
    batch_size: int = 64
    epochs: int = 50
    samples_per_epoch: int = 4096
    validation_gaps_per_length: int = 64
    test_gaps_per_length: int = 256
    learning_rate: float = 0.001
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    dense_size: int = 16
    dropout: float = 0.2
    early_stopping_patience: int = 7
    scheduler_patience: int = 3
    seed: int = 20260908
    threads: int = 4

    def validate(self):
        """Reject ambiguous or unsafe configurations before creating inputs."""
        from .features import SAFE_SENSOR_FEATURES

        assert self.scenario in {"A", "B"}, "Scenario must be A or B."
        assert self.target in self.sensor_features, "Target must be an input feature."
        assert len(set(self.sensor_features)) == len(self.sensor_features)
        assert set(self.sensor_features) <= set(SAFE_SENSOR_FEATURES), (
            "Features with possible target leakage are not allowed."
        )
        assert self.target == "T (degC)", "This version reconstructs temperature."
        periods = [set(self.train_years), set(self.validation_years), set(self.test_years)]
        assert all(periods)
        assert max(periods[0]) < min(periods[1]) <= max(periods[1]) < min(periods[2])
        assert self.context_steps > 0 and min(self.gap_lengths) > 0
        assert self.sampling_context_steps >= self.context_steps
        assert len(set(self.gap_lengths)) == len(self.gap_lengths)
        assert self.batch_size > 0 and self.samples_per_epoch >= self.batch_size
        assert self.epochs > 0 and len(self.hidden_sizes) == 2

    def save(self, path: Path):
        """Save the exact settings used by a run as readable JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        """Read, validate, and return a user-editable configuration."""
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        config.validate()
        return config


def experiment_matrix():
    """Return nine runs that cover all requested controlled comparisons.

    The 12-hour multivariate model is the main experiment. The same inputs,
    architecture, seeds, and gap positions are used across context comparisons.
    Temperature-only inputs make scenarios A and B identical, so that model
    needs to be trained only once.
    """
    configs = [ExperimentConfig()]
    configs.append(ExperimentConfig(
        name="A_temperature_only_context_12h", sensor_features=["T (degC)"]
    ))
    configs.append(ExperimentConfig(name="B_multivariate_context_12h", scenario="B"))
    for context_hours in [3, 6, 24]:
        for scenario in ["A", "B"]:
            configs.append(ExperimentConfig(
                name=f"{scenario}_multivariate_context_{context_hours}h",
                scenario=scenario, context_steps=context_hours * 6,
            ))
    return configs
