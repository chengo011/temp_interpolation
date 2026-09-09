import argparse
from pathlib import Path
"""Command-line entry point; all project outputs remain below this directory."""

ROOT = Path(__file__).resolve().parent

def project_path(value):
    """Resolve user-provided paths and reject outputs outside the project."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All input and output paths must remain inside the project folder.")
    return path


def main():
    """Expose preparation, full experiments, and saved-model interpolation."""
    parser = argparse.ArgumentParser(description="Jena Climate temperature gap reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Download, audit, split, and save fixed gaps")
    run = commands.add_parser("run", help="Train all nine models, then evaluate and plot")
    run.add_argument("--run-name", default="full_experiment")
    inference = commands.add_parser("interpolate", help="Reconstruct bounded gaps in a new CSV")
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--model-dir", required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        from temperature_interpolation.pipeline import prepare_project
        prepare_project(ROOT)
    elif arguments.command == "run":
        if Path(arguments.run_name).name != arguments.run_name or arguments.run_name in {".", ".."}:
            raise ValueError("Run name must be a single directory name.")
        from temperature_interpolation.pipeline import run_suite
        run_suite(ROOT, arguments.run_name)
    else:
        from temperature_interpolation.inference import interpolate_csv
        result = interpolate_csv(project_path(arguments.input), project_path(arguments.output), project_path(arguments.model_dir))
        print(f"Reconstructed {result.is_reconstructed.sum()} temperature values.")


if __name__ == "__main__":
    main()
