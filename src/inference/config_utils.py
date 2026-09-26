"""JSON config support for the inference CLIs: ``--config file.json`` sets defaults, CLI flags override."""
import argparse
import json
import os


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def setup_parser_with_config_support(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default="", help="JSON file whose keys become argument defaults")
    return parser


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Read ``--config`` first, apply it as defaults, then parse the full command line on top."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        config = load_config(known.config)
        dests = {a.dest for a in parser._actions}
        unknown = [k for k in config if k not in dests]
        if unknown:
            print(f"[config] ignoring unknown keys in {known.config}: {unknown}")
        parser.set_defaults(**{k: v for k, v in config.items() if k not in unknown})
        print(f"Loaded configuration from {known.config}")
    return parser.parse_args(argv)


def save_config(args: argparse.Namespace, output_path: str):
    config = {k: v for k, v in vars(args).items() if not callable(v)}
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {output_path}")
