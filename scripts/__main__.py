"""Allow `python -m localswarm` to launch the CLI."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
