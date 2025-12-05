"""
Compatibility entry point so you can still run:

    uv run python main.py [args...]

Wrapper untuk menjelankan program yang inti `pedestrian_line_counter.main`.
"""

from pedestrian_line_counter.main import main


if __name__ == "__main__":  # pragma: no cover
    main()

