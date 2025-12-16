"""
main script untuk menjalankan fungsionalitas dari program yang ada
cara menjalankannya adalah dengan command

    uv run python main.py [args...]

Wrapper untuk menjelankan program yang inti `pedestrian_line_counter.main`.
"""

from pedestrian_line_counter.main import main

if __name__ == "__main__":  # pragma: no cover
    main()

