"""Hugging Face Spaces entrypoint.

The real app lives in src/app.py so local dev can run `python -m src.app` while
Spaces can use the conventional root-level app.py.
"""

from src.app import main


if __name__ == "__main__":
    main()
