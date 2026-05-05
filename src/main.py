from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.train import run


def main():
    run()


if __name__ == "__main__":
    main()
