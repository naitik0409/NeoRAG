#!/usr/bin/env python
import sys

from neorag.cli import main

if __name__ == "__main__":
    sys.exit(main(["build-dataset"] + sys.argv[1:]))
