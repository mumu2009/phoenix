#!/usr/bin/env python3
"""hb_mapper CLI wrapper that applies the ONNX temp serialization patch."""
import sys
import hb_mapper_patch  # noqa: F401  # Must be imported before horizon_tc_ui.
from horizon_tc_ui.hb_mapper import main

if __name__ == "__main__":
    sys.exit(main())
