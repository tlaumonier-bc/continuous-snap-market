"""Make the project root importable so tests can `import snapmarket`."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
