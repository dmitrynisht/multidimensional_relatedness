import os
import sys
from pathlib import Path

# Add the project root to sys.path to import config
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import and run the configuration script
import config

pass