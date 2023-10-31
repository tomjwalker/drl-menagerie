import sys
import os

# Get the current directory of this conftest.py file.
current_dir = os.path.dirname(__file__)

# Calculate the path to the project root directory.
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root directory to the Python path.
sys.path.insert(0, project_root)
