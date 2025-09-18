import os
import sys
from pathlib import Path


def setup_environment():
    # Get the absolute path to the project root
    project_root = Path(__file__).resolve().parent

    # Construct the absolute path to the src directory
    src_path = project_root / 'src'

    # Add the src directory to the PYTHONPATH and sys.path
    os.environ['PYTHONPATH'] = str(src_path)
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    # Construct the absolute path to the imports subfolder of src
    imports_path = src_path / 'imports'
    
    # Add the imports subfolder of src to the PYTHONPATH and sys.path
    os.environ['PYTHONPATH'] += os.pathsep + str(imports_path)
    if str(imports_path) not in sys.path:
        sys.path.append(str(imports_path))

    # Set the PROJECT_ROOT environment variable
    os.environ['PROJECT_ROOT'] = str(project_root)

    # Verify PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH')
    print(f"PYTHONPATH: {pythonpath}")

    # # Verify PROJECT_ROOT
    # project_root_env = os.environ.get('PROJECT_ROOT')
    # print(f"PROJECT_ROOT: {project_root_env}")

    # # Verify sys.path
    # print("sys.path:", sys.path)


# Run the setup function
setup_environment()