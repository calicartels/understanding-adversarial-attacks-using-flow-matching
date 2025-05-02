import os
import subprocess
from pathlib import Path

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Clone flow_matching if not already present
if not os.path.exists('flow_matching'):
    run_command('git clone https://github.com/facebookresearch/flow_matching.git')

# 2. Install package
run_command('pip install -e flow_matching')

# 3. Set up image example directory
os.chdir('flow_matching/examples/image')
run_command('pip install -r requirements.txt')

# 4. Create output directory and __init__.py files
os.makedirs('output_dir', exist_ok=True)
Path('models/__init__.py').touch()
Path('training/__init__.py').touch()

print("Setup completed successfully!") 