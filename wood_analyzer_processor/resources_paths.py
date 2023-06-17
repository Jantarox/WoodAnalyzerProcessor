import os
import sys
from pathlib import Path

root_path = Path(sys.prefix).parent  # for installable package
# root_path = Path(__file__).parents[1]  # for local testing

config_path = os.path.join(root_path, "resources", "config.yaml")
model_path = os.path.join(root_path, "resources", "model.json")
weights_path = os.path.join(root_path, "resources", "weights.h5")