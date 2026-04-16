#!/bin/bash
# Rollback to IPPO training mode
# Disables World Model and MAPPO, returns to baseline IPPO

CONFIG_FILE="config/world_model_config.yaml"

echo "Rolling back to IPPO mode..."

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found, creating default config..."
    mkdir -p config
    cat > "$CONFIG_FILE" << EOF
global:
  use_world_model: false
  use_mappo: false
  n_agents: 5
  device: "cuda"
EOF
    echo "Default config created."
    exit 0
fi

# Disable World Model and MAPPO
.venv/bin/python3 << EOF
import yaml
import sys

config_path = "$CONFIG_FILE"
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    if 'global' not in config:
        config['global'] = {}
    
    config['global']['use_world_model'] = False
    config['global']['use_mappo'] = False
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Successfully disabled World Model and MAPPO")
    print("Current settings:")
    print(f"  use_world_model: {config['global']['use_world_model']}")
    print(f"  use_mappo: {config['global']['use_mappo']}")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

# Verify rollback
echo ""
echo "Verifying rollback..."
.venv/bin/python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); assert c['global']['use_world_model'] == False; assert c['global']['use_mappo'] == False; print('Verification passed!')"

echo ""
echo "Rollback complete. You can now run:"
echo "  python train/train_sb3.py --steps <steps>"
echo "  python train/train_self_play.py --steps <steps>"