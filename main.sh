#!/bin/bash

# Run create_scene.py and automatically provide 'y' if extra input is requested
python3 create_scene.py <<EOF
y
EOF

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "create_scene.py executed successfully. Running execute_scene.py..."
    # Run execute_scene.py
    python3 execute_scene.py lab-ur5e
else
    echo "create_scene.py failed. Aborting."
    exit 1
fi