#!/bin/bash

# Example commands to run the different models

# Train the teacher model
echo "Example command to train the teacher model:"
echo "python main.py --config configs/teacher_config.json"
echo ""

# Train the Student2 model (Quantized CNN)
echo "Example command to train the Student2 model (Quantized CNN):"
echo "python main.py --config configs/Student2_config.json"
echo ""

# Train the Student1 model (ResNet18)
echo "Example command to train the Student1 model (ResNet18):"
echo "python main.py --config configs/Student1_config.json"
echo ""

# Override GPU ID
echo "Example command to override GPU ID:"
echo "python main.py --config configs/teacher_config.json --gpu 0"
