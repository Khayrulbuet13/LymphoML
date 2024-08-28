#!/bin/bash

# Define the path to the Python script template
template_path="TEMPLATE.py"

# Define an array of hyperparameters configurations
# Each configuration is a string: "lr batch_size epochs num_workers im_size"
configs=(
    "1e-4 16 LR_1e-4" 
    "5e-4 16 LR_5e-4" 
    "1e-5 16 LR_1e-5" 
    "5e-5 16 LR_5e-5" 
    "1e-6 16 LR_1e-6" 
    "5e-6 16 LR_5e-6"
    "5e-5 16  batch_size_16 " 
    "5e-5 32  batch_size_32 " 
    "5e-5 64  batch_size_64 " 
    "5e-5 120 batch_size_120" 

    # Add more configurations as needed
)

# Batch size
batch_size=4

# Total number of configurations
total=${#configs[@]}

# Execute configurations in batches
for ((i=0; i<total; i+=batch_size)); do
    # Execute a batch of configurations in parallel
    for ((j=0; j<batch_size && i+j<total; j++)); do
        config=(${configs[i+j]}) # Split configuration into an array
        script_name="generated_script_$((i+j+1)).py" # Generate a unique script name

        # Replace placeholders in the template with actual values from the configuration
        cp $template_path $script_name
        sed -i "s/@@LR/${config[0]}/" $script_name
        sed -i "s/@@BATCH_SIZE/${config[1]}/" $script_name
        sed -i "s/@@MODEL_NAME/${config[2]}/" $script_name
        sed -i "s/@@CUDA_VISIBLE_DEVICES/${j}/" $script_name

        echo "Starting: $script_name with config ${configs[i+j]}"
        python $script_name &
    done
    # Wait for the batch to complete
    wait
    echo "Batch $((i/batch_size+1)) completed."
done

echo "All batches completed."
