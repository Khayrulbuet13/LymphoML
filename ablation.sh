#!/bin/bash

# Define the path to the Python script template
template_path="TEMPLATE.py"

# Define an array of hyperparameters configurations
# Each configuration is a string: "lr batch_size epochs num_workers im_size"
configs=(
    "2 .75 1.50 20 GB-2"
    "3 .75 1.50 20 GB-3"
    "4 .75 1.50 20 GB-4"
    "3 .50 1.50 20 LCS-.50"
    "3 .75 1.50 20 LCS-.75"
    "3 .10 1.50 20 LCS-1.00"
    "3 .50 1.00 20 LCE-1.00"  
    "3 .50 1.50 20 LCE-1.50"
    "3 .50 1.75 20 LCE-1.75"
    "3 .75 1.50 10 ROT-10" 
    "3 .75 1.50 20 ROT-20"
    "3 .75 1.50 30 ROT-30"


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
        sed -i "s/@@GB/${config[0]}/" $script_name
        sed -i "s/@@LCS/${config[1]}/" $script_name
        sed -i "s/@@LCE/${config[2]}/" $script_name
        sed -i "s/@@ROT/${config[3]}/" $script_name
        sed -i "s/@@ROT/${config[3]}/" $script_name
        sed -i "s/@@MODEL_NAME/${config[4]}/" $script_name
        sed -i "s/@@CUDA_VISIBLE_DEVICES/${j}/" $script_name

        echo "Starting: $script_name with config ${configs[i+j]}"
        python $script_name &
    done
    # Wait for the batch to complete
    wait
    echo "Batch $((i/batch_size+1)) completed."
done

echo "All batches completed."
