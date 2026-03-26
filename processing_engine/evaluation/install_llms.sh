#!/bin/bash

# List of models.
# Built-in Ollama models use their standard names.
# Hugging Face models require the hf.co/ prefix and the exact repository name.
MODELS=(
    "mistral-large"
    "llama3.1:70b"
    "hf.co/dicta-il/dictalm-3.0-instruct-gguf"
    "hf.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"
    "hf.co/cohereForAI/c4ai-command-a-03-2025"
)

echo "=================================================="
echo "Starting the model download process via Ollama..."
echo "=================================================="

for MODEL in "${MODELS[@]}"; do
    echo "Attempting to download model: $MODEL"
    echo "Please wait..."
    
    # Run the download command
    ollama pull "$MODEL"
    
    # Check if the command completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Success: Model $MODEL has been downloaded and is ready to use."
    else
        echo "❌ Failure: Could not download $MODEL."
        echo "Possible reason: The model does not exist in the Ollama registry, or GGUF files are missing in its Hugging Face repository."
    fi
    echo "--------------------------------------------------"
done

echo "Process completed!"
