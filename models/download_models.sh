#!/bin/bash

# Check if at least two arguments were provided (excluding the optional first one)
if [ $# -lt 2 ]; then
    echo "Insufficient arguments provided. At least two arguments are required."
    exit 1
fi

# Check for the optional "essential" argument and download the essential models if present
if [ "$1" == "essential" ]; then
        echo "Downloading Essential Models (EfficientNet, Stage A, Previewer)"
        wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_a.safetensors -P . -q --show-progress
        wget https://huggingface.co/stabilityai/StableWurst/resolve/main/previewer.safetensors -P . -q --show-progress
        wget https://huggingface.co/stabilityai/StableWurst/resolve/main/effnet_encoder.safetensors -P . -q --show-progress
    shift # Move the arguments, $2 becomes $1, $3 becomes $2, etc.
fi

# Now, $1 is the second argument due to the potential shift above
second_argument="$1"
binary_decision="${2:-bfloat16}" # Use default or specific binary value if provided

case $second_argument in
    big-big)
        if [ "$binary_decision" == "bfloat16" ]; then
            echo "Downloading Large Stage B & Large Stage C"
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b_bf16.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c_bf16.safetensors -P . -q --show-progress
        else
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c.safetensors -P . -q --show-progress
        fi
        ;;
    big-small)
        if [ "$binary_decision" == "bfloat16" ]; then
            echo "Downloading Large Stage B & Small Stage C (BFloat16)"
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b_bf16.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c_lite_bf16.safetensors -P . -q --show-progress
        else
            echo "Downloading Large Stage B & Small Stage C"
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c_lite.safetensors -P . -q --show-progress
        fi
        ;;
    small-big)
        if [ "$binary_decision" == "bfloat16" ]; then
            echo "Downloading Small Stage B & Large Stage C (BFloat16)"
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b_lite_bf16.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c_bf16.safetensors -P . -q --show-progress
        else
            echo "Downloading Small Stage B & Large Stage C"
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b_lite.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c.safetensors -P . -q --show-progress
        fi
        ;;
    small-small)
        if [ "$binary_decision" == "bfloat16" ]; then
            echo "Downloading Small Stage B & Small Stage C (BFloat16)"
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b_lite_bf16.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c_lite_bf16.safetensors -P . -q --show-progress
        else
            echo "Downloading Small Stage B & Small Stage C"
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_b_lite.safetensors -P . -q --show-progress
            wget https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_c_lite.safetensors -P . -q --show-progress
        fi
        ;;
    *)
        echo "Invalid second argument. Please provide a valid argument: big-big, big-small, small-big, or small-small."
        exit 2
        ;;
esac
