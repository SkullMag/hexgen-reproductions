export INITIAL_PEERS=$1

python3 _petals.py --model-name meta-llama/Llama-2-70b-chat-hf --token "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face access token}" --init-peers $INITIAL_PEERS