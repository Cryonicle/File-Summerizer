#!/bin/bash

echo "install screen command"

apt install screen -y

echo "install ollama app"

curl -fsSL https://ollama.com/install.sh | sh

echo "run ollama app"

screen -d -m ollama serve

echo "wait ollama serve"

sleep 10

echo "pull llama3 model"

ollama pull llama3