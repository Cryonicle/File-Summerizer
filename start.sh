#!/bin/bash

# echo "set permission file"

# chmod +x run-ollama.sh

# echo "run bash file"

# bash run-ollama.sh

# echo "run fastapi app"

# echo "forward ip"

# echo 1 > /proc/sys/net/ipv4/ip_forward
# sudo sysctl -p

# echo "install iptables command"
# sudo apt update
# sudo apt install iptables -y

# echo "set iptables port forward"

# sudo iptables -t nat -A PREROUTING -p tcp --dport 11434 -j DNAT --to-destination 2.176.196.117:11434
# sudo iptables -t nat -A POSTROUTING -j MASQUERADE

cp /app/ollama_client.py /usr/local/lib/python3.12/site-packages/llama_index/llms/ollama/base.py

python app.py
