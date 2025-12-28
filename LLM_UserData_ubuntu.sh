#!/bin/bash

apt update
apt install -y python3-venv
apt install -y python3-dev

# Set TimeZone
timedatectl set-timezone Asia/Tokyo
systemctl restart cron

# Set EC2 StopTimer
crontab -l > tempcron
echo '5 18 * * * /sbin/shutdown -h +5' >> tempcron
echo '0 21 * * * /sbin/shutdown -h now' >> tempcron
echo '0 3 * * * /sbin/shutdown -h now' >> tempcron
crontab tempcron

curl -fsSL https://ollama.com/install.sh | sh

OVERRIDE_DIR=/etc/systemd/system/ollama.service.d
mkdir -p ${OVERRIDE_DIR}
cat > ${OVERRIDE_DIR}/override.conf <<EOF
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=16384"
Environment="OLLAMA_HOST=0.0.0.0:8080"
Environment="OLLAMA_ORIGINS=*"
EOF

cat > /tmp/commandfile <<EOF
cd $HOME
echo "export OLLAMA_HOST=0.0.0.0:8080" >> .bashrc
echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" >> .bashrc
sudo systemctl daemon-reload
#sudo systemctl enable ollama
#sudo systemctl restart ollama
sudo systemctl stop ollama

cd $HOME
python3 -m venv venv
source venv/bin/activate
pip install -U pip
git clone https://github.com/bromp10jp/tune-unsloth.git
#cd tune-unsloth
#pip install -U -r requirements.txt
#python unsloth_learn.py
#python unsloth_merge.py
deactivate

cd $HOME
python3 -m venv venv_llamacpp
source venv_llamacpp/bin/activate
pip install -U pip
pip imstall awscli
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt
sudo apt install -y git cmake build-essential ninja-build libcurl4-openssl-dev
cmake -B build -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j4 --target llama-quantize
#cd ~/tune-unsloth
#python ../llama.cpp/convert_hf_to_gguf.py ./merged_model_full --outfile model-f16.gguf --outtype f16
#~/llama.cpp/build/bin/llama-quantize ./model-f16.gguf ./model-q4_k_m.gguf Q4_K_M
#aws s3 cp ./model-q4_k_m.gguf s3://oregon-up/
deactivate

EOF
su - -c "(cp /tmp/commandfile ~/.;bash ~/commandfile)" ubuntu

echo "UserData Completed."

