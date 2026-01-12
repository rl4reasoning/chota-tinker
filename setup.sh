# tinker needs >=3.11
conda create --name ct python=3.11 -y

conda activate ct # ct == chota-tinker :)

# install the latest vllm
# assumes CUDA 12.9 ?
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129

# for prime RL stuff
# pip install prime-cli
# pip install verifiers
# pip install math-verify

# export TINKER_API_KEY="tml-xUnO6eCIzrJex0Ge9C5ji3frSTRivhhj3Bi1yAVhEcOgw6Pe6YdLTQzPwfn1toHTnAAAA"

