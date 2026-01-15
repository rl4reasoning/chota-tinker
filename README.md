# 🥜 chota-tinker
Some infra to speed up tinkering with RL on smaller-scale

why another RL lib?

- we like tinker and the API it exposes, but doesn't have support for small models like qwen3 0.6b, 1b, 4b etc. (and its paid) -- this lib tries to match tinker like API but running locally on PyTorch
- there are other tinker like libs (in jax, and pytorch too), but there are all bloated -- we want something that people can hack inside of tinker too and add their new algorithms (new gradient update rules etc.) -- hence we try everything is minimal, setting very hard constraints on LOC and number of files. (trying to continue the tradition from: https://github.com/rl4reasoning/rl-baselines)
- we make sure its fast, so minimalism doesn't come at a price (or perhaps try to target a better trade-off b/w minimalism and efficiency :))


> [!TIP]
> "chota" stands for mini in Hindi 😄

# Installation

```bash
# Load modules (for Compute Canada clusters)
module load cuda/12.6 httpproxy gcc arrow/19.0.1 python/3.12 opencv/4.11

# Create venv and install
export UV_CACHE_DIR=$SCRATCH/.uv_cache
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```


## TODOs
- [x] see if training works or not
- [ ] we need to speed up training
    - [ ] fused loss fns
    - [ ] we can have custom model impl. for better memory and throughput during training, like: (https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-tx/tx/models/qwen3.py)[SkyRL tx]
    - [ ] add other baselines
        - [ ] MARA
