# 🥜 chota-tinker
Some infra to speed up tinkering with RL on smaller-scale

why another RL lib?

- we like tinker and the API it exposes, but doesn't have support for small models like qwen3 0.6b, 1b, 4b etc. (and its paid) -- this lib tries to match tinker like API but running locally on PyTorch
- there are other tinker like libs (in jax, and pytorch too), but there are all bloated -- we want something that people can hack inside of tinker too and add their new algorithms (new gradient update rules etc.) -- hence we try everything is minimal, setting very hard constraints on LOC and number of files. (trying to continue the tradition from: https://github.com/rl4reasoning/rl-baselines)
- we make sure its fast, so minimalism doesn't come at a price (or perhaps try to target a better trade-off b/w minimalism and efficiency :))


> [!TIP]
> "chota" stands for mini in Hindi 😄


## TODOs
- [x] see if training works or not
- [ ] we need to speed up training
    - [ ] fused loss fns
    - [ ] we can have custom model impl. for better memory and throughput during training, like: (https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-tx/tx/models/qwen3.py)[SkyRL tx]
    - [ ] add other baselines
        - [ ] MARA
