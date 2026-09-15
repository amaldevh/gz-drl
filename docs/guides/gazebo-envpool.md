# Collect transitions with GazeboPool

<span id="use-gazebopool-environments"></span>
<span id="asynchronous-sendreceive-loop"></span>
<span id="episode-horizon"></span>
<span id="configuration-changes-can-alter-shapes"></span>

GazeboPool runs the simulation and task logic in C++ and returns observations,
rewards, and episode status to Python. Use it when training with a registered
environment. Begin with a synchronous pool to check the task interface, then
use asynchronous batches when your training loop can process partial results.

## Inspect the task

The registry is installed as `gzdrl.envs`. A spec exposes configuration and
spaces without starting a simulator:

```python
from gzdrl import envs

print(envs.list_all_envs())
spec = envs.make_spec("GazeboPoolHoverEnv-v0", num_envs=4, seed=42)
print(spec.config)
print(spec.gymnasium_observation_space)
print(spec.gymnasium_action_space)
```

The default hover task has 22 observation values and four normalized rotor
actions per environment. Other tasks and configuration-dependent dimensions
are listed in {doc}`/environments/index`.

## Step all environments together

With `batch_size=0`, the batch size becomes `num_envs`. The Gymnasium adapter
then supports a synchronous reset and step loop:

```python
import numpy as np
from gzdrl import envs

pool = envs.make_gymnasium(
    "GazeboPoolHoverEnv-v0",
    num_envs=4,
    batch_size=0,
    num_threads=4,
    seed=42,
)
observation, info = pool.reset()

for _ in range(100):
    action = np.zeros((4, 4), dtype=np.float32)
    observation, reward, terminated, truncated, info = pool.step(action)

print(observation.shape, reward.shape)
```

The observation batch has shape `(4, 22)` and the reward batch has shape `(4,)`.
Zero actions are an interface check; they do not define a hover policy. In a
training loop, the policy produces one action per observation row.

GazeboPool resets an environment on the next step after it has finished an
episode. Account for the reset transition when adapting the pool to a training
library; its batched interface is not automatically an SB3 `VecEnv`.

## Receive partial batches

When `batch_size < num_envs`, use `async_reset()`, `recv()`, and `send()`:

```python
import numpy as np
from gzdrl import envs

pool = envs.make_gymnasium(
    "GazeboPoolHoverEnv-v0",
    num_envs=8,
    batch_size=4,
    num_threads=4,
    seed=42,
)
pool.async_reset()

for _ in range(100):
    observation, reward, terminated, truncated, info = pool.recv()
    action = np.zeros((len(info["env_id"]), 4), dtype=np.float32)
    pool.send(action, info["env_id"])
```

A receive returns whichever environments completed the batch. Pass those IDs
back with their corresponding actions; row order can change between batches.
`batch_size` must be between zero and `num_envs`, inclusive, and `num_envs` must
be at least one. See {doc}`/concepts/vectorization` for the scheduling model.

## Configure an experiment

The same registry provides `make_gym()` and `make_dm()` for Gym and dm-env.
Their return conventions differ; choose the adapter expected by your learner.
The examples above use Gymnasium's separate `terminated` and `truncated` flags.

For all seven bundled tasks, the registry copies `max_steps_per_episode` into
`max_episode_steps` unless you explicitly supply the latter. The task also has
its own completion conditions. Inspect the full spec when setting a rollout
horizon.

History length, rotation representation, and agent count can change the spaces.
Save the complete configuration with each policy and recreate the policy's
input and output layers when those dimensions change. See
{doc}`/reference/configuration` for shared fields and
{doc}`/examples/index` for training integrations.
