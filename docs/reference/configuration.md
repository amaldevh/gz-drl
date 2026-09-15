# Environment configuration

Every GazeboPool task concatenates EnvPool's common fields, GzDRL's Gazebo
fields, and the selected task's fields. Obtain the authoritative values from a
generated spec:

```python
from gzdrl import envs

spec = envs.make_spec("GazeboPoolHoverEnv-v0", num_envs=8, batch_size=4)
print(spec.config)
print(spec.gymnasium_observation_space)
print(spec.gymnasium_action_space)
```

## Common scheduler fields

| Field | Default | Meaning |
|---|---:|---|
| `num_envs` | 1 | Persistent independent environments |
| `batch_size` | 0 | Receive batch; zero becomes `num_envs` |
| `num_threads` | 0 | Worker count; zero selects an automatic value |
| `max_num_players` | 1 | EnvPool multi-player capacity |
| `thread_affinity_offset` | -1 | Nonnegative values enable Linux CPU pinning |
| `seed` | 42 | Base environment seed |
| `gym_reset_return_info` | API-dependent | Whether Gym reset returns info |
| `max_episode_steps` | integer maximum | EnvPool truncation horizon |

The registry validates `num_envs >= 1`, `0 <= batch_size <= num_envs`, a signed
32-bit seed, and `max_num_players >= 1`.

## GzDRL base fields

| Field | Default | Meaning |
|---|---:|---|
| `test_env` | false | Use a fixed test partition instead of a unique runtime partition |
| `test_envid` | 0 | Identifier used with `test_env` |
| `resources_path` | packaged SDF path | Gazebo resource search addition |
| `plugins_path` | packaged plugin path | Gazebo system-plugin search addition |
| `gz_partition_offset` | 0 | Offset for generated transport partitions |

Task pages list fields that materially define their default interface. The C++
spec is authoritative for the complete set and for dimensions computed from
configuration.

```{warning}
Changing an observation-generating field after a policy was created can make
the policy and environment incompatible even when the registered ID is the
same. Store the full config with every checkpoint.
```
