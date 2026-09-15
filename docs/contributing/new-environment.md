# Add an environment page

After implementing and registering a task, create
`docs/environments/<slug>/index.md`:

```markdown
---
title: My task
summary: A one-sentence task and control-interface description.
order: 80
env_id: GazeboPoolMyTaskEnv-v0
observation: "float32[42]"
action: "float32[4]"
catalog: true
---

# My task

**Registered ID:** `GazeboPoolMyTaskEnv-v0`

Explain the objective, action mapping, observation composition, termination,
world/model assumptions, shape-changing fields, and important defaults.
```

The catalog metadata must match the registered binding exactly. Document the
shape before the outer environment-batch dimension. If a field such as history
length or agent count changes the shape, include its formula.

Before publishing:

- verify the ID appears in `envs.list_all_envs()`;
- construct the spec and check action/observation shapes;
- run reset and a short step loop;
- state the domain-randomization default and cadence boundary; and
- identify legacy/test status with `catalog: false` when it is not a supported
  production task.
