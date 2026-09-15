# How GzDRL works

<span id="concepts"></span>

The central operation is one environment transition: an action changes the
simulated vehicle, and the resulting state becomes an observation. Start with
that sequence, then examine how parallel execution, coordinate frames, and
control choices affect an experiment.

```{toctree}
:maxdepth: 1

execution-semantics
vectorization
state-frames-units
control-abstractions
determinism
```
