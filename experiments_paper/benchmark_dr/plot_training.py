# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os
from pathlib import Path
import seaborn as sns
sns.set_theme(style="whitegrid", palette="colorblind")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prefix = os.environ.get(
        "GZDRL_DR_DATA_DIR", str(Path(__file__).resolve().parent / "results")
    )
    paths = [
        "dr_inverted_pendulum/"
           ]
    # failed runs paths idx: 3 @800k, 5 @ 12M, 7@1.2M
    # delays = [0, 20, 40, 80]
    labels = ["DR"]
    all_dfs = {}
    for path in paths:
        tf_files = glob.glob(
            os.path.join(prefix, path, "**", "*events*"), recursive=True
        )
        eas = [EventAccumulator(f).Reload() for f in tf_files]
        keys = ['rollout/ep_rew_mean' ]
        dfs = []
        for i, ea in enumerate(eas):
            df = pd.DataFrame()
            scalars = ea.scalars.Items(keys[0])
            for key in keys:
                df[key] = [s.value for s in scalars]
                df["step"] = [s.step for s in scalars]
                df["Index"] = [i for s in scalars]
            dfs.append(df)
        df = pd.concat(dfs)
        all_dfs[path] = df
        sns.lineplot(data=df, x="step", y="rollout/ep_rew_mean", label=labels[paths.index(path)])
    plt.show()
