# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set a professional visual theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# specific settings for high-res output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_data(filename="sweep_benchmark_results.pkl"):
    """Loads the pickle file and returns a Cleaned DataFrame."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None

    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    
    # ensure numeric types
    cols = ["num_envs", "batch_size", "num_threads", "fps", "duration"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    print(f"Loaded {len(df)} records.")
    return df

def plot_heatmap_fps(df):
    """
    Generates a Heatmap: Batch Size vs Num Envs.
    Color represents FPS. 
    If multiple threads exist for a coordinate, takes the MAX FPS (best case).
    """
    # Pivot the data. We take 'max' in case there are multiple thread configs for the same env/batch
    # This shows the "Peak Performance" for that hardware config
    pivot_table = df.pivot_table(
        index="num_envs", 
        columns="batch_size", 
        values="fps", 
        aggfunc=np.max
    )
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".0f", 
        cmap="viridis", 
        cbar_kws={'label': 'Steps Per Second (FPS)'},
        linewidths=.5
    )
    
    plt.title("Peak Throughput (FPS) Heatmap\n(Higher is Better)", pad=20, fontweight='bold')
    plt.ylabel("Number of Parallel Envs")
    plt.xlabel("Batch Size")
    plt.savefig("benchmark_heatmap_fps.png")
    plt.close()
    print("Saved benchmark_heatmap_fps.png")

def plot_surface_fps(df, column="batch_size", values="fps"):
    """
    Generates a Surface Plot: Batch Size vs Num Envs.
    Color represents FPS. 
    If multiple threads exist for a coordinate, takes the MAX FPS (best case).
    """
    # Pivot the data. We take 'max' in case there are multiple thread configs for the same env/batch
    # This shows the "Peak Performance" for that hardware config
    pivot_table = df.pivot_table(
        index="num_envs", 
        columns=column, 
        values=values, 
        aggfunc=np.max
    )
    
    fig = plt.figure(figsize=(10, 8))
    xvals =list(pivot_table.index)
    yvals =list(pivot_table.columns)
    X, Y = np.meshgrid(yvals, xvals)
    Z = pivot_table.values
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Steps Per Second (FPS)')
    ax.set_title("Peak Throughput (FPS) Surface Plot\n(Higher is Better)", pad=20, fontweight='bold')
    ax.set_ylabel("Number of Parallel Envs")
    ax.set_xlabel("Batch Size")
    ax.set_zlabel("Steps Per Second (FPS)") 
    plt.savefig(f"benchmark_surface_{column}_{values}.png")
    plt.close()
    print(f"Saved benchmark_surface_{column}_{values}.png")
    
def plot_batch_scaling(df):
    """
    Line Plot: How FPS scales with Batch Size, grouped by Number of Envs.
    """
    plt.figure(figsize=(12, 7))
    
    # Use a distinct palette if many env counts, or sequential if few
    unique_envs = df['num_envs'].nunique()
    palette = sns.color_palette("rocket", unique_envs)
    
    sns.lineplot(
        data=df, 
        x="batch_size", 
        y="fps", 
        hue="num_envs", 
        palette=palette,
        marker="o",
        linewidth=2.5
    )
    
    plt.title("Throughput Scaling by Batch Size", pad=20, fontweight='bold')
    plt.ylabel("Steps Per Second (FPS)")
    plt.xlabel("Batch Size")
    plt.legend(title="Num Envs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("benchmark_batch_scaling.png")
    plt.close()
    print("Saved benchmark_batch_scaling.png")

def plot_thread_impact(df):
    """
    If multiple thread counts were swept, visualize Thread Efficiency.
    """
    if df['num_threads'].nunique() <= 1:
        print("Skipping thread plot (only 1 thread config found).")
        return

    plt.figure(figsize=(12, 7))
    
    # We filter to see the impact of threads on the largest batch_size available
    # to see true compute scaling
    max_batch = df['batch_size'].max()
    subset = df[df['batch_size'] == max_batch]
    
    sns.barplot(
        data=subset,
        x="num_threads",
        y="fps",
        hue="num_envs",
        palette="viridis"
    )
    
    plt.title(f"Thread Scalability (at Batch Size {max_batch})", pad=20, fontweight='bold')
    plt.ylabel("Steps Per Second (FPS)")
    plt.xlabel("Number of Threads")
    plt.legend(title="Num Envs")
    plt.savefig("benchmark_thread_impact.png")
    plt.close()
    print("Saved benchmark_thread_impact.png")

def calculate_speedup(df):
    """
    Calculates and prints the max speedup achieved compared to the baseline.
    """
    # Assuming baseline is the entry with lowest batch_size and num_envs
    min_envs = df['num_envs'].min()
    min_batch = df[df['num_envs'] == min_envs]['batch_size'].min()
    
    baseline = df[
        (df['num_envs'] == min_envs) & 
        (df['batch_size'] == min_batch)
    ]['fps'].mean()
    
    max_fps = df['fps'].max()
    best_config = df.loc[df['fps'].idxmax()]
    
    print("-" * 30)
    print("PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"Baseline FPS ({min_envs} envs, {min_batch} batch): {baseline:.2f}")
    print(f"Maximum FPS: {max_fps:.2f}")
    print(f"Total Speedup: {max_fps / baseline:.2f}x")
    print(f"Best Configuration:")
    print(f"  - Envs: {best_config['num_envs']}")
    print(f"  - Batch: {best_config['batch_size']}")
    print(f"  - Threads: {best_config['num_threads']}")
    print("-" * 30)

def plot_batch_latency(df):
    """
    Line Plot: Batch Latency vs Batch Size.
    
    Calculates the average time it takes to simulate ONE step for the whole batch.
    Formula: Latency (ms) = (Batch Size / FPS) * 1000
    """
    # Calculate latency in milliseconds
    df['latency_ms'] = (df['batch_size'] / df['fps']) * 1000
    
    plt.figure(figsize=(12, 7))
    unique_envs = df['num_envs'].nunique()
    palette = sns.color_palette("rocket", unique_envs)
    
    sns.lineplot(
        data=df, 
        x="batch_size", 
        y="latency_ms", 
        hue="num_envs", 
        palette=palette,
        marker="o",
        linewidth=2.5
    )
    
    plt.title("Batch Latency Analysis (Lower is Better)", pad=20, fontweight='bold')
    plt.ylabel("Time per Batch Step (ms)")
    plt.xlabel("Batch Size")
    plt.legend(title="Num Envs", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a reference line for 60Hz or 30Hz real-time if relevant
    plt.axhline(y=16.6, color='r', linestyle='--', alpha=0.5, label='60Hz Real-time limit')
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("benchmark_latency.png")
    plt.close()
    print("Saved benchmark_latency.png")

def plot_scaling_efficiency(df):
    """
    Line Plot: FPS per Environment vs Total Environments.
    
    Ideally, this line should be flat (linear scaling). 
    If it drops steeply, you have significant overhead or resource contention.
    """
    # Calculate FPS contribution per environment
    df['fps_per_env'] = df['fps'] / df['num_envs']
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df,
        x="num_envs",
        y="fps_per_env",
        hue="batch_size",
        palette="viridis",
        marker="s",
        linewidth=2
    )
    
    plt.title("Scaling Efficiency: FPS per Environment", pad=20, fontweight='bold')
    plt.ylabel("FPS Contribution per Env")
    plt.xlabel("Total Number of Parallel Envs")
    plt.legend(title="Batch Size", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("benchmark_efficiency.png")
    plt.close()
    print("Saved benchmark_efficiency.png")

def plot_affinity_impact(df):
    """
    Box Plot: FPS distribution based on Thread Affinity Offset.
    
    Helps determine if pinning threads (affinity > 0) stabilizes or improves performance
    compared to letting the OS schedule freely (affinity = -1 or 0).
    """
    if df['thread_affinity_offset'].nunique() <= 1:
        print("Skipping affinity plot (only 1 config found).")
        return

    plt.figure(figsize=(10, 6))
    
    # We filter for the top 10% performing configurations to see impact on peak performance
    top_performers = df[df['fps'] > df['fps'].quantile(0.5)]
    
    sns.boxplot(
        data=top_performers,
        x="thread_affinity_offset",
        y="fps",
        palette="coolwarm"
    )
    
    plt.title("Impact of Thread Affinity on Top 50% Configs", pad=20, fontweight='bold')
    plt.ylabel("Steps Per Second (FPS)")
    plt.xlabel("Thread Affinity Offset")
    plt.savefig("benchmark_affinity_impact.png")
    plt.close()
    print("Saved benchmark_affinity_impact.png")


if __name__ == "__main__":
    # 1. Load Data
    df = load_data("sweep_benchmark_results.pkl")
    
    if df is not None:
        print("\n--- Generating Metrics ---")
        calculate_speedup(df)
        
        print("\n--- Generating Visualizations ---")
        # Original Plots
        plot_surface_fps(df)
        plot_surface_fps(df, column="num_threads", values="fps")  # Alternative surface plot
        plot_heatmap_fps(df)
        plot_batch_scaling(df)
        plot_thread_impact(df)
        
        # New Plots
        plot_batch_latency(df)
        plot_scaling_efficiency(df)
        plot_affinity_impact(df)
    
        print("\nAll 6 plots generated successfully.")