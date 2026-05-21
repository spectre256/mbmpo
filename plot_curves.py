import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob
import os

def plot_rl_curves_from_folder(folder_path="."):
    # Ensure the folder actually exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Find all CSV files inside the specified folder
    search_path = os.path.join(folder_path, "*.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No CSV files found in folder: '{folder_path}'")
        return

    print(f"Found {len(files)} CSV files in '{folder_path}'. Aggregating data...")
    
    # Read all CSVs into a list of DataFrames
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Quick sanity check to ensure the file has the expected headers
            if 'iteration' in df.columns and 'ep_reward' in df.columns:
                dfs.append(df)
            else:
                print(f"Skipping {os.path.basename(f)}: Missing required columns.")
        except Exception as e:
            print(f"Could not read {os.path.basename(f)}: {e}")

    if not dfs:
        print("No valid metric CSV files to plot.")
        return
    
    # Extract iterations from the first valid file
    iterations = dfs[0]['iteration'].values
    
    # Extract rewards into 2D arrays (shape: num_runs x num_iters)
    ep_rewards = np.array([df['ep_reward'].values for df in dfs])
    # Calculate Means
    ep_mean = np.mean(ep_rewards, axis=0)

    # Calculate 90% Confidence Interval
    confidence = 0.90
    z_value = stats.norm.ppf((1 + confidence) / 2.0)
    
    n_runs = len(dfs)
    if n_runs > 1:
        ep_std_err = np.std(ep_rewards, axis=0) / np.sqrt(n_runs)
    else:
        ep_std_err = np.zeros_like(ep_mean)

    ep_ci = z_value * ep_std_err

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot Real Environment Episodic Reward
    plt.plot(iterations, ep_mean, label=f"Real Episode Reward (n={n_runs})", color='blue', linewidth=2)
    plt.fill_between(iterations, ep_mean - ep_ci, ep_mean + ep_ci, color='blue', alpha=0.15)

    # Formatting
    plt.title("MB-MPO Training Performance (Mean ± 90% CI)", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    
    # Save the plot in the target folder so it stays with the data
    output_plot_path = os.path.join(folder_path, "training_curve.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"Saved aggregated plot to {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_rl_curves_from_folder("./data")