import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob
import os

def get_aggregated_data(folder_path):
    """Helper function to load and aggregate CSVs from a specific folder."""
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return None

    search_path = os.path.join(folder_path, "*.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No CSV files found in folder: '{folder_path}'")
        return None

    print(f"Found {len(files)} CSV files in '{folder_path}'. Aggregating...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'iteration' in df.columns and 'ep_reward' in df.columns:
                dfs.append(df)
            else:
                print(f"Skipping {os.path.basename(f)}: Missing required columns.")
        except Exception as e:
            print(f"Could not read {os.path.basename(f)}: {e}")

    if not dfs:
        print(f"No valid metric CSV files to plot in '{folder_path}'.")
        return None
    
    # Extract iterations from the first valid file
    iterations = dfs[0]['iteration'].values
    
    # Extract rewards into 2D arrays (shape: num_runs x num_iters)
    ep_rewards = np.array([df['ep_reward'].values for df in dfs])
    
    # Calculate Mean
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

    return iterations, ep_mean, ep_ci, n_runs

def plot_model_comparison(folder1, label1, folder2, label2, output_folder="."):
    """Plots and compares learning curves from two different model folders."""
    
    data1 = get_aggregated_data(folder1)
    data2 = get_aggregated_data(folder2)

    if not data1 and not data2:
        print("No valid data found to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Plot Model 1
    if data1:
        iters1, mean1, ci1, n1 = data1
        plt.plot(iters1, mean1, label=f"{label1}", color='blue', linewidth=2)
        plt.fill_between(iters1, mean1 - ci1, mean1 + ci1, color='blue', alpha=0.15)

    # Plot Model 2
    if data2:
        iters2, mean2, ci2, n2 = data2
        plt.plot(iters2, mean2, label=f"{label2}", color='orange', linewidth=2)
        plt.fill_between(iters2, mean2 - ci2, mean2 + ci2, color='orange', alpha=0.15)

    # Formatting
    plt.title("MB-MPO Models Comparison (Mean ± 90% CI)", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_plot_path = os.path.join(output_folder, "model_comparison_curve.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"\nSaved comparison plot to {output_plot_path}")
    
    plt.show()

if __name__ == "__main__":
    # Example usage: Replace these with your actual folder paths and labels
    plot_model_comparison(
        folder1="./data", label1="Baseline", 
        folder2="./disagree",  label2="Disagreement",
        output_folder="."
    )