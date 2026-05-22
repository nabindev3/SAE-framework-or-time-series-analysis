import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_metadata", type=str, default="activations/ETTh1_metadata.parquet")
    parser.add_argument("--base_metadata", type=str, default="activations_base/ETTh1_metadata.parquet")
    parser.add_argument("--probe_scores", type=str, default="activations/probe_scores.parquet")
    parser.add_argument("--output_dir", type=str, default="eval/results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading metadata...")
    try:
        df_small = pd.read_parquet(args.small_metadata)
        df_probe = pd.read_parquet(args.probe_scores)
        df_base = pd.read_parquet(args.base_metadata)
    except FileNotFoundError as e:
        print(f"Cannot run cascade eval yet: {e}")
        print("Deferred: Requires full extraction of both small and base backbones, and a valid probe run.")
        return
        
    if 'pred_P3_InputStats_SAE' not in df_probe.columns:
        print("Probe scores do not contain valid predictions. Run full training and probing first.")
        return
        
    df_eval = pd.merge(df_probe[['window_id', 'pred_P3_InputStats_SAE']], 
                       df_small[['window_id', 'crps_raw']], 
                       on='window_id', suffixes=('', '_small'))
                       
    df_eval = pd.merge(df_eval, 
                       df_base[['window_id', 'crps_raw']], 
                       on='window_id', suffixes=('_small', '_base'))
                       
    if len(df_eval) == 0:
        print("No test data available for cascade evaluation.")
        return
        
    print(f"Evaluating cascade on {len(df_eval)} test windows.")
    
    cost_small = 1.0
    cost_base = 5.0
    
    thresholds = np.linspace(0, 1.0, 50)
    
    pareto_points = []
    for t in thresholds:
        route_to_base = df_eval['pred_P3_InputStats_SAE'] >= t
        final_crps = np.where(route_to_base, df_eval['crps_raw_base'], df_eval['crps_raw_small'])
        mean_crps = np.mean(final_crps)
        mean_cost = np.mean(np.where(route_to_base, cost_base, cost_small))
        pareto_points.append((mean_cost, mean_crps))
        
    pareto_points = np.array(pareto_points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'bo-', label='SAE-Routed Cascade')
    plt.scatter([cost_small], [df_eval['crps_raw_small'].mean()], color='green', s=100, zorder=5, label='Always Small')
    plt.scatter([cost_base], [df_eval['crps_raw_base'].mean()], color='red', s=100, zorder=5, label='Always Base')
    
    plt.xlabel('Average Inference Cost (Arbitrary Units)')
    plt.ylabel('Average CRPS (Lower is better)')
    plt.title('Cascade Pareto Frontier: SAE Feature Routing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.output_dir, 'pareto_frontier.png')
    plt.savefig(plot_path)
    print(f"Saved Pareto frontier plot to {plot_path}")
    
if __name__ == "__main__":
    main()
