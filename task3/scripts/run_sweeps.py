import os
import subprocess
import json

def run_sweeps():
    data_root = "data/PACS" # Default fallback
    import sys
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
        
    print(f"Running sweeps with data_root={data_root}")
    
    # DAN-DG lambda_dg sweep
    lambda_dgs = [0.1, 1.0, 10.0]
    for l_dg in lambda_dgs:
        print(f"\n--- Running DAN-DG sweep for lambda_dg={l_dg} ---")
        
        # Train
        train_cmd = [
            "python", "task3/train.py", 
            "--config", "task3/configs/dan_dg.yaml", 
            "--data_root", data_root,
            "--lambda_dg", str(l_dg)
        ]
        subprocess.run(train_cmd, check=True)
        
        # We rename the checkpoint and csv so they don't get overwritten
        os.rename("task3/results/dan_dg_best_checkpoint.pth", f"task3/results/dan_dg_lam{l_dg}_best.pth")
        
        # Evaluate
        eval_cmd = [
            "python", "task3/evaluation/evaluate_sketch.py",
            "--checkpoint", f"task3/results/dan_dg_lam{l_dg}_best.pth",
            "--data_root", data_root
        ]
        subprocess.run(eval_cmd, check=True)
        
    # SAM rho sweep
    rhos = [0.01, 0.05, 0.1]
    for rho in rhos:
        print(f"\n--- Running SAM sweep for rho={rho} ---")
        
        # Train
        train_cmd = [
            "python", "task3/train.py", 
            "--config", "task3/configs/sam.yaml", 
            "--data_root", data_root,
            "--rho", str(rho)
        ]
        subprocess.run(train_cmd, check=True)
        
        # Rename checkpoint
        os.rename("task3/results/sam_best_checkpoint.pth", f"task3/results/sam_rho{rho}_best.pth")
        
        # Evaluate
        eval_cmd = [
            "python", "task3/evaluation/evaluate_sketch.py",
            "--checkpoint", f"task3/results/sam_rho{rho}_best.pth",
            "--data_root", data_root
        ]
        subprocess.run(eval_cmd, check=True)
        
    print("\nAll sweeps completed successfully! Check task3/results/sketch_results.json for the final summary.")

if __name__ == '__main__':
    run_sweeps()

