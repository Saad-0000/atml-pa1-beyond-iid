import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from task1.configs import configs

def plot_translation_curve(results, output_dir):
    """Plots translation displacement vs prediction consistency."""
    plt.figure(figsize=(8, 6))
    deltas = [0, 8, 16, 32]
    markers = {"ResNet-50_head": "o-", "ViT-B/16_head": "s-", "CLIP-ViT-B/32_head": "^-", "CLIP-ViT-B/32_zeroshot": "d--"}
    
    for model_name, data in results.items():
        if "translation_sweep" not in data:
            continue
            
        consistencies = []
        for d in deltas:
            key = f"delta_{d}"
            # At delta=0, consistency with itself is exactly 1.0
            val = data["translation_sweep"][key]["mean_consistency"] if d != 0 else 1.0
            consistencies.append(val)
            
        label = model_name.replace("_head", " (Trained)").replace("_zeroshot", " (Zero-shot)")
        marker = markers.get(model_name, "o-")
        plt.plot(deltas, consistencies, marker, label=label, linewidth=2, markersize=8)

    plt.title("Translation Equivariance: Consistency vs. Displacement")
    plt.xlabel("Displacement (pixels)")
    plt.ylabel("Prediction Consistency (Fraction)")
    plt.xticks(deltas)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "translation_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved translation curve to {save_path}")

def plot_accuracy_drops(results, output_dir):
    """Plots a grouped bar chart for accuracy drops under perturbations."""
    models = list(results.keys())
    interventions = ["grayscale", "hue_rotation", "patch_shuffle"]
    
    # Setup bar chart
    x = np.arange(len(models))
    width = 0.25
    plt.figure(figsize=(10, 6))
    
    for i, intervention in enumerate(interventions):
        drops = [results[m][intervention]["accuracy_drop"] for m in models]
        offset = (i - 1) * width
        plt.bar(x + offset, drops, width, label=intervention.replace("_", " ").title())

    plt.title("Robustness: Accuracy Drop Under Interventions")
    plt.xlabel("Model")
    plt.ylabel("Accuracy Drop (Relative to Clean)")
    
    formatted_models = [m.replace("_head", "\n(Trained)").replace("_zeroshot", "\n(Zero-shot)") for m in models]
    plt.xticks(x, formatted_models)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "accuracy_drops_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved accuracy drops bar chart to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default=os.path.join(configs.OUTPUT_DIR, "bias_evaluation_results.json"))
    parser.add_argument("--output_dir", type=str, default=configs.OUTPUT_DIR)
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: {args.results_file} not found. Run run_task1.py first.")
    else:
        with open(args.results_file, "r") as f:
            results = json.load(f)
            
        plot_translation_curve(results, args.output_dir)
        plot_accuracy_drops(results, args.output_dir)