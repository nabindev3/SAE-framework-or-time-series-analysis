import json
import os

def main():
    report_path = "eval/report.md"
    results_path = "probing/results/probe_results.json"
    
    if not os.path.exists(report_path):
        print(f"Report file {report_path} not found.")
        return
        
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found.")
        return
        
    with open(results_path, "r") as f:
        res = json.load(f)
        
    with open(report_path, "r") as f:
        content = f.read()
        
    def _fmt_delta(key):
        return (f"{res.get(key, 0):+.3f}  "
                f"(95% CI [{res.get(key+'_CI_lower', 0):+.3f}, "
                f"{res.get(key+'_CI_upper', 0):+.3f}])")

    # Replace the generic fillers
    content = content.replace("[FILL: P3−P1 point, 95% CI]", _fmt_delta("delta_sae"))
    content = content.replace("[FILL: P3−P2 point, 95% CI]", _fmt_delta("delta_sae_over_raw"))
    
    content = content.replace("[FILL: ~360k]", f"{res.get('n_train', 0) * 512}")
    
    content = content.replace("[FILL: n_total]", str(res.get('n_total', 0)))
    content = content.replace("n_train = [FILL]", f"n_train = {res.get('n_train', 0)}")
    content = content.replace("n_test =\n[FILL]", f"n_test =\n{res.get('n_test', 0)}")
    content = content.replace("n_test = [FILL]", f"n_test = {res.get('n_test', 0)}")
    content = content.replace("hard fraction (test) = [FILL]", f"hard fraction (test) = {res.get('hard_fraction', 0):.2f}")
    
    # Table replacements
    content = content.replace("| P1 stats     | [FILL]               |", f"| P1 stats     | {res.get('P1_AUROC', 0):.3f} ({res.get('P1_CI_lower', 0):.3f}, {res.get('P1_CI_upper', 0):.3f}) |")
    content = content.replace("| P2 stats+raw | [FILL]               |", f"| P2 stats+raw | {res.get('P2_AUROC', 0):.3f} ({res.get('P2_CI_lower', 0):.3f}, {res.get('P2_CI_upper', 0):.3f}) |")
    content = content.replace("| P3 stats+sae | [FILL]               |", f"| P3 stats+sae | {res.get('P3_AUROC', 0):.3f} ({res.get('P3_CI_lower', 0):.3f}, {res.get('P3_CI_upper', 0):.3f}) |")
    
    def _row(label, key):
        return (f"| {label} | {res.get(key, 0):+.3f} | "
                f"[{res.get(key+'_CI_lower', 0):+.3f}, "
                f"{res.get(key+'_CI_upper', 0):+.3f}] |")
    content = content.replace("| P2 − P1 (raw value)   | [FILL]| [FILL]  |",
                              _row("P2 − P1 (raw value)  ", "delta_raw"))
    content = content.replace("| P3 − P1 (sae value)   | [FILL]| [FILL]  |",
                              _row("P3 − P1 (sae value)  ", "delta_sae"))
    content = content.replace("| P3 − P2 (sae over raw)| [FILL]| [FILL]  |",
                              _row("P3 − P2 (sae over raw)", "delta_sae_over_raw"))
    
    with open(report_path, "w") as f:
        f.write(content)
        
    print(f"Populated {report_path} with results.")

if __name__ == "__main__":
    main()
