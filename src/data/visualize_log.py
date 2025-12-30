import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_patchify_info(path):
    """_summary_
    Plot the information from the logging during satellite data extraction. 
    """
    with open(path) as f:
        log = json.load(f)

    rows = []
    for folder, stats in log.items():
        rows.append({"folder": folder, "kept": stats.get("kept", 0), "total": stats.get("total", 0), "covered": stats.get("percentage covered", 0)})

    df = pd.DataFrame(rows)
    df_summary = df.groupby("folder")[["kept","total"]].sum().reset_index()
    df_summary.plot(x="folder", y=["kept","total"], kind="bar", figsize=(10,4))
    plt.show()

if __name__ == "__main__":
    path = Path("image_patches/logging")
    plot_patchify_info(path)