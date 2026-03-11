import json
import os
from datetime import datetime

def log_details(info, accuracy, best_model):
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

   
    metric_name = "accuracy" if info["task"] == "classification" else "r2_score"
    
    log_data={
        "dataset": info['dataset_name'],
        "timestamp": datetime.now().isoformat(),
        "dataset_samples": info['num_samples'],
        "dataset_features": info['num_features'],
        "task":info["task"],
        metric_name: accuracy,
        "best_model": best_model
    }

    file="logs/logs.json"

    if os.path.exists(file):
        data=json.load(open(file))
    else:
        data=[]
    
    data.append(log_data)
    with open(file, "w") as f:
        json.dump(data, f, indent=4)