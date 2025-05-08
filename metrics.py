import pandas as pd
class MetricsTracker:
    def __init__(self):
        self.results = []

    def store(self, mode, metrics):
        self.results.append({
            "Mode": mode,
            "Balanced Accuracy": metrics.get("Balanced Accuracy"),
            "Accuracy": metrics.get("Accuracy"),
            "Precision": metrics.get("Precision"),
            "Recall": metrics.get("Recall"),
            "F1": metrics.get("F1")
        })

    def to_dataframe(self):
        return pd.DataFrame(self.results)

    def save_to_csv(self, model_name):
        df = self.to_dataframe()
        name = model_name.split("/")[-1]
        df.to_csv(f"results_{name}.csv", index=False)
        print(f"Results saved to results_{name}.csv")
