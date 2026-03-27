import pandas as pd


class CSV_Logger:
    def __init__(self, path: str):
        self.df = pd.DataFrame()
        self.currentLog = dict()
        self.path = path

    def log(self, key: str, value: float):
        self.currentLog[key] = value
        return

    def flush(self):
        new_row = pd.DataFrame([self.currentLog])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.currentLog.clear()

    def export_csv(self):
        self.df.to_csv(self.path + ".csv")
