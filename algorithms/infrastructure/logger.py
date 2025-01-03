from datetime import datetime
import os
import os.path as osp
import numpy as np
import pandas as pd


class Logger:
    def __init__(self, directory=None):
        self.filename = None
        self.first_write = True
        self.keys = []
        self.row = {}

        self._init_dir(directory)

    def _init_dir(self, dirname):
        if dirname is None:
            current_timestamp = datetime.now()
            timestamp = current_timestamp.strftime('%Y-%m-%d-%H:%M:%S')
            dirname = "logs-" + timestamp + "/"

        if osp.exists(dirname):
            print("Warning, logging dir already exists, continuing" )
        else:
            os.makedirs(dirname)
        filepath = os.path.join(dirname, "logs.csv")
        open(filepath, 'w')
        self.filename = filepath

    def write(self, data, step):
        """
            Accepts a dictionary of key : scalar pairs as well as the current
            (training/evaluation) step.
        """

        def get_value(key):
            val = data.get(key, "")

            if isinstance(val, (float, np.floating)):
                val_str = f"{val:.3f}"
            else:
                val_str = str(val)
            return val_str

        if self.first_write:
            self.keys = data.keys()
            with open(self.filename, 'a') as file:
                file.write("step;" + ";".join(map(str, self.keys)))
                file.write("\n")
            self.first_write = False

        for key in data.keys():
            assert(key in self.keys), f"Data key {key} does not match the csv keys {self.keys}"

        val_strings = [ get_value(key) for key in self.keys ]
        print(f"Step {step} - ", end="")
        for key in self.keys:
            print(f"{key} : {get_value(key)},", end=" ")
        print()
        with open(self.filename, 'a') as file:
            file.write(f"{step};" + ";".join(val_strings))
            file.write("\n")

    def export_df(self):
        """
            Exposes the contents of the csv file as a pandas dataframe.
        """
        return pd.read_csv(self.filename, sep=';')

        

