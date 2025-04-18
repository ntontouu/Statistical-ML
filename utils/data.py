from dataclasses import dataclass
import re
import csv

import numpy as np

def arff_data(text):
    data_match = re.search(r'(?ims)^\s*@data\s*$(.*?)(?=^\s*@|\Z)', text, re.MULTILINE | re.IGNORECASE)
    if data_match:
        data = data_match.group(1).strip()
        data_lines = [line.strip() for line in data.split('\n') if line.strip() and not line.strip().startswith('%')]
        lines = csv.reader(data_lines)
        dataset = list(lines)
        return dataset

    return None

@dataclass
class data:
    def load_arff(self, f_location:str):
        """Load data from an arff file"""

        with open(f_location, 'r', encoding='utf-8') as _file:
            txt = _file.read()
            data = arff_data(txt)
        _file.close()

        features = []
        labels = []

        for row in data:
            features.append(
                [float(value) for value in row[:-1]]
            )
            labels.append(row[-1])

        unique_labels = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        enumerated_labels = [unique_labels[label] for label in labels]

        self.features = np.array(features)
        self.labels = enumerated_labels

    def load_csv(self, f_location: str):
        """Load CSV data from a file."""
        with open(f_location, 'r', encoding='utf-8') as _file:
            lines = csv.reader(_file)
            data = list(lines)
            
            self.attributes = data[0]
            data = data[1:]
        _file.close()

        features = []
        labels = []

        for row in data:
            features.append([float(value) for value in row[:-1]])
            labels.append(row[-1])

        unique_labels = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        enumerated_labels = [unique_labels[label] for label in labels]

        self.features = np.array(features)
        self.labels = enumerated_labels




