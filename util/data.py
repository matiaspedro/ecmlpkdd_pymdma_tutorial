import pandas as pd

def load_wine(path: str = None):
    if path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        data = pd.read_csv(url, header=None)
        data.columns = [
            "Class",
            "Alcohol",
            "MalicAcid",
            "Ash",
            "AlcalinityOfAsh",
            "Magnesium",
            "TotalPhenols",
            "Flavanoids",
            "NonflavanoidPhenols",
            "Proanthocyanins",
            "ColorIntensity",
            "Hue",
            "OD280/OD315OfDilutedWines",
            "Proline",
        ]
    else:
        data = pd.read_csv(path)

    return data


def load_seeds(path: str = None):
    if path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        data = pd.read_csv(url, header=None, delim_whitespace=True)
        data.columns = [
            "area",
            "perimeter",
            "compactness",
            "length_of_kernel",
            "width_of_kernel",
            "asymmetry_coefficient",
            "length_of_kernel_groove",
            "class",
        ]
    else:
        data = pd.read_csv(path)
    return data


def load_adult(path: str = None):
    if path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        data = pd.read_csv(url, header=None)
        data.columns = [
            "Age",
            "Workclass",
            "fnlwgt",
            "Education",
            "Education-Num",
            "Marital-Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital-gain",
            "Capital-loss",
            "Hours-per-week",
            "Native-Country",
            "Class",
        ]
    else:
        data = pd.read_csv(path)
    return data
