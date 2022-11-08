from numpy.random import default_rng
import numpy as np
import pandas as pd
from pathlib import Path
import pkg_resources
DATA_PATH = Path(pkg_resources.resource_filename(__name__, 'data'))

rng = default_rng()

class PromptSampler():
    def __init__(self, terms, weights=None, name=None):
        self.terms = terms
        if (type(self.terms) is np.ndarray):
            self.terms = self.terms.tolist()
        self.weights = None
        if weights is not None:
            self.weights = np.array(weights)
        self.name = name

    def __len__(self):
        return len(self.terms)

    def __str__(self):
        return ", ".join(self.terms)

    def str(self):
        ''' For convenient casting if needed. '''
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if type(other) is list:
            other = PromptSampler(other)

        assert type(other) == type(self)
        if (type(self.weights) is np.ndarray) and (type(other.weights) is np.ndarray):
            weights = np.concatenate([self.weights, other.weights])
        else:
            # if only one sampler has weights, then drop weights
            weights = None
        return PromptSampler(self.terms+other.terms, weights=weights)

    def __getitem__(self, n):
        ''' Shorthand for sample'''
        return self.sample(n)

    def sample(self, n=None, ignore_weights=False):
        if n == None:
            n = len(self)
        if (type(self.weights) is np.ndarray):
            p = None if ignore_weights else self.weights/sum(self.weights)
            x = rng.choice(list(zip(self.terms, self.weights)), n, replace=False, p=p)
            return PromptSampler(x[:,0].tolist(), x[:,1])
        else:
            choices = rng.choice(self.terms, size=n, replace=False)
            return PromptSampler(choices.tolist())

class Prompter():
    cache = dict()
    ref = {'videojunk': DATA_PATH/'videojunk_tags.csv',
            'serene': DATA_PATH/'serene_settings.csv'
            }

    def __init__(self):
        pass

    def __getitem__(self, key):
        try:
            if key in self.ref.keys():
                return self._load_dataset(self.ref[key])
            else:
                # try to load as a path, or a custom cache key if added
                return self._load_dataset(key)
        except KeyError:
            raise KeyError(f"'{key}' not seen in class reference, filesystem, or cache")

    def add_csv(self, name, path):
        self.ref[name] = path
        return self[name]

    def add_terms(self, name, terms, weights=None):
        ps = PromptSampler(terms, weights)
        self.cache[name] = ps
        return self[name]

    def _load_dataset(self, path):
        if str(path) not in self.cache:
            df = pd.read_csv(path)
            weights = df.weight if 'weight' in df.columns else None
            ps = PromptSampler(df.prompt.tolist(), weights)
            self.cache[str(path)] = ps
        return self.cache[str(path)]