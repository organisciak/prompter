from numpy.random import default_rng, randint
import numpy as np
import pandas as pd
from pathlib import Path
import pkg_resources
import ipywidgets as widgets
from IPython.display import display
import os

DATA_PATH = Path(pkg_resources.resource_filename(__name__, 'data'))

rng = default_rng()

class PromptSampler():
    def __init__(self, terms, weights=None, name=None):
        self.terms = terms
        if (type(self.terms) is str):
            # assume comma separated
            self.terms = [x.strip() for x in self.terms.split(',')]
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
            choices = rng.choice(self.terms, size=min(n, len(self)), replace=False)
            return PromptSampler(choices.tolist())

class Prompter():
    cache = dict()
    ref = { 'aweadj': DATA_PATH/'awe_adj.csv',
            'videojunk': DATA_PATH/'videojunk_tags.csv',
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

    def parse_interrogations(self, fpath, name=None):
        ''' Parse an output data_desc.csv-style file from CLIP interrogator and return two PromptSampler: one
        for the BLIP settings, and one for the CLIP styles.'''
        blips = []
        clips = []

        if not name:
            name = fpath.parent.name

        for prompt in pd.read_csv(fpath).prompt:
            blip, clip = prompt.strip().split(', ', 1)
            blips.append(blip)
            clips += [c.strip() for c in clip.split(',')]

        self.cache[f"{name}_scenes"] = PromptSampler(blips)
        self.cache[f"{name}_styles"] = PromptSampler(clips)
        
        return self.cache[f"{name}_scenes"], self.cache[f"{name}_styles"]

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

class ImageOutHandler():

    def __init__(self, outdir=None):
        '''
        A class for handling generated images in a notebook. Handles file naming and saving,
            saving prompts to a file, and having widgets for deleting saved images (or saving unsaved images)
            on demand.

        If you manually delete images, the prompts.txt file can be analyzed to see which prompt
            tags were associated with deleted images.
        '''
        self.outdir = outdir
        # save images in memory for when ask=True
        self.save_queue = {}

    def _clean_prompt(self, prompt):
        '''
        Simple cleaning for a filename
        '''
        return "".join([x for x in list(prompt.replace(" ", "_")) if x.isalpha() or (x == "_")])

    def reset(self):
        self.save_queue = {}

    def _save_im_btn(self, b):
        fname = b.description[5:-1]
        im = self.save_queue[fname]
        print("saving", fname)
        im.save(fname)
        b.disabled = True

    def _delete_im_btn(self, b):
        fname = b.description[7:-1]
        print("deleting", fname)
        if Path(fname).is_file():
            os.remove(fname)
        else:
            print("Failed to delete")
        b.disabled = True

    def display_ims(self, images, prompts, fname_suffix=None, unsafe_detected=None, save=True, print_ims=True, 
                    ask=False, save_prompts=True):
        '''
        Handled generated images.

        images: list of PIL images.
        prompts: prompt string, or list of strings
        unsafe_detected: safety_checker judgments, if available.

        save: save images to disk (ImageOutHandler.outdir)
        save_prompts: save image name and prompt to prompts.txt. Only if save=True.
        fname_suffix: additional identifier. Else a random int is appended.
        print_ims: show images in notebook. Usually you want this on, unless you're generating a lot at once.

        ask: Buttons for on-demand commands. If save=False, offer a save button. If save=True, offer a delete button.
        '''
        # some models don't return the safety checks.
        if unsafe_detected is None:
            unsafe_detected = [False] * len(images)
        
        if type(prompts) == str:
            prompts = [prompts] * len(images)

        if not save:
            save_prompts = False

        if not fname_suffix:
            fname_suffix = str(randint(0, 2147483647))

        for i, (image, unsafe, prompt) in enumerate(zip(images, unsafe_detected, prompts)):
            if unsafe:
                print('skipping for content filter')
                continue
        
            clean_prompt = self._clean_prompt(prompt)
            fname = self.outdir / f"{clean_prompt[:20]}-{fname_suffix}-{i}.png"

            if print_ims:
                display(image)
                print(prompt)

            if save:
                image.save(fname)
                if ask:
                    button = widgets.Button(description=f"Delete {fname}?")
                    display(button)
                    button.on_click(self._delete_im_btn)
                    print('-----')
            elif ask:
                self.save_queue[str(fname)] = image
                button = widgets.Button(description=f"Save {fname}?")
                display(button)
                button.on_click(self._save_im_btn)
                print('-----')

            if save_prompts:
                with open(self.outdir / "prompts.txt", mode='a') as f:
                    f.write(f"{fname}\t{prompt}\n")