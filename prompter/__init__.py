from numpy.random import default_rng, randint
import numpy as np
import pandas as pd
from pathlib import Path
import pkg_resources
import ipywidgets as widgets
from IPython.display import display
import os
import re
import json

DATA_PATH = Path(pkg_resources.resource_filename(__name__, 'data'))

rng = default_rng()

class PromptSampler():
    def __init__(self, terms, weights=None, name=None, description=None):
        '''
        Terms - the terms to use. If a list, used directly. 
                If a string, will split on commas andstrip whitespace.
                If a dict, expects {'terms':[], 'weights':[], 'name':''}, where only terms is required.
                If a path, assumes json
        '''
        self.terms = terms
        if (type(self.terms) is str):
            # assume comma separated
            self.terms = [x.strip() for x in self.terms.split(',')]
        elif (type(self.terms) is np.ndarray):
            self.terms = self.terms.tolist()
        elif isinstance(self.terms, Path) and self.terms.name.endswith('.json'):
            with open(self.terms) as f:
                self.terms = json.load(f)

        self.weights = None
        if self.weights is not None:
            self.weights = np.array(weights)
        self.name = name
        self.description = description

        if type(self.terms) is dict:
            self.terms, w, n, d = self._from_dict(self.terms)
            # don't overwrite manually provided weights and names
            if self.weights is None:
                self.weights = w
            if self.name is None:
                self.name = n
            if self.description is None:
                self.name = d

    def __len__(self):
        return len(self.terms)

    def __str__(self, sep=', '):
        return sep.join(self.terms)

    def str(self, sep=', '):
        ''' For convenient casting if needed. '''
        return self.__str__()

    def to_dict(self):
        return dict(terms=self.terms, weights=self.weights, name=self.name)

    def to_json(self, fname=None, mode='w'):
        if fname is None:
            return json.dumps(self.to_dict())
        else:
            with open(fname, mode=mode) as f:
                json.dump(self.to_dict(), f, indent=True)

    def _from_dict(self, indict):
        terms = indict.get('terms', [])
        assert len(terms), "Dict didn't have any terms in it"
        weights = indict.get('weights', None)
        name = indict.get('name', None)
        desc = indict.get('description', None)
        return terms, weights, name, desc

    def _from_json(self, injson):
        if '{' not in injson.strip():
            # assume a filepath
            with open(injson) as f:
                data = json.load(f)
        else:
            data = json.loads(injson)
        terms, weights, name, desc = self._from_dict(data)
        return terms, weights, name, desc


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
    
    cache = {}
    ref = { 'aweadj': DATA_PATH/'awe_adj.json',
            'videojunk': DATA_PATH/'videojunk_tags.json',
            'serene': DATA_PATH/'serene_settings.json'
            }

    def __init__(self, name=None, description=None, templates=[]):
        '''
        templates: optional list of templates, or list of dicts with a 'template' key
        '''
        if len(templates) and type(templates[0]) is not dict:
            templates = [{'template':template for template in templates}]
    
        self.templates = templates
        self.name = name
        self.description = description

        # Auto-populate data dir files which aren't explicitly notes, using filename
        # this really doesn't need to be in an instance _init_ since it's a class variable
        for fpath in DATA_PATH.glob('*json'):
            if fpath not in Prompter.ref.values():
                Prompter.ref[fpath.stem] = fpath

    def __getitem__(self, key):
        try:
            if key in self.cache.keys():
                return self.cache[key]
            elif key in self.ref.keys():
                path = self.ref[key]
                return self._load_dataset(path, name=key)
            else:
                # try to load as a path, or a custom cache key if added
                return self._load_dataset(key)
        except KeyError:
            raise KeyError(f"'{key}' not seen in class reference, filesystem, or cache")
        
    def template(self, template=0, fill=[]):
        '''
        Construct from a template, where portions to replace are marked with double curly braces.

        template: A template string, or an integer referring to a cached template.

        Refer to saved PromptSamplers by name in the curly braces - optionally with square brackets to
        refer to number to sample. If curly braces are empty, what to put in is pulled from the list
        in the fill arg, in order.
        
        e.g.
            `Prompt.template("{{}} {{serene[1]}}, {{videojunk}}", ["A poster of"])`

            will fill 'A poster of' followed by one term from the 'serene' prompt list and
            all videojunk terms.
        '''
        if type(template) is int:
            template = self.templates[0]['template']

        # make into a formal template
        template_str, sampler_args = self._parse_template(template)

        fill_ind = 0
        if type(fill) is str:
            fill = [fill]

        fargs = []
        for name, count in sampler_args:
            if name == '':
                fargs.append(fill[fill_ind])
                fill_ind += 1
            elif count is None:
                fargs.append(self[name])
            else:
                fargs.append(self[name][count])

        return template_str.format(*fargs)

    def _parse_template(self, template):
        # make into a formal template
        template_str = re.sub(r"{{.*?}}", r"{}", template)
        samplerstrings = re.findall(r"{{(.*?)}}", template)

        sampler_args = []
        for substr in samplerstrings:
            substr = substr.strip()
            if substr == '':
                sampler_args.append((None, None))
            else:
                name, count = re.search(r"^(.*?)\[?(\d*)\]?$", substr).groups()
                if count != '':
                    count = int(count)
                else:
                    count = None
                sampler_args.append((name, count))

        return template_str, sampler_args

        

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

    ## Growing PS list ##
    def add_terms(self, name, terms, weights=None):
        ''' Initialize a prompt sampler and add to class cache.'''
        ps = PromptSampler(terms, weights)
        self.cache[name] = ps
        return self[name]

    def add(self, promptsampler, name=None):
        ''' Add already initialized PromptSampler to class cache. '''
        if name is None:
            assert promptsampler.name is not None, "Need a name either in PromptSampler instance or supplied by arg"
            name = promptsampler.name
        self.cache[name] = promptsampler
        return self[name]

    def add_csv(self, name, path):
        self.ref[name] = path
        return self[name]

    ## I/O ##
    def _load_dataset(self, path, name=None):
        if name is None:
            name = Path(path).stem
        
        if name not in self.cache.keys():
            path = Path(path)
            ps = PromptSampler(path)
            self.cache[name] = ps
            self.ref[name] = str(path)
        
        return self.cache[name]

    def to_dict(self, name=None, description=None, cache_keys=None):
        '''Serialize cache to dictionary. If cache_keys provided, only save certain keys'''
        if cache_keys is None:
            cache_keys = self.cache.keys()
        serial = dict(
            name=name,
            description=description,
            templates=self.templates,
            args={name: ps.to_dict() for name, ps in self.cache.items() if name in cache_keys}
        )
        return serial

    def to_json(self, fname=None, name=None, description=None, cache_keys=None, mode='w'):
        if name is None:
            name = self.name
        if description is None:
            description = self.description
        outdict = self.to_dict(name, description, cache_keys)
        if fname is None:
            return json.dumps(outdict)
        else:
            with open(fname, mode=mode) as f:
                json.dump(outdict, f, indent=True)

    def read_json(self, fname):
        with open(fname, mode='r') as f:
            indict = json.load(f)
        return self.from_dict(indict)

    def from_dict(self, indict):
        if self.name is None:
            self.name = indict.get('name', None)
        if self.description is None:
            self.description = indict.get('description', None)
        
        self.templates += indict.get('templates', [])
        for name, psdict in indict.get('args', []).items():
            ps = PromptSampler(psdict)
            self.add(ps, name=name)

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