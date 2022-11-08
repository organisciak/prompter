## Prompter

A basic prompt helper for text-to-image prompt randomization.


```python
!pip install git+https://github.com/organisciak/prompter
```

The `Prompter` class loads datasets of prompt settings, style modifiers, etc.


```python
from prompter import Prompter, PromptSampler
```


```python
p = Prompter()
p['serene']
```




    a quiet forest, a shooting star over a mountain, a still lake at sunset, a babbling brook in a forest, a field of flowers in bloom, a winding country road, a snow-capped mountain range, a sandy beach with palm trees, a beautiful beach, a sunny meadow, a snowy mountain reflected in a frozen lake, a cityscape at dusk, a family of ducks swimming in a pond, a serene lake with a small waterfall, a beautiful garden, a field of wildflowers, a river winding through a lush forest, a country cottage, a cottage in a lush forest, a river winding through a valley, a waterfall in a jungle, a hot air balloon floating over a field of wildflowers, a castle on a hill, a winding country road, a deserted island, a lighthouse on a rocky cliff, a campfire under a bright galaxy and stars, a couple in silhouette walking hand in hand along a beach at sunset, a single tree in a field of tall grass, a herd of deer in a forest



Prompts are returned as `PromptSampler` instances. From there, they can be sampled or combined.


```python
ps = p['serene']
ps.sample(1)
```




    a cityscape at dusk



Sampling can be done with an integer key:


```python
ps[2]
```




    a country cottage, a field of flowers in bloom



Combining is done through addition:


```python
vj = p['videojunk']
ps[1] + vj[3]
```




    a winding country road, databending, bright saturated colours, holography



The datasets here are just ones I use to keep other code organized.

If you want to add more, you can use Prompter's `add_csv(name, path_to_csv)` or `add_terms(name, terms)`. Note that these add as class variables, not instance variables, so they're accessible for all `Prompter`.  You can also make a prompt sampler directly with `PromptSampler(terms, weights=None)`. Adding weights adds weighted sampling.


```python
ps = PromptSampler(['a','b','c', 'd', 'e'], weights=[1, 20, 1, 1, 1])
print(ps.sample(2)) # likely to include 'b' because of weight
print(ps.sample(2, ignore_weights=True)) # uniform distribution
```

    b, e
    c, e


Added `PromptSampler` instances return another `PromptSampler`, so you can construct bigger classes for tidiness.


```python
settings = p['serene'] + ['a cityscape at dusk']
styles = PromptSampler(['by Hayao Miyazaki'])
modifiers = p['videojunk'] + ['trending on artstation'] # just some tags to add some grime
settings[1] + styles[1] + modifiers[2]
```




    a still lake at sunset, by Hayao Miyazaki, rainbow fur, inspired by jean moebius giraud


