# Project details

- [Project details](#project-details)
  - [Experiment setting](#experiment-setting)
  - [Configuration based building](#configuration-based-building)
    - [YAML custom tags](#yaml-custom-tags)
    - [YAML aliases - DRY](#yaml-aliases---dry)
    - [YAML merging - DRY #2](#yaml-merging---dry-2)
  - [Parameters](#parameters)
    - [Currently in use](#currently-in-use)
    - [Adviced by Numenta community](#adviced-by-numenta-community)

## Experiment setting

Quote from our paper:

In our experiments, we used classic grid world environments, which were represented as mazes on a square grid. Each state can be defined by an agent's position; thus, the state space $S$ contains all possible agent positions. An agent begins in a fixed state $s_0$, with the goal of locating a single reward in a fixed position $s_g$. The environment's transition function is deterministic. The action space is made up of four actions that move the agent to each adjacent grid cell. However, when the agent attempts to move into a maze wall, the position of the agent remains unchanged. It is assumed that the maze is surrounded by obstacles, making it impossible for an agent to move outside. Every timestep, an agent receives an observation -- a binary image of a small square window encircling it (we used $5 \times 5$ size with an agent being at its center). The observation has six binary channels: three color channels for the floor, walls, out-of-bounds obstacles, and the goal. We use maze floor coloring to add semantic clues to observations. When an agent achieves the goal state $s_g$, it receives a large positive value $+1.0$. Each step is also slightly punished with $-0.002$, causing an agent to seek the shortest path. The optimal path from the starting point to the goal in each testing environment was between 8 and 20 steps. We also set a time limit for episodes: 200 for an 8x8 environment and 350 for a 12x12 environment.

## Configuration based building

This section describes configs syntax, rules and how to use it.

We use [YAML 1.2](https://yaml.org/spec/1.2/spec.html) format to represent configs and parse them with [ruamel.yaml](https://yaml.readthedocs.io/en/latest/overview.html) python package.

If you're new to YAML, check out [design section](https://en.wikipedia.org/wiki/YAML#Design) on Wikipedia - it provides required basics of the format in a very short form.

___
_Here's a little side note covering all information sources:_

For more details on YAML format we encourage you to use [1.2 standard specification](https://yaml.org/spec/1.2/spec.html).

`ruamel.yaml` itself has a very shallow [documentation](https://yaml.readthedocs.io/en/latest/overview.html) which is not of a much use. But, since it's a fork of PyYAML package, PyYAML's [docs](https://pyyaml.org/wiki/PyYAMLDocumentation) are mostly applicable as well. Both packages have slightly different API and internals.

There're also some useful answers on Stackoverflow from the author of `ruamel.yaml` (mostly in questions on PyYAML). And the last bastion of truth is, of course, `ruamel.yaml` code sources.
___

The most important non-obvious features we use are:

- custom tags
- aliases
- merging

### YAML custom tags

Custom tags are used for value postprocessing. Consider following example:

```yaml
actor:
  first_name: Picolas
  last_name: Cage
```

Parsing this YAML will result in an `actor` being a dictionary `{first_name: "Picolas", last_name: "Cage"}`:

```python
from ruamel.yaml import YAML

text = """
actor:
  first_name: Picolas
  last_name: Cage
"""

yaml = YAML()
data = yaml.load(text)
actor = data['actor']
assert isinstance(actor, dict) and actor['first_name'] == 'Picolas'
```

What if we wanted both `first_name` and `last_name` to be lower case? Or what if we wanted an `actor` to be represented as an object (e.g. of custom class `Person` or just a plain string `'Picolas Cage'`)? That's exactly what custom tags allow you to do.

In order to use custom tag you should tell a parser both a tag and a callback function.

```python
from ruamel.yaml import YAML, BaseLoader

text = """
actor: !my_awesome_tag
  first_name: Picolas
  last_name: Cage
"""

tag = '!my_awesome_tag'
def awesome_callback(loader: ruamel.yaml.BaseLoader, node):
  actor_dict = loader.construct_mapping(node)
  fname = actor_dict['first_name']
  lname = actor_dict['last_name']
  return f'{fname} {lname}'

yaml = YAML()
yaml.constructor.add_constructor(tag, awesome_callback)   # tag registration

data = yaml.load(text)
assert data['actor'] == 'Picolas Cage'
```

Callback function should have specific interface as in the example above. `loader` has construction methods for all supported basic classes ('_mapping', '_sequence', '_scalar' and etc.). Custom tag should start with `!`; tags starting with `!!` are reserved and provided by the standard.

Building a custom class object out of the dictionary node is a popular case. Note that most of the time you don't need a complex factory method. Often you just want to pass a node dictionary as kwargs to an object `__init__` method:

```python
from ruamel.yaml import YAML, BaseLoader

class Person:
  def __init__(self, first_name, last_name):
    self.name = f'{first_name} {last_name}'

tag = '!create_person`
def create_person(loader: BaseLoader, node):
  kwargs = loader.construct_mapping(node)
  return Person(**kwargs)

yaml = YAML()
yaml.constructor.add_constructor(tag, create_person)
actor = yaml.load("""
actor: !create_person
  first_name: Picolas
  last_name: Cage
""")
assert isinstance(actor, Person) and actor.name == 'Picolas Cage'
```

By convention all tags are registered in `config.py` file with two methods:

- `register_static_methods_as_tags(cls, yaml)` - registers each static function `xyz` of the provided class as a callback to a tag `!xyz`
  - note all lower case symbols in tag name as function names conventionally lower cased
  - there's only one class with such callbacks - `TagProxies`
- `register_classes(yaml)` - registers specified list of classes. For each class `Xyz` it registers a tag `!Xyz`.
  - note upper case first letter in a tag as in class names conventionally camel cased
  - so, you can deduce the type of a tag directly from its name

So, if you don't know a tag a) look its definition if it has explicit callback or b) look corresponding class constructor.

### YAML aliases - DRY

Let's consider following example: you have to specify the same value multiple times in your config. DRY principle encourage you to specify this value only once and then reference to it any time further. That's what YAML aliases support means and allows you to do. Aliases are defined with `&` sign. To reference defined alias use `*` sign:

```yaml
random_seed: &seed 42   # alias named "seed" to the value 42 stored in "random_seed"
actor: &cage            # alias named "cage" to the dict stored in "actor"
  first_name: Picolas
  last_name: Cage

random_movie:
  main_actor: *cage     # reference to the dict stored in "actor"
  rnd_name_seed: *seed  # value 42
```

This feature provides you more centralized way to manage parameters. Alias is implemented as a copy-by-reference to the origin, i.e. everyone references to the same origin object. You can also use tags to build custom class objects and reference to them further in the document:

```yaml
actor: !Person &cage    # alias to the Person object stored in "actor"
  first_name: Picolas
  last_name: Cage

another_actor: !Person &ben
  first_name: Benedict
  last_name: Cucumberbatch
  meme_sibling: *cage   # reference to the Person object stored in "actor"
```

### YAML merging - DRY #2

This's useful for the following scenario - there're multiple nodes sharing the same subset of attributes. On solution is to make an alias to the subset and then _append_ or _extract_ it to each node. That's what merging feature allows you to do:

```yaml
.base_attrs: &base
  x: ...
  y: ...

object_one:
  <<: *base   # "base" node is extracted, i.e. "x" and "y" will be appended to this node
  z: ...

object_two:
  <<: *base
  w: ...
  u: ...
```

Note that this feature works only with dictionaries.
Another popular use case - to create the same distinct objects:

```yaml
.obj_x: &obj_x
  a: 1

obj_y:
  obj_x: !X
    <<: *obj_x    # extract init params and then create distinct object by custom class tag (line above)

obj_z:
  obj_x: !X
    <<: *obj_x
```

In the example above `obj_y` and `obj_z` will have different `obj_x` objects, although they will be initialized the same.

You may noticed here (or in repo configs) unusual node names starting with dot. We use following _root node_ naming convention:

- `xyz` - for scalars and objects that you will use at runtime in your code. That's the endpoints of the configuration process.
- `.xyz` - for auxilary dictionaries with initialization parameters used at parsing and object construction.

## Parameters

### Currently in use

**SA encoder**:

- `n_values`: (>=3, 3)
- `value_bits`: 16
- `activation_threshold`: 14
- Derived params or attributes
  - `total_bits`: >=48
  - `sparsity`: ~15-35%

Rationale

- `n_values`
  - defined by the number of unique states/observations
  - tried envs with n_values 3-100
- `value_bits`
  - advised >= 20
- `activation_threshold`
  - = _value_bits_ - 2
  - __highly important__
  - each part of SA (state and action) has their own activation_threshold = *value_bits* - 1 = 7
    - -1 is for noise or similar values (=12.5% of noise for 8 bit)
    - that's -2 in total for both parts
    - doubtful decision to sum up, maybe it's better to take max?
  - for now choosing actual value for threshold is not _that_ important, as we use encoder which maps different values to non-overlapped SDRs and everything is denoised.

**Temporal Memory**:

- `n_columns`: >= 48
- `cells_per_column`: 1 or ??
- `activation_threshold`: 14
- `learning_threshold`: 11
- `initial_permanence`: 0.5
- `connected_permanence`: 0.4
- `permanenceIncrement`: 0.1
- `permanenceDecrement`: 0.05
- `predictedSegmentDecrement`: 0.0001
- `maxNewSynapseCount`: 16
- `maxSynapsesPerSegment`: 16
- `maxSegmentsPerCell`: 4

Rationale:

- `n_columns`
  - = *sa.total_bits*
- `cells_per_column`
  - for MDP first-order memory is enough
  - for POMDP - don't know yet
- `activation_threshold`
  - = *sa.activation_threshold* = 14
- `learning_threshold`
  - lower bound is: *action.value_bits* = 8
    - it's a very common case to have this exact overlap
    - e.g. SAs $\{(x_i, 0)\}$ with fixed action all have 8 bits of overlap with each other
    - and we shouldn't penalize these pairs
    - hence 8 < *learning_threshold* < *activation_threshold*
- `initial_permanence`
  - initial permanence value after its creation
- `connected_permanence`
  - threshold defining whether a synapse is connected or not
    - only connected synapses can activate segments
  - the difference between initial and connected may be important
    - and also its relation with learning parameters
    - to make initial and connected to be equal is a good default strategy
  - needs investigation in general case
- `permanenceIncrement`
  - have non-zero importance, requires further investigation
- `permanenceDecrement`
  - recommended: 1/2 - 1/3 of an increment (sure? why?)
- `predictedSegmentDecrement`
  - recommended: _permanenceIncrement_ * sparsity
    - Numenta's logic behind it is not very applicable to our case ATM
    - as input vectors have skewed entropy
- `maxNewSynapseCount`
  - = *sa.value_bits* = 16
  - weak understanding of the importance
  - seems reasonable to set it to at least *activation_threshold*, such that each new segment could be activated right after creation (otherwise there won't be enough synapses)
  - obviously it's upper bounded with *maxSynapsesPerSegment*
- `maxSynapsesPerSegment`
  - = *sa.value_bits* = 16
  - **important parameter**
  - obvious lower bound: *activation_threshold*
  - not so obvious upper bound: *sar.value_bits*
    - each segment should recognize no more than one SA pattern, because otherwise they could interfer, which makes backtracking difficult
    - on the other hand it leads to a larger number of segments, i.e. TM just memorizes all transitions (atm that's true and desired)
    - should be investigated in future

### Adviced by Numenta community

**Spatial Pooler**:

- `n_colunms`: >= 2000
  - more is better
  - similarity metric - overlap
    - different "values" => low overlap score
    - similar "values" => high overlap score
  - so, there should be enough columns to distinguish levels of similarity, given some noise
- `sparsity`: 2%
  - how many bits are active
  - ok: 1-10%
  - `n_active_bits` should be >= 20
  - TODO: add equations from numenta paper

**Temporal Memory**:

- `n_columns`: >= 2000
  - same as for Spatial Pooler
- `cells_per_column`: 8
  - defines the number of different ways a context is represented (grows exponentially)
- `activation_threshold`
  - number of active synapses enough for segment activation
  - = *n_active_bits* - R
    - expected number of active columns
    - minus some accepted similarity radius R (or noise)
- `learning_threshold`
  - ??
- `initial_permanence`: 0.5
- `connected_permanence`: 0.5
- `permanenceIncrement`: 0.1
- `permanenceDecrement`: 2-4 times smaller than *permanenceIncrement*
- `predictedSegmentDecrement`: *activation_threshold* \* *sparsity*
  - used to punish on reaching *learning_threshold*
- `maxNewSynapseCount`: 32
- `maxSynapsesPerSegment`: 255
