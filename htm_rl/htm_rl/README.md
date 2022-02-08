# Project details

- [Project details](#project-details)
  - [Experiment setting](#experiment-setting)
  - [Configuration based building](#configuration-based-building)
    - [YAML custom tags](#yaml-custom-tags)
    - [YAML aliases - DRY](#yaml-aliases---dry)
    - [YAML merging - DRY #2](#yaml-merging---dry-2)

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
