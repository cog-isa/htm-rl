# Research and development log

- [Research and development log](#research-and-development-log)
  - [TODO](#todo)
  - [Thoughts and ideas](#thoughts-and-ideas)
  - [2020.08.08 Sat](#20200808-sat)
    - [Gridworld transfer learning experiment](#gridworld-transfer-learning-experiment)
    - [TM related interesting links](#tm-related-interesting-links)
  - [2020.09.06 Sun](#20200906-sun)
  - [2020.09.18 Fri](#20200918-fri)
  - [2020.09.25 Fri](#20200925-fri)

## TODO

**Urgent**:

- [x] add ruamel.yaml to requirements
- [ ] describe config based building details
  - conventions
  - implementation
  - patches
- [ ] describe run arguments
- [x] update FAQ part on Terminology, Encoding, Planning, Parameters
- [ ] add pytorch, tdqm to requirements
- [ ] mention vs code setup for markdown

Research + functional tasks

- [x] Adapt planning to goal-based strategy
  - [x] Switch from reward-based planning to goal-based
    - [x] Cut out reward from encoding and memorizing
    - [x] Track history of rewarding states and plan according to any of them
      - add naive list-based rewarding states tracking
  - [x] Test transfer learning capabilities
    - [x] Adapt environments for random initial states
    - [x] Adapt environments for random rewarding states
    - [x] Make the set of testing environments
    - [x] Adapt test runners for a multi-environment tests
    - [x] Make config for an experiment
    - [x] Run experiments
  - [x] Report results
    - [x] Update method description
    - [x] Add experiment results
    - [ ] TBD
- Not acknowledged and questionable:
  - [ ] Split SAR TM into 2 TMs
    - State TM: (s, a) $\rightarrow$ s'
    - Action TM: s $\rightarrow$ a
    - Direct external rewards aren't a thing
    - Reinforcement isn't tracked ATM
  - [ ] Investigate `MaxSegmentsPerCell` parameter impact
  - [ ] Implement integer encoder w/ overlapping buckets
    - overlapping should be a parameter
    - it defines the level of uncertainty
    - MDP planning becomes a light version of POMDP planning because of uncertainty
  - [ ] Investigate relation between overlapping and sufficient activation thresholds
  - [ ] Investigate `MaxSynapsesPerSegment` parameter impact
  - [ ] Start testing on POMDPs

Non-critical issues needing further investigation

Auxialiary tasks, usability improvements and so on

- [x] config based tests
  - [x] test config + builder classes
  - [x] improve config based building:
    - one config file for one test run (=all agents one test)
    - or even one config file for the whole experiment (=all agents all tests)
- [x] fine grained trace verbosity levels
- [x] setup release-based dev cycle
  - add tagging to git commits
  - how to add release notes
  - ?notes for major releases should contain algo details from FAQ
- [x] release v0.1 version of the SAR-based agent
- [ ] for v1.x
  - [ ] ? gym-like env interface
  - [ ] ? refactor envs and env generators (naming, names)
  - [ ] start live-logging
- [ ] for v2.x
  - [ ] remove legacy SAR-based parts
- [ ] extend Quick intro based on recent experience with students
  - [ ] add task on SP to the Readme
    - [x] fix entropy formula
    - [ ] add questions and requirements to test that learning is working
    - [ ] the same for boosting
  - [ ] update intro to TM part
    - [ ] mini-task on prediction
    - [ ] mini-task on backtracking

## Thoughts and ideas

- consider using SP between an input an TM
  - only states need SP, because actions and reward are just ints (naive encoding is enough)
  - concat them together
  - it will take care of sparsity
  - maybe smoothes the difference in size for a range of diff environments
    - bc even large envs may have a very small signal
- consider TD($\lambda$)-based approach from Sungur's work
- split SAR-based TM into State TM + Action TM
  - both has apical connections to each other
  - reward or goal-based approach? Could be interchangeable
- goal-based hierarchies of TM blocks
- SP hierarchies for large input images
  - with every SP working similar to convolution filter
- consider doing live-logging experiments in markdown there

## 2020.08.08 Sat

### Gridworld transfer learning experiment

First results show that the agent with 1-goal goal list performs better than the agents with larger goal list size.

### TM related interesting links

- [Temporal Pooler](https://github.com/numenta/htmresearch/wiki/Overview-of-the-Temporal-Pooler)
  - a concept of the algo by htm.research
  - makes TM more robust to intra-sequence noise
  - general idea as I understand it - to add exponential averaging of activations over time
  - but.. found that on the forum (May 2020):
  > As far as I know, there is no official implementation of a "temporal pooler", just experimental research code. If you are talking about the union pooler logic in the research repo, I’m not sure anyone is actually working on this anymore.
- [Network API FAQ](https://github.com/htm-community/htm.core/blob/master/docs/NetworkAPI.md)
- [Tiebreak TM](https://github.com/htm-community/htm.core/blob/master/py/htm/advanced/algorithms/apical_tiebreak_temporal_memory.py)
  - basal + apical connections
  - hence two kinds of depolatizations
  - cell is depolarized iff
    - apical + basal depolarization
    - only basal and no apical at all

## 2020.09.06 Sun

- SL/RL for SP  
  Spatial Pooler learns clustering, and does it unsupervised. Note that each output bit has no predefined meaning - we can shuffle output SDR and get exactly the same clustering mapping, i.e. it is isomorphic with respect to [any fixed] output bits permutations.

  Meaningful output SDR requires supervised or reinforcement learning to propagate this information to the mapping. And the question is how to make this learning for SP? At the moment I find these questions as the most important and interesting, honestly.

- Why MCTS and what it brings?  
  It brings learnable behavior aka "policy". The main reason to use MCTS is that it's proved to be good in model-based approach, when you have an env model. It doesn't fit well into our "naturalistic" approach, though, but seems like a good way to utilize or/and test strong HTM features of the memory.  
  So, the goal is to analyse an ability to model environments and test this ability with a model-based method [MCTS].

- Noise/uncertainty effects
  - add noise to input or make env partially observable?
  - what's the difference?

___

Model-based, planning, MCTS:

- Цель
  - оценить способность SP+TM моделировать среды, используя MCTS в качестве метода обучения
- Зачем
  - хороший бейзлайн для сравнения
  - потенциально открывает довольно сложные среды для тестирования
- Реализовать MCTS в общем виде
  - использует "черный ящик", моделирующий среду
  - для начала в кач-ве реализации черного ящика - только сама среда
- Добавить выученную модель среды
  - ТМ учит переходы (s, a) -> (s', a')
  - До какого момента обучать модель среды?
  - Как учитывать неполноту знаний о переходах из состояния?
  - На каких средах тестировать?
  - На этом этапе используются полностью определенные детерминированные небольшие MDP, так что ТМ попросту запоминает переходы.
- Усложнять среды (варианты в порядке постепенного усложнения)
  - более информативные и связанные состояния, используем кодирование состояний с пересечениями
    - реализации новых сред на первых порах не понадобятся - будем использовать ручные, позже понадобится
    - нужен новый кодировщик
    - сложность в создании сред с хорошими информативными состояниями
  - переходим к POMDP
  - опционально добавление шума и стохастичности сред
    - стохастичность - потенциальная боль, с ходу непонятно как решать
    - шум - проверка стойкости к нему, проверка нашего понимания гиперпараметров
  - __NB__: изучение возможностей памяти через призму MCTS производится косвенно по результатам политики, обученной на модели среды, что может оказаться сложнее для анализа, чем непосредственное исследование самих возможностей кодирования и памяти

Исследование возможностей SP:

- В чем цель
  - лучше понять сильные и слабые стороны SP
  - изучить эмбеддинги, полученные с помощью SP - как много инфы сохраняется, какая семантика, как влияют гиперпараметры
- Изучение на примере кластеризации SP (в связке с классификатором на основе выхода SP)
- Исследовать качество классификации на различных датасетах
  - на специализированных, подогнанных под наши цели
  - на датасетах общего назначения с картинками
    - побочно можно затронуть задачу кодирования изображений, например, с помощью аналога сверток - небольших SP
- Постепенно усложнять датасеты/задачи:
  - аналогично предполагаемому усложнению сред (см соотв пункт в планировании)
- Почему это важно
  - кодирование с помощью SP - базовая и самая мощная фишка HTM
  - критично понимать и чувствовать его возможности
  - оно устанавливает верхнюю границу качества для всего более высокоуровневого, так что эту верхнюю границу желательно знать в проводимых экспериментах
    - например, если получившееся кодирование неудачно, то никакой классный планировщик не сможет выучить нормальную политику

Обучение с учителем (SL) и с подкреплением (RL) для SP:

- Зачем
  - SP умеет делать кластеризацию, позиция выходных бит не несет информации - т.е. для метода обучения SP выходной вектор инвариантен относительно перестановки бит
  - обучение кодированию с фиксированной позицией выходных бит выглядит очень полезным - тогда кластеризация превращается в классификацию
  - обучать можно по-разному: SL предполагает наличие желаемого/правильного ответа, а RL - наличие сигнала о пользе
  - оба способа желательно иметь в арсенале, т.к. [возможно] оба механизма используются в мозге
- Цель
  - пополнить арсенал средств методами обучения для SP
  - выбрать и проверить несколько наиболее простых альтернатив - возможно, этого будет достаточно для первого раза
  - подготовить эксперимент для его последующего продолжения
    - т.е. на первых порах нас интересует быстрый простой результат
    - по необходимости предполагается продолжать исследовать эти вопросы
- Исследовать качество обучения SP
  - для RL на задачах с бандитами
  - для SL на задачах классификации
- Это направление пересекается с направлением про возможности SP
  - их можно проводить совместно, постепенно усложняя датасеты/задачи
- Почему это важно
  - имея SL/RL для SP мы сможем создавать RL-агентов разной степени сложности, т.е. перейти к вопросам топологии (архитектуры)
  - откроется возможность перейти к реальным нейрофизиологическим аналогиям

Техническое направление:

- Добавить визуализацию SDR/SP/TM, используя готовые инструменты

## 2020.09.18 Fri

- check [REAL](https://github.com/AIcrowd/REAL2020_starter_kit/tree/master/baseline)
- раз в неделю созваниваться командой
  - первый созвон сделать в ближайшее время
  - обсудить репозиторий и задачи
- ссылку на оверлиф
- смысл: функция полезности
- дать доступ в репо

## 2020.09.25 Fri

- Artem
  - htmschool: 3d + 2d in one repository
  - PandaVis: slow, hanging
    - let's choose it for now
    - try it for our experiments
- Eugene
  - lidar: distance + type of obstacle (wall or reward)
- Petr
  - report
    - to 01.10 reg to the [conf](http://iiti.rgups.ru/ru/important-dates/)
    - to 15.10 report
  - REAL
    - planner - dig deeper
    - absence of goal looks interesting
    - linked [lab](https://www.istc.cnr.it/group/locen) - interesting works on intrinsic motivation; maybe to contact them in future
  - MCTS
    - make pseudocode or images
    - to the next call: prepare a talk
    - explicit goal setting
