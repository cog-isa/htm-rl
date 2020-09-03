# Project details

- [Project details](#project-details)
  - [Постановка задачи и среда](#постановка-задачи-и-среда)
  - [Agent](#agent)
    - [Планировщик. Шаг 1: планирование до цели](#планировщик-шаг-1-планирование-до-цели)
    - [Планировщик. Шаг 2: нахождение обратного пути](#планировщик-шаг-2-нахождение-обратного-пути)
  - [Тестирование](#тестирование)
    - [Тестирование на задаче transfer learning в одной среде](#тестирование-на-задаче-transfer-learning-в-одной-среде)
      - [multi_way_v0](#multi_way_v0)
      - [multi_way_v1](#multi_way_v1)
      - [multi_way_v2](#multi_way_v2)
    - [Тестирование на задаче transfer learning в нескольких средах](#тестирование-на-задаче-transfer-learning-в-нескольких-средах)
      - [Эксперименты с редкой сменой наград и сред](#эксперименты-с-редкой-сменой-наград-и-сред)
      - [Эксперименты со средней частотой смены наград и сред](#эксперименты-со-средней-частотой-смены-наград-и-сред)
      - [Эксперименты с частой сменой наград и сред](#эксперименты-с-частой-сменой-наград-и-сред)
      - [Эксперименты с ультрачастой сменой наград и сред](#эксперименты-с-ультрачастой-сменой-наград-и-сред)
    - [Выводы](#выводы)
  - [Project specific terminology](#project-specific-terminology)
  - [Encoding](#encoding)
  - [Planning alforithm details](#planning-alforithm-details)
    - [Step 1. Forward prediction](#step-1-forward-prediction)
    - [Step 2. Backtracking from goal states](#step-2-backtracking-from-goal-states)
    - [Step 1: Forward prediction](#step-1-forward-prediction-1)
    - [Step 2: Backtracking](#step-2-backtracking)
    - [Step 3: Re-check](#step-3-re-check)
  - [Configuration based building](#configuration-based-building)
  - [Run arguments](#run-arguments)
  - [Parameters](#parameters)
    - [Currently in use](#currently-in-use)
    - [Adviced by Numenta community](#adviced-by-numenta-community)

## Постановка задачи и среда

Агент играет в среде GridWorld - лабиринт на квадратной сетке. Агент начинает игру в фиксированной точке старта, и ему требуется найти единственную награду, которая находится в фиксированном месте.

Рассматриваются два близких варианта GridWorld - с учетом направления взгляда агента и без:

- В первом случае в каждый момент времени агент находится в некоторой клетке лабиринта, указывая лицом одно из 4х направлений: восток, север, запад, юг
  - Из текущего состояния он может совершить два действия: а) шагнуть прямо (в соседнюю клетку) или б) повернуться на 90 градусов против часовой стрелки.
  - Опционально в среде может быть разрешено третье действие - поворот на 90 градусов по часовой стрелке (по умолчанию запрещено).
- Во втором случае агент не имеет направления взгляда и может шагать непосредственно в любом из четырех направлений (восток, север, запад, юг), т.е. в одну из четырех соседних клеток.

Переходы между состояниями однозначны, т.е. среда описывается детерминированным MDP. Когда агент пытается шагнуть "в стенку", он остается на месте.

На каждом невыигрышном шаге агент получает небольшую отрицательную награду (-0.01), на выигрышном - большую положительную (1.0).

## Agent

Текущая реализация агента содержит, грубо говоря, две политики - случайную и спланированную. Спланированную политику предоставляет планировщик. В случае, если он не может предоставить политику, применяется случайная.

Таким образом основная логика агента содержится в реализации планировщика.

Задача планировщика - найти путь до награды, если она достижима за не более чем `planning_horizon` шагов. Он это делает в два этапа:

1. Планирование до награды из текущего состояния
2. Нахождение обратного пути из состояния с наградой в текущее

Чтобы агент мог планировать, он наделен памятью для запоминания увиденных переходов между состояниями, которая эмулирует функцию динамики среды $f: (s, a) \rightarrow s'$. При этом память агента имеет некоторые особенности.

Главная особенность - она способна работать с суперпозициями (=множествами). Это означает, что память может отвечать на запросы вида _имея набор состояний $\{s_i\}$ и набор действий $\{a_j\}$, в какие состояния $\{s'_m\}$ могут привести все возможные пары $(s_i, a_j)$_? В ответ на такой запрос память возвращает суперпозицию следующих состояний.

Вторая особенность - для каждого конкретного состояния $s'$ из суперпозиции в ответе на запрос к памяти есть возможность узнать $(s, a) = f^{-1}(s')$

Третья особенность - память агента дополнительно снабжена подмодулем памяти о выигрышных состояниях. На текущий момент этот подмодуль реализован с помощью наивного списка состояний, в которых агент получил положительную награду - мы называем этот подмодуль списком целей. Список целей имеет ограниченный размер S, являющийся гиперпараметром.

На основе этих особенностей и построен алгоритм планировщика.

### Планировщик. Шаг 1: планирование до цели

Агент опирается на свой опыт и исходит из предположения, что награда может находиться в одном из выигрышных состояний предыдущих эпизодов. Поэтому состояния из списка целей являются приоритетными для исследования.

Пусть в некоторый момент времени $t_0$ агент находится в состоянии $s_0$. Чтобы ответить на вопрос, есть ли достижимое состояние из списка целей в радиусе $N$ действий, начнем планировать из суперпозиции $(s_0, A)$, где $A$ - все множество доступных действий агента.

После первого шага предсказания мы будем иметь множество состояний $\{s_i\}^1$, в которые можно попасть из текущего за одно действие агента. Продолжим планировать из суперпозиции этих состояний так же по всем возможным действиям. И так далее до тех пора пока либо не обнаружим хотя бы одно состояние из списка целей, либо не исчерпаем лимит горизонта планирования $N$ = `planning_horizon`.

Если цель найдена, то планировщик переходит к следующему этапу.

NB: первый этап может обнаружить сразу несколько достижимых целевых состояний в радиусе $T$ шагов. Второй этап проводится последовательно для каждого из этих состояний до первого успеха. Выбор очередности перебора целей в текущей реализации явно не определен.

### Планировщик. Шаг 2: нахождение обратного пути

Итак, на текущий момент, планировщик уверен, что некоторое целевое состояние $s_T$может быть достигнуто за $T$ шагов, но не знает в точности как - на первом этапе мы работали с суперпозициями и использовали все возможные действия, поэтому какие конкретно действия привели к целевому состоянию, неизвестно.

На момент окончания успешного планирования за $T$ шагов, имеем всю историю суперпозиций состояний на каждом из этих шагов $[s_0, \{s_i\}^1, .. \{s_i\}^t]$.

По целевому состоянию планировщик выбирает пару состояние-действие $(s_{T-1}, a_{T-1}) = f^{-1}(s_T)$, которые привели к нему. Таким образом для предпоследнего шага известно требуемое действие $a_{T-1}$ и состояние $s_{T-1}$, в которое нужно прийти.

Далее рекурсивно повторяем процедуру, на каждом шаге $t$ находя пару состояние-действие $(s_{t-1}, a_{t-1}) = f^{-1}(s_t)$, пока не вернемся в начальную позицию $s_0$.

Здесь, правда, стоит уточнить, что $f^{-1}$ в общем случае возвращает не единственную пару $(s, a)$, а множество пар, и не всякая пара рекурсивно приведет нас обратно в начальную позицию (т.е. реально достижима из начальной позиции). Поэтому рекурсивно проверяется каждая из возможных пар до первого успешного возвращения в начало.

Заметим, что в процессе возвращения назад алгоритм формирует однозначную последовательность действий $[a_0, .. , a_{T-1}]$, которая приведет агента из начального состояния $s_0$ в искомое целевое состояние $s_T$.

Если процесс нахождения обратного пути завершился успешно, планировщик имеет политику действий на $T$ шагов вперед, которую агент безусловно выполняет. В случае успеха агент находит награду, и эпизод завершается, иначе агент исключает рассмотренное целевое состояние из списка целей до окончания текущего эпизода и пытается заново спланировать действия.

## Тестирование

Цель тестирования:

- Исследовать влияние на производительность агента:
  - длины горизонта планирования
  - размера списка целей
- Сравнить производительность агента с бейзлайнами:
  - Random Walk агент
  - DQN агент

Тестирование агента содержало два различных набора экспериментов в соответствии с исследуемыми задачами transfer learning, отличающихся количеством сред в каждом из экспериментов.

В качестве реализации рандомной стратегии использовался частный случай htm агента с нулевым горизонтом планирования. У DQN агента в сравнении использовались результаты обеих стратегий - жадной и $\epsilon$-жадной.

### Тестирование на задаче transfer learning в одной среде

В рамках одной среды проводилась серия испытаний по N эпизодов. В течение одного испытания из N эпизодов агент учился находить награду в некоторой ячейке. Между испытаниями местоположение награды изменялось. Исследовалась способность агентов адаптироваться к новому местоположению награды.

Для этого набора экспериментов были выбраны 3 среды типа Gridworld без учета направления взгляда агента.

Ниже к каждой из этих сред приведены графики результатов рассматриваемых агентов: а) награда за эпизод, б) количество шагов за эпизод и в) длительность выполнения одного эпизода (в секундах). На всех графиках изображена скользящая средняя с окном, указанным в заголовке графика (например, `MA = 20`)

Легенда к названиям агентов на графиках:

- `htm` в названии обозначает htm агента
  - суффикс `_X` означает горизонт планирования `X`, например:
    - `htm_0` - нулевой горизонт планирования, т.е. случайный агент
    - `htm_4` - горизонт планирования 4
  - суффикс `_Xg` означает размер списка целей `X`, например:
    - `htm_2_1g` - горизонт планирования 2, длина списка целей 1
    - `htm_1_16g` - горизонт планирования 1, длина списка целей 16
    - если размер списка целей не указан явно, предполагается неограниченная длина списка
- `dqn` в названии обозначает DQN агента
  - `_greedy` - суффикс жадной стратегии
  - `_eps` - суффикс $\epsilon$-жадной стратегии

Легенда к схемам тестовых сред:

- `-` пустая клетка,
- `#` стена,
- `@` агент,
- `X` награда

#### multi_way_v0

Последовательность из четырех испытаний по 100 эпизодов. Положения награды в испытаниях было следующим:

```bash
#####   #####   #####   #####
#@--#   #@-X#   #@--#   #@--#
#-#-#   #-#-#   #-#-#   #-#X#
#--X#   #---#   #X--#   #---#
#####   #####   #####   #####
```

Сравнение агента с бейзлайнами:

![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v0__steps.svg)
![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v0__steps_rel_htm_0.svg)

Сравнение DQN жадного и $\epsilon$-жадного агента:

![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v0__dqn__steps.svg)
![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v0__dqn__steps_rel_dqn_greedy.svg)

Сравнение агента с разными горизонтами и размером списка целей:

- горизонт планирования 1
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v0__1__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v0__1__steps_rel_htm_1_1g.svg)

- горизонт планирования 2
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v0__2__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v0__2__steps_rel_htm_2_1g.svg)

- горизонт планирования 4
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v0__4__steps.svg)

#### multi_way_v1

Последовательность из пяти испытаний по 200 эпизодов. Положения награды:

```bash
#######   #######   #######   #######   #######
#@--###   #@--###   #@--###   #@--###   #@--###
#-#-###   #-#-###   #-#X###   #-#-###   #-#-###
#X----#   #-----#   #-----#   #-----#   #----X#
###-#-#   ###X#-#   ###-#-#   ###-#-#   ###-#-#
###---#   ###---#   ###---#   ###--X#   ###---#
#######   #######   #######   #######   #######
```

Сравнение агента с бейзлайнами:

![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v1__steps.svg)
![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v1__steps_rel_htm_0.svg)

Сравнение DQN жадного и $\epsilon$-жадного агента:

![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v1__dqn__steps.svg)
![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v1__dqn__steps_rel_dqn_greedy.svg)

Сравнение агента с разными горизонтами и размером списка целей:

- горизонт планирования 1
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v1__1__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v1__1__steps_rel_htm_1_1g.svg)

- горизонт планирования 2
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v1__2__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v1__2__steps_rel_htm_2_1g.svg)

- горизонт планирования 4
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v1__4__steps.svg)
  
#### multi_way_v2

Последовательность из шести испытаний по 200 эпизодов. Положения награды:

```bash
#########    #########    #########    #########    #########    #########
#---#####    #-X-#####    #---#####    #---#####    #---#####    #---#####
#-#-#####    #-#-#####    #-#-#####    #-#-#####    #-#-#####    #-#-#####
#--@---##    #--@---##    #--@--X##    #--@---##    #X-@---##    #--@---##
###--####    ###--####    ###--####    ###--####    ###--####    ###--####
###-#--X#    ###-#---#    ###-#---#    ###-#---#    ###-#---#    ###-#--X#
###---###    ###---###    ###---###    ###---###    ###---###    ###---###
###-#####    ###-#####    ###-#####    ###X#####    ###-#####    ###-#####
#########    #########    #########    #########    #########    #########
```

Сравнение агента с бейзлайнами:

![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v2__steps.svg)
![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v2__steps_rel_htm_0.svg)

Сравнение DQN жадного и $\epsilon$-жадного агента:

![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v2__dqn__steps.svg)
![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v2__dqn__steps_rel_dqn_greedy.svg)

Сравнение агента с разными горизонтами и размером списка целей:

- горизонт планирования 2
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v2__2__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/multi_way_transfer/multi_way_v2__2__steps_rel_htm_2_1g.svg)

- горизонт планирования 4
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v2__4__steps.svg)

- горизонт планирования 8 и 12
  
  ![episode steps absolute](../../reports/v1_0/multi_way_transfer/multi_way_v2__8-12__steps.svg)

### Тестирование на задаче transfer learning в нескольких средах

Эксперименты данного режима проводились на наборе различных сред-лабиринтов. Схема каждого эксперимента следующая:

- эксперимент проводится последовательно на $N_{env}$ средах
- при фиксированной среде последовательно рассматриваются $N_{rew}$ положений награды
- при фиксированном положении награды рассматривается $N_{s_0}$ начальных положений агента
- в фиксированной конфигурации агент играет $N_{ep}$ эпизодов

Исследовалась способность агента адаптироваться не только к новому местоположению награды в рамках одной среды, но и к новой среде.

Эксперименты тоже проводились на средах типа Gridworld без учета направления взгляда агента. В рамках каждого эксперимента все среды генерировались случайно на некотором фиксированном квадрате $n \times n$ (в основном $5 \times 5$ и $6 \times 6$).

Примеры сгенерированных сред

Легенда:

- темно фиолетовый - стены
- желтый - начальная позиция агента
- салатовый - награда

Примеры сред:

- 5x5
  ![gridworld 5x5 #1](../../reports/v1_0/gridworld_transfer/gridworld_5x5_1_1_200_1337_map_0_1173222464.svg)
  ![gridworld 5x5 #2](../../reports/v1_0/gridworld_transfer/gridworld_5x5_1_1_200_1337_map_2_1561234712.svg)

- 8x8
  ![gridworld 8x8 #1](../../reports/v1_0/gridworld_transfer/gridworld_8x8_10_1_10_1337_map_0_1173222464.svg)
  ![gridworld 8x8 #2](../../reports/v1_0/gridworld_transfer/gridworld_8x8_10_1_10_1337_map_7_2032734714.svg)

Ключевым различием между экспериментами были суммарное число эпизодов с фиксированной наградой и суммарное число эпизодов с фиксированной средой, которые обратно пропорционально влияли на сложность - чем меньше эпизодов в распоряжении агента, тем быстрее ему требуется адаптироваться (к новой награде и/или к новой среде, соответственно).

#### Эксперименты с редкой сменой наград и сред

- Gridworld 5x5
- $N_{s_0} = 100$
- $N_{rew} = 2$
- $N_{env} = 8$

Ниже представлены графики для двух разных сидов.

Сравнение агента с бейзлайнами:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__steps.svg)
![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__steps_rel_htm_0.svg)
![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__steps.svg)
![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__steps_rel_htm_0.svg)

Сравнение DQN жадного и $\epsilon$-жадного агента:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__dqn__steps.svg)
![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__dqn__steps.svg)

Сравнение агента с разными горизонтами и размером списка целей:

- горизонт планирования 1
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__1__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__1__steps_rel_htm_1_1g.svg)
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__1__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__1__steps_rel_htm_1_1g.svg)

- горизонт планирования 2
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__2__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__2__steps_rel_htm_2_1g.svg)
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__2__steps.svg)
  ![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__2__steps_rel_htm_2_1g.svg)

- горизонт планирования 4-8
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__4-8__steps.svg)
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_1337__4-8__steps_rel_htm_4_1g.svg)
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__4-8__steps.svg)
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_100_2_8_42__4-8__steps_rel_htm_4_1g.svg)

#### Эксперименты со средней частотой смены наград и сред

- Gridworld 5x5
- $N_{s_0} = 50$
- $N_{rew} = 4$
- $N_{env} = 4$

Сравнение агента с бейзлайнами:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__steps.svg)
![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__steps_rel_htm_0.svg)

Сравнение DQN жадного и $\epsilon$-жадного агента:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__dqn__steps.svg)
![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__dqn__steps_rel_dqn_greedy.svg)

Сравнение агента с разными горизонтами и размером списка целей:

- горизонт планирования 1
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__1__steps.svg)

- горизонт планирования 2
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__2__steps.svg)

- горизонт планирования 4-8
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__4-8__steps.svg)
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_50_4_4_1337__4-8__steps_rel_htm_4_1g.svg)

#### Эксперименты с частой сменой наград и сред

- Gridworld 5x5
- $N_{s_0} = 20$
- $N_{rew} = 1$
- $N_{env} = 20$

Сравнение агента с бейзлайнами:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_20_1_20_1337__steps.svg)
![episode steps log-relative](../../reports/v1_0/gridworld_transfer/gridworld_5x5_20_1_20_1337__steps_rel_htm_0.svg)

Сравнение DQN жадного и $\epsilon$-жадного агента:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_20_1_20_1337__dqn__steps.svg)

Сравнение агента с разными горизонтами и размером списка целей:

- горизонт планирования 1
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_20_1_20_1337__1__steps.svg)

- горизонт планирования 2
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_20_1_20_1337__2__steps.svg)

- горизонт планирования 4-8
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_20_1_20_1337__4-8__steps.svg)

#### Эксперименты с ультрачастой сменой наград и сред

Нужно?

- Gridworld 5x5
- $N_{s_0} = 1$
- $N_{rew} = 1$
- $N_{env} = 200$

Сравнение агента с бейзлайнами:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_1_1_200_1337__steps.svg)

Сравнение DQN жадного и $\epsilon$-жадного агента:

![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_1_1_200_1337__dqn__steps.svg)

Сравнение агента с разными горизонтами и размером списка целей:

- горизонт планирования 1
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_1_1_200_1337__1__steps.svg)

- горизонт планирования 2
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_1_1_200_1337__2__steps.svg)

- горизонт планирования 4-8
  
  ![episode steps absolute](../../reports/v1_0/gridworld_transfer/gridworld_5x5_1_1_200_1337__4-8__steps.svg)

### Выводы

Ключевые выводы для экспериментов с неизменной средой:

- Чем больше горизонт планирования, тем лучше играет агент
  - причем улучшение кумулятивно, т.е. каждый следующий +1 горизонта дает больший прирост, чем предыдущий
- Агент с планированием играет лучше, чем случайный
  - чем сложнее среда, тем менее заметен в относительных масштабах переход от horizon_planning=0 к horizon_planning=1
  - т.е. при фиксированном горизонте планирования с увеличением сложности среды результат в относительных масштабах начинает прижиматься к результатам случайного агента
- Агент с большим накопленным набором целей направленнее (=оптимальнее) исследует среду
  - предположительно потому, что менее вероятно использование случайной стратегии
  - в частных случаях это субоптимально - когда награда сразу достижима в рамках имеющегося горизонта планирования и ее положение не изменилось с предыдущего эпизода, т.к. агент с большим списком целей может выбрать для исследования псевдо-цель
  - вероятно, причина данного эффекта кроется в специальном виде использованных сред, начального положения агента и положения наград.
- Увеличение размера списка целей и величины горизонта планирования положительно сказываются на способности агента приспосабливаться к новому местоположению награды
- Увеличение горизонта планирования ухудшает производительность агента
  - однако это ухудшение происходит до некоторого момента, пока горизонт планирования не приблизится к длине оптимального пути к цели
- Агент учится и приспосабливается быстрее, чем DQN [в терминах числа эпизодов]
- Если горизонта планирования достаточно, то агент быстро учится решать задачу планирования идеально, достигая оптимальной стратегии
  - учится в ~1.5-2.5 быстрее DQN
  - но для любого фиксированного горизонта, если начать усложнять среду, то в какой-то момент DQN обгонит по результатам, потому что DQN решает задачу [почти всегда] оптимально, а htm агент - только в случае достаточного горизонта планирования
- Иногда htm agent сходится к субоптимальному решению
  - если горизонта планирования достаточно для субоптимального решения из начальной точки
  - и имеется несколько длинных узких "ходов" до цели, имеющих примерно одинаковую длину
  - тогда есть ненулевая вероятность, что агент не успеет запомнить путь через оптимальный коридор и будет планировать только через субоптимальный.
  - в такой ситуации eps-жадная стратегия DQN, наоборот, рано или поздно откроет оптимальное решение

Ключевые выводы для экспериментов со сменой сред:

- В вырожденном случае смены среды и награды на каждом эпизоде все агенты проигрывают случайному
  - DQN при этом играет значительно хуже HTM агента
- HTM агент в среднем быстрее чем DQN адаптируется к изменениям среды или в среде
  - благодаря этому обычно он ведет себя стабильнее
- Увеличение списка целей негативно сказывается на результатах агента
  - видимо, польза от выполнения псевдо-целей в среднем ниже штрафа
  - можно объяснить тем, что в среднем псевдо-цели не приближают агента к реальной и только увеличивают ожидаемое число шагов с использованием случайной стратегии

## Project specific terminology

State-action (SA)

- Given trajectories : $\tau_i = (s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, ...)$
- SA is a tuple $sa_t = (s_t, a_t)$
  - i.e. action $a_t$ taken from a state $s_t$
- TM learns agent's trajectories $\tau_i$
  - it operates with SA [encoded as] SDR sequences $(sa_0, sa_1, ... )$
  - which results that TM learns both
    - transitions $(s_{t-1}, a_{t-1}) \rightarrow s_t$
    - and which actions $a$ agent did from a state $s_t$
- TM then can make predictions
  - it can predict next state and all actions agent did from this state
  - each prediction is an SDR - a union (= bitwise OR) of pairs $(s, a^*)$
  - we give this union a special name (see below)

SA SDR Superposition

- Union (= bitwise OR) of any number of SA SDRs
- Rationale
  - if an agent saw transition $(s_{t-1}, a_{t-1}) \rightarrow s_t$ and did different actions $a_t$ from a state $s_t$
  - then prediction for a pair $(s_{t-1}, a_{t-1})$ will be a union of all seen pairs $(s_t, a_t)$
  - aka superposition of these SAs
- TM can work with SA SDR superpositions both as input and output
  - In general case (mentioned just before) TM's output is a superposition
  - TM accepts SA superpositions as input as well
    - it's possible because SA SDR superposition still is just an SDR
- *NB*: prediction process is called depolarization

## Encoding

SA SDR encoder

- Encodes SA to SDR
- Uses separate state and action SDR encoders
  - Resulting SDRs are concatenated
- We consider only discrete environments
  - Hence both states and actions are discrete sets
- Each state or action can be represented as integer number
  - i.e. $s_t \in [0, |S|)$ and $a_t \in [0, |A|)$
  - to encode states and action we use Integer SDR encoder

Integer SDR encoder

- Encodes integer number $x \in$ [0, `n_values`) to SDR
- Parameters:
  - `n_values` - a number of unique values
  - `value_bits` - a number of bits used to encode each unique value
- Resulting SDR has `total_bits` = `n_values` $\times$ `value_bits` bits
  - they can be logically divided into `n_values` buckets of `value_bits` contiguous bits
    - each bucket corresponds to a value from the range
  - e.g. 2 $\rightarrow$ `0000 0000 1111` encoded by 3-by-4 integer encoder
    - 3 buckets are separated by space to make it clear
    - every value id encoded by 4 bits bucket
    - note that buckets don't intersect
- PROS
  - Easy to encode/decode
  - Easy to pretty print and read
  - Hence easy to debug
  - Different values have no intersection
    - eliminates some unnecessary complexities
- CONS
  - Different values have no intersection
    - narrows the use of SDRs
    - doesn't use information about states similarity
  - You have no direct control on sparsity, which is `1 / n_values`
    - you may have problems with too low or too high sparsity levels
    - we tested that TM works well with low sparsity
      - which doesn't mean it has not negative effects at all
    - just remember that sparsity may lead to problems

State SDR encoder

- For simple MDP environments we use integer encoding
- For complex environments encoding states [or observations] becomes tricky
  - of course, you still can enumerate all possible states and use Integer SDR encoder
  - but the number of all possible states grows very fast
    - in practice it works well only for small environments
  - also Integer SDR encoder cuts out information about states similarity, which may not be desirable
  - one possible solution is to encode every pixel [or grid cell] separately then concatenate results, which preserves information about similarity between states
- At the moment we use only simple MDP environments and hence integer encoding for states

## Planning alforithm details

Let's consider an agent playing in some MDP environment. At some moment he is in a state, which we will denote as $s_0$, because it will be our starting point for planning.  
The agent also has a fixed set of goal states $\{g_i\}$, which he wants to reach.

The goal of the planning is to answer two questions. Is it possible to reach any of the goal states with at maximum of $n_max$ steps (i.e. actions)? And if it is possible, then what is the sequence of actions (i.e. policy) leading to that goal state?

Planning algorithm consists of 2 high-level steps: forward prediction and backtracking from the goal, which could be written in Python-like pseudocode as:

```python
def plan_actions(initial_state):
  reached_goals, active_segments_timeline = predict_to_goals(initial_state)
  planned_actions = backtrack_from_goals(
    reached_goals, active_segments_timeline
  )

  return planned_actions
```

### Step 1. Forward prediction

We start from the state $s_0$ and want to predict all reachable next states $\{s_1^i\}$. To do this we could check all learned transitions $(s_0, a_0^i) \rightarrow s_1^i$. It's achievable simultaneously if we use $sa_0 = (s_0, \cup a^i)$ as a starting input for the agent's TM. $sa_0$ is then a SA SDR superposition and semantically means exactly a superposition of all possible state-action pairs $(s_0, a_0^i)$ which activate predictions for all desired transitions.

__TBD: make simple MDP graph example__

As a prediction TM gives another SA SDR superposition $sa_1 = \cup (s_1, a_1^i)$ - a superposition of all next_state-next_action pairs. $sa_1$ can be further used as an input to TM, which gets us $sa_2$ prediction and so on.

As you see, for each step $k$, $sa_k$ contains a superposition of all states reachable in exactly $k$ actions (=moves). As a side note, the whole resulting process looks very similar to parallel breadth-first-search graph algorithm.

Recall that the goal of forward prediction is to find reachable goal states. Therefore at each step $k$ we look if any of the goal states $g_i$ are among the states in $sa_k$ superposition by checking their overlap $o = overlap(g_i, sa_k)$. We want to find only the closest goals, so if any of the goal states are matched, we stop forward prediction process and set these goal states as a result.

Also during forward prediction we need to record active segments at each step $k$ denoted as `active_segments_timeline`. It's required by the second step of planning.

Note also, that cell activation and segment activation, despite named similar, refer to the different phases of TM workflow. The latter is a part of prediction, not activation - active segment depolarizes cell, i.e. predicts its future activation instead of activating it.

Below is a pseudocode for forward prediction phase.

```python
def predict_to_reward(initial_state):
  # initializes starting $sa_0$ as superposition with all possible actions
  initial_sa = Sa(initial_state, AllActions)
  proximal_input = encoder.encode(initial_sa)

  reached_goals = []
  active_segments_timeline = []
  for i in range(max_steps):
    # activates & depolarizes cells
    active_cells, depolarized_cells = memory.process(
      proximal_input, learn=False
    )

    # records active segments for $i$-th step
    active_segments_timeline.append(
      memory.active_segments(active_cells)
    )

    # gets $sa_{i+1}$ SA SDR superposition
    proximal_input = columns_from_cells(depolarized_cells)

    # gets list of matched goal states
    reached_goals = [
      goal_state
      for goal_state in agent.goal_states
      if is_matched(goal_state, proximal_input)
    ]
    if reached_goals:
      break

  return reached_goals, active_segments_timeline
```

### Step 2. Backtracking from goal states

As a result of forward prediction phase we have a list of goal states reachable in $T$ actions along with recorded segments activation timeline. The goal of backtracking phase is to restore a correct sequence of actions leading to any of these goal states.

At high level view we start from any of the goal states and try moving backward [through time] to the initial state using segments activation timeline graph:

```python
def backtrack_from_goals(reached_goal_states, active_segments_timeline):
  T = len(active_segments_timeline)
  # cells predicted to be active at timestep T, when the goal states were reached
  depolarized_cells = active_segments_timeline[T-1].keys()

  for goal_state in reached_goal_states:
    # gets predicted cells related to the goal state
    depolarized_goal_cells = intersect(goal_state, depolarized_cells)

    # try backtracking from the goal state to the initial state
    planning_successful, planned_actions = backtrack_from_state(
        depolarized_goal_cells, T-1
    )
    if planning_successful:
      return planned_actions
```

The key thing for this phase is moving backward in an implicitly built graph of active cells. Consider having chosen some goal state $g$. Let's start with the first step of backtracking - timestep $T-1$ - when the goal state $g$ was __predicted__.

Here we should pause a bit and expand the meaning ot "the goal state was predicted". Recall that we work with numeric MDP states encoded as SDR with Integer SDR encoder. A single value $x$ is decoded from an SDR $S$ means that $S$ has enough ON bits related to $x$, with "enough" is defined by `activation threshold` hyperparameter. So "the goal state $g$ was predicted" means that there were at least `activation_threshold` ON bits related to it in a prediction.

From forward prediction phase we know the set of active cells at timestep $T-1$. And the fact that the goal state $g$ was predicted at that timestep creates a condition for backtracking - we have to make sure that it stays true. Obviously, the whole set of active cells at $T-1$ doesn't break this prediction condition.

But there's another condition - at each timestep there's only one transition $(s, a) \rightarrow s'$ allowed, because that's how MDP works. On the other hand the forward prediction process works with superpositions, and the set of active cells $sa_{T-1}$ may contain multiple different $(s, a)$ pairs, from which we may choose only those single pairs that don't break the prediction condition. The question is how to find these pairs?

The segments activations timeline allows us to take all active segments of the goal state $g$ cells at time $T-1$. Each active segment is associated with a set of active cells that activated it. By design each TM segment is supposed to recognize a single pattern, which is a state-action pair in our case.

__NB__: In general that's not true. A segment might recognize a set of patterns, and the most robust to the punishment during learning are those sets that have non-zero intersection, which minimizes expected punishment rate. It means, that each segment learns a cluster [or clusters] of patterns. But we can bound the segment's allowed size, which is a number of synapses, to fit no more than a single state-action pair size - then each segment will be learning to recognize only a single pattern. Still, this doesn't mean that all segment's synapses are always connected to the same state or action cells, although learning tends to fit them this way.

We might expect that the active cells sets $\{S_i^{T-1}\}$ associated with the active segments $\{A_i^{T-1}\}$ at time $T-1$ contain clusters, where each pair of active cells set has a high overlap score. Formally, we want to split $\{A_i^{T-1}\}$ into clusters $\{C_k^{T-1}| A_i, A_j \in C_k^{T-1}: overlap(A_i, A_j) \geq N_{cl}\}$ and keep only those that 

At any step $k$ we have the set of cells that should be activated at $k+1$ timestep in order to reach the goal state. It means that some key portion of these cells should be predicted at timestep $k$ - at least `activation threshold` cells - which gives us

- Given a set of depolarized cells `dep_cells` [at time $t$]
- Call active segment as _potential_ if it induces enough depolarization among `dep_cells`
  - _ehough_ means $\geq$ activation threshold
  - i.e. a number of depolarized cells will be enough for subsequent depolarizations
- Take presynaptic active cells of a potential segment as depolarized cells [at time $t-1$]

Backtracking algorithm

- Start from depolarized cells in reward = 1 columns [at time T]
- Recursively take backtracking steps for any potential segment until time 0 is reached

Details

- Given a set of depolarized cells `dep_cells` [at time $t$]
  - Each depolarized cell have at least one active segment
  - Each active segment is associated with a set of active presynaptic cells
    - This set defines some SAR SDR
    - i.e. each segment is conditioned on some correct SAR
- I want to find all potential sets of active presynaptic cells that induce enough depolarization of `dep_cells`
  - perfect case
    - `dep_cells` conditioned on the same SAR
    - i.e. this SAR induces full `dep_cells` depolarization
  - perfect multiple case
    - `dep_cells` conditioned on more than one SAR
    - i.e. each of these SAR induces full `dep_cells` depolarization
- real world relaxations
  - some SARs don't induce enough depolarization
    - i.e. $\lt$ threshold
  - segments don't exactly match each other
    - cells are conditioned on very similar SARs but not exactly the same

What is saved during this phase

- active [presynaptic] cells at time $t$ are saved as `backtracking_SAR_conditions[t]`
- they define condition on SAR at time $t-1$ to get desired depolarization at time $t$

Второй шаг - рекурсивный бэктрекинг из столбцов, соответствующих награде, назад во времени:

```python
def backtrack_from_reward(active_segments_timeline):
    final_depolarized_cells = active_segments_timeline[T-1].keys()
    depolarized_reward_cells = get_reward_cells(final_depolarized_cells)

    return backtrack(depolarized_reward_cells, T-1)


def backtrack(desired_depolarization: SparseSdr, t: int):
    # presynaptic cells clusters, each can induce desired depolarization
    cell_clusters = get_backtracking_candidate_clusters(
        desired_depolarization, activation_threshold, t
    )

    for cluster in cell_clusters:
        activation_timeline = backtrack(cluster, t-1)
        if backtracking_succeeded:
            activation_timeline.append(cluster)
            return activation_timeline
```

Откуда берутся кластеры-кандидаты:

```python
def get_backtracking_candidate_clusters(
        desired_depolarization: SparseSdr,
        sufficient_activation_threshold: int,
        t: int
):
    # Active presynaptic cell clusters (clusterization by columns)
    active_segments = active_segments_timeline[t][desired_depolarization]
    candidate_clusters = merge_segments_into_clusters(active_segments)

    # Keep clusters that induce sufficient depolarization among desired
    for cluster in candidate_clusters:
        count_induced_depolarization(cluster, desired_depolarization)

        if induced_depolarization >= sufficient_activation_threshold:
            backtracking_candidate_clusters.append(cluster)

    return backtracking_candidate_clusters
```

Последний шаг - предсказание награды по пути успешного бэктрекинга:

```python
def check_activation_timeline_leads_to_reward(
        initial_sar: Sar, activation_timeline
):
    proximal_input = encoder.encode(initial_sar)

    for i in range(T):
        action = extract_action(activation_timeline[i])
        proximal_input.action = action

        active_cells, depolarized_cells = agent.process(
            proximal_input, learn=False
        )
        proximal_input = columns_from_cells(depolarized_cells)

        planned_actions.append(action)

    if is_rewarding(proximal_input):
        return planned_actions
```

### Step 1: Forward prediction

Setting up

- Given initial state $s_0$
- To find a reward, I want to check all possible paths, starting from $s_0$
  - Then SAR superposition  
    (state = $s_0$, {all actions}, reward = 0)
  - .. represents all possible beginnings of these paths
  - So, encode it to SDR and set it as initial proximal input to TM

Forward prediction

- Given some SAR superposition as proximal input at time $t$
  - induces cells activation
- TM prediction: some SAR superposition at time $t+1$
  - prediction is in form of depolarized cells
- Save all active segments (discussed later in section)
- Check if depolarized SAR superposition contains reward = 1
- Continue forward prediction by taking predicted SAR superposition as input at $t+1$
  - until reward is found
  - or max forward steps are done

A bit of details on TM work, step by step:

- Given current proximal input [at time $t$]
- It directly induces columns activation
- Active columns [and depolarization from $t-1$] induce cells activation
  - any depolarized cell in active columns becomes active
  - or all cells become active if active column has no depolarized cells (bursting)
- Cells activation induces segments activation
  - number of active presynaptic cells $\geq$ activation threshold
- Segments activation induces cells depolarization
  - any active segment depolarizes its postsynaptic cell
- Depolarized cells can be collapsed into depolarized columns
  - any depolarized cell depolarizes its column
- Depolarized columns define predicted SAR SDR superposition [at time $t+1$]

What is saved during forward prediction phase:

- At each time step $t$ there's a set of segments that are active
- These active segments are saved into `active_segments_timeline[t]`
  - as dictionary: `depolarized_cell` $\rightarrow$ `cell_active_segments_list`
  - `cell_active_segments_list` - list of cell's active segments
  - each active segment represented as a set of active presynaptic cells

### Step 2: Backtracking

Backtracking step

- Given a set of depolarized cells `dep_cells` [at time $t$]
- Call active segment as _potential_ if it induces enough depolarization among `dep_cells`
  - _ehough_ means $\geq$ activation threshold
  - i.e. a number of depolarized cells will be enough for subsequent depolarizations
- Take presynaptic active cells of a potential segment as depolarized cells [at time $t-1$]

Backtracking algorithm

- Start from depolarized cells in reward = 1 columns [at time T]
- Recursively take backtracking steps for any potential segment until time 0 is reached

Details

- Given a set of depolarized cells `dep_cells` [at time $t$]
  - Each depolarized cell have at least one active segment
  - Each active segment is associated with a set of active presynaptic cells
    - This set defines some SAR SDR
    - i.e. each segment is conditioned on some correct SAR
- I want to find all potential sets of active presynaptic cells that induce enough depolarization of `dep_cells`
  - perfect case
    - `dep_cells` conditioned on the same SAR
    - i.e. this SAR induces full `dep_cells` depolarization
  - perfect multiple case
    - `dep_cells` conditioned on more than one SAR
    - i.e. each of these SAR induces full `dep_cells` depolarization
- real world relaxations
  - some SARs don't induce enough depolarization
    - i.e. $\lt$ threshold
  - segments don't exactly match each other
    - cells are conditioned on very similar SARs but not exactly the same

What is saved during this phase

- active [presynaptic] cells at time $t$ are saved as `backtracking_SAR_conditions[t]`
- they define condition on SAR at time $t-1$ to get desired depolarization at time $t$

### Step 3: Re-check

- Given active presynaptic cells at time $t$ needed to induce desired depolarization
- Start forward prediction again from initial sar superposition
- But now
  - Check that SAR `backtracking_SAR_conditions[t]` is consistent with the current active SAR
  - Extract action $a_t$ from `backtracking_SAR_conditions[t]`
  - Action $a_t$ defines a path to make at time $t$
  - Replace action superposition in proximal input with $a_t$
- Check that given `backtracking_SAR_conditions` lead to reward

## Configuration based building

This section describes configs syntax, rules and how to use it.

For configs we use yaml format, particularly 1.1 and 1.2 versions of its standard (the mix of them). You can read more about the format and its standards [here](TBD).

As for implementation of yaml parser we use [ruamel.yaml](TBD) package. It's a fork of even more popular and seasoned alternative [pyyaml](TBD).

Most shenanigans are based on the custom tags feature, supported by pyyaml and ruamel.yaml python packages.

- standard yaml tags
- custom tags
  - building through constructors and factory methods
  - naming conventions
  - how to register class tag
  - how to register factory method
- DRY
  - aliases
  - merging feature from 1.1 standard
    - how it works
  - how to use them
- ruamel patches
  - use cases with undesired default behavior
  - how they had been patched
- examples

## Run arguments

TBD:

- the set of arguments
- their relation
- examples

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
