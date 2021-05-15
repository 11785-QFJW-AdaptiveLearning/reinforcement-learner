# reinforcement-learner

Apply reinforcement learning to realize Adaptive Scheduling of educational activities.



## How to Start

#### 1 Adaptive Learning System

The settings are defined by "BKT_param" in main.py.
```python
BKT_param = {'numskill': 5, 'activity_per_skill': 4, 'pretest_per_skill': 5, 'p_L': 0.5,
                 'penalty': 0.1, 'learned_discount': 0.8, 'learned_penalty': 1.5, 'learned_sweet': 1}
```

To run our RL Agent with BKT simulator environment, please run main.py.

```python
python main.py
```

Some informations of each game will be output in the console. And the program will finally generate two files: 

* baseline_bkt_\<settings>_\<time>.npy file records history data of post-test scores, rewards and penalty
* baseline_bkt_\<settings>_\<time>.png file plots the running average every 100 games

Both the two file are suffixed by generated time.



#### 2 Linear Assignment

Since our baseline model is compared against the linear assignment condition, you can run linearAssign.py to see the post-test scores, rewards and penalty in linear assignment case.

You can also change the settings in "BKT_param" in linearAssign.py and then run the code.
```python
python linearAssign.py
```

Some informations of each game will be output in the console. And the program will finally generate two files: 

* linear_assign_bkt_\<settings>_\<time>.npy file records history data of post-test scores, rewards and penalty
* linear_assign_bkt_\<settings>_\<time>.png file plots the running average every 100 games

Both the two file are suffixed by generated time.



#### 3 Compare

To get the compare result figure shown in mid-term report, use compare.py.

1. Change the file names shown in the following code to the results you want to compare.

```python
linear = np.load('linear_assign_04110312.npy', allow_pickle=True).item()
rs = np.load('baseline_penalty2_04110317.npy', allow_pickle=True).item()
```

2. Run compare.py. A figure named compare.png will be generated.

```python
python compare.py
```
