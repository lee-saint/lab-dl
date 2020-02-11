"""
MDP(Markov Decision Process)
1. Q-Value Iteration Algorithm(가치 반복 알고리즘)
2. Q-Learning(Q-학습)
"""
import numpy as np

# 상태 공간(state-space): [s0, s1, s2]
# 행동 공간(action-space): [a0, a1, a2]

# Q-Value Iteration
transition_probs = [  # shape: (s, a, s')
    [  # shape: (a, s') / 현재 상태가 s0일 때
        [0.7, 0.3, 0.0],  # s0에서 a0 행동을 했을 때 s0, s1, s2로 전이될 확률
        [1.0, 0.0, 0.0],  # s0에서 a1 행동을 했을 때 s0, s1, s2로 전이될 확률
        [0.8, 0.2, 0.0]   # s0에서 a2 행등을 했을 때 s0, s1, s2로 전이될 확률
    ],
    [  # 현재 상태가 s1일 때
        [0.0, 1.0, 0.0],  # s1에서 a0 행동을 할 때
        None,  # s1에서 a1 행동을 할 때
        [0.0, 0.0, 1.0]   # s1에서 a2 행동을 할 때
    ],
    [  # 현재 상태가 s2일 때
        None,  # (s2, a0) -> s0, s1, s2
        [0.8, 0.1, 0.1],  # (s2, a1) -> s0, s1, s2
        None   # (s2, a2) -> s0, s1, s2
    ]
]

rewards = [  # shape: (s, a, s')
    [
        [10, 0, 0],  # (s0, a0) -> s0, s1, s2
        [0, 0, 0],  # (s0, a1) -> s0, s1, s2
        [0, 0, 0]   # (s0, a2) -> s0, s1, s2
    ],
    [
        [0, 0, 0],  # (s1, a0) -> s0, s1, s2
        None,  # (s1, a1) -> s0, s1, s2
        [0, 0, -50]   # (s1, a2) -> s0, s1, s2
    ],
    [
        None,  # (s2, a0) -> s0, s1, s2
        [40, 0, 0],  # (s2, a1) -> s0, s1, s2
        None  # (s2, a2) -> s0, s1, s2
    ]
]

# 각 상태(s)에서 가능한 action(a)의 리스트
possible_actions = [
    [0, 1, 2], [0, 2], [1]
]

# Q(s, a)
Q_values = np.full(shape=(3, 3), fill_value=-np.inf)
print(Q_values)
# 모든 상태(s)와 행동(a)에 대해서 Q_value의 값을 0으로 초기화
for state, action in enumerate(possible_actions):
    Q_values[state, action] = 0.0
print(Q_values)

gamma = 0.95  # 할인율
history_q_iter = []  # Q_value 반복 알고리즘 추정값을 저장할 리스트

for iteration in range(50):
    # Q0 -> Q1 -> Q2 -> ... -> Q_k -> Q_k+1 -> ... -> Q49
    Q_prev = Q_values.copy()  # Q_values 리스트가 계속 갱신되기 때문에
    history_q_iter.append(Q_prev)  # 복사한 값을 history에 추가
    for s in range(3):  # 가능한 상태(state) 개수만큼 반복
        for a in possible_actions[s]:  # 그 상태(state)에서 가능한 행동의 개수만큼 반복
            Q_values[s, a] = np.sum(
                [transition_probs[s][a][sp] * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp])) for sp in range(3)]
            )

print(Q_values)
print(np.argmax(Q_values, axis=1))  # 최적의 정책(policy)

# gamma = 0, 0.9로 변경해서 테스트
