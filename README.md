# meta_repository
```mermaid
graph TD

%% Task Sampling and Inner Loop
A[Task τᵢ Sampling] --> B[Inner Loop Strated]

B --> C1[Inner Policy π_ϕ Learning (PPO)]
B --> C2[Inner Critic V_ϕ Learning (TD, PPO ,etc)]

%% Outer Loop Begins
C1 --> D1[Outer Policy π_θ Update]
C2 --> D2[Outer Critic V_θ Update]

%% Value Matching for Critic
D2 --> E1[Loss_V = MSE(V_θ(s), V_ϕ(s))]

%% Policy Matching via Action Probs
D1 --> E2[Loss_π = PPO Loss with inner action_probs]

%% Final Parameter Update
E1 --> F[θ ← θ - α∇_θ Loss_V]
E2 --> G[θ ← θ - β∇_θ Loss_π]

%% Notes
subgraph Legend
    note1[Inner → Task-specific adaptation]
    note2[Outer → Generalization across tasks]
end
```
