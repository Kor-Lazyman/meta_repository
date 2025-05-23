# meta_repository
```mermaid
graph TD

%% Task Sampling and Inner Loop
A[Task τᵢ 샘플링] --> B[Inner Loop 시작]

B --> C1[Inner Policy π_ϕ 학습 (PPO)]
B --> C2[Inner Critic V_ϕ 학습 (TD, PPO 등)]

%% Outer Loop Begins
C1 --> D1[Outer Policy π_θ 업데이트]
C2 --> D2[Outer Critic V_θ 업데이트]

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
