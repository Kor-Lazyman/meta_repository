# meta_repository
```mermaid
graph TD

%% Task Sampling and Inner Loop
A[Sample Task τᵢ] --> B[Start Inner Loop]

B --> C1[Train Inner Policy (pi_phi) with PPO]
B --> C2[Train Inner Critic (V_phi)]

%% Outer Loop Begins
C1 --> D1[Outer Policy (pi_theta) Update]
C2 --> D2[Outer Critic (V_theta) Update]

%% Value Matching for Critic
D2 --> E1[Value Loss = MSE(V_theta(s), V_phi(s))]

%% Policy Matching via Action Probs
D1 --> E2[Policy Loss = PPO using action_probs from Inner Loop]

%% Final Parameter Update
E1 --> F[Update theta with ∇Loss_V]
E2 --> G[Update theta with ∇Loss_Policy]

%% Notes
subgraph Legend
    note1[Inner Loop = Task-specific Adaptation]
    note2[Outer Loop = Meta-generalization]
end
```

    note1[Inner → Task-specific adaptation]
    note2[Outer → Generalization across tasks]
end
```
