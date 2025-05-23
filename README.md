# meta_repository
```mermaid
graph TD

%% Task Sampling and Inner Loop
A[Sample Task tau_i] --> B[Start Inner Loop]

B --> C1[Train Inner Policy pi_phi using PPO]
B --> C2[Train Inner Critic V_phi]

%% Outer Loop Begins
C1 --> D1[Compute Outer Policy Loss using action_probs from Inner]
C2 --> D2[Compute Outer Value Loss: MSE(V_theta(s), V_phi(s))]

%% Final Parameter Update
D1 --> E1[Update Policy pi_theta ← pi_theta - alpha * grad_Loss_Policy]
D2 --> E2[Update Value V_theta ← V_theta - beta * grad_Loss_Value]

%% Notes
subgraph Explanation
    note1[Inner Loop = Fast adaptation per task]
    note2[Outer Loop = Meta-update for generalization]
end
```

```

