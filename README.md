```mermaid
flowchart TD

%% Task-level Meta-Learning Loop
Start([Start Meta-Learning]) --> SampleTasks[Sample Batch of Tasks "{tau_1, ..., tau_N}"]

%% Inner Loop
SampleTasks --> InnerLoop[For each task tau_i: Run Inner Loop]
InnerLoop --> InnerPolicy[Train Inner Policy "(pi_phi_i)" with PPO]
InnerLoop --> InnerValue[Train Inner Critic "(V_phi_i)"]

%% Outer Loop
InnerPolicy --> PolicyLoss[Compute Outer Policy Loss using action_probs from Inner]
InnerValue --> ValueLoss[Compute Outer Value Loss: MSE"(V_theta(s), V_phi_i(s))"]

PolicyLoss --> UpdatePolicy[Update pi_theta: grad_Loss_Policy]
ValueLoss --> UpdateValue[Update V_theta: grad_Loss_Value]

UpdatePolicy --> Merge[Aggregate gradients across tasks]
UpdateValue --> Merge

Merge --> UpdateParams[Apply meta-update to theta]

UpdateParams --> End([Next Meta-Iteration])
```
