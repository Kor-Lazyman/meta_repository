# meta_repository
flowchart TD
    Start([시작: Task 배정])

    subgraph Inner Loop [Inner Policy (φ): Task-specific Adaptation]
        A1[환경과 상호작용\n→ (s_t, a_t, r_t) 수집]
        A2[inner policy log_probs_old 계산]
        A3[inner value function → V^φ(s_t)]
        A4[advantage 계산: A^φ = R - V^φ(s_t)]
        A5[inner memory 저장]
    end

    subgraph Outer Loop [Outer Policy (θ): Meta-Update]
        B1[inner memory에서\n(s_t, a_t, A^φ) 가져오기]
        B2[outer policy로 log_probs_new 계산]
        B3[outer value function V^θ(s_t) 계산]
        B4[ratio = exp(log_probs_new - log_probs_old)]
        B5[policy loss 계산 (PPO surrogate)]
        B6[value loss = MSE(V^θ, reward + γV^φ(s'))]
        B7[총 loss = policy + value + entropy]
        B8[outer policy θ 업데이트]
    end

    End([다음 Task로 이동])

    Start --> A1 --> A2 --> A3 --> A4 --> A5 --> B1
    B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8 --> End
