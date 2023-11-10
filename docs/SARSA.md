1. &ensp;&ensp;&ensp;&ensp; **Initialise** learning rate $\alpha$, exploration hparam e.g. $\epsilon$, value network $q_{\theta}(s, a)$
2. &ensp;&ensp;&ensp;&ensp; **for** m = 0, ..., MAX_STEPS **do**:
3. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Gather N experiences, $(S_i, A_i, R_i, S'_i, A'_i)$
4. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; **for** i = 0, ..., N, **do**:
5. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Calculate the target q values, $q_{tar:SARSA; i} = R_i + \gamma \cdot q_{\pi}(S'_i, A'_i) $
6. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; **end for**
7. &ensp;&ensp;&ensp;&ensp; Calculate loss, e.g. using MSE: $(1/N) * \sum_i{(q_{tar:SARSA; i} - q_{\pi}(S_i, A_i))^2}$
8. &ensp;&ensp;&ensp;&ensp; Update network's params using gradient descent (or better), $\theta \leftarrow \theta - \alpha \nabla_{\theta} J(\theta)$
9. &ensp;&ensp;&ensp;&ensp; Decay $\epsilon$
10. &ensp;&ensp;&ensp;&ensp; **end for** 
