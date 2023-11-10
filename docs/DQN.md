1. &ensp;&ensp;&ensp;&ensp; **Initialise** learning rate $\alpha$
2. &ensp;&ensp;&ensp;&ensp; **Initialise** exploration hparam, e.g. $\tau$
3. &ensp;&ensp;&ensp;&ensp; **Initialise** number of batches per training step, B
4. &ensp;&ensp;&ensp;&ensp; **Initialise** number of updates per batch, U
5. &ensp;&ensp;&ensp;&ensp; **Initialise** batch size, N
6. &ensp;&ensp;&ensp;&ensp; **Initialise** experience replay memory, with max size K
7. &ensp;&ensp;&ensp;&ensp; **Initialise** according to init method, weights for value network $q_{\theta}$
8. &ensp;&ensp;&ensp;&ensp; **for** m = 0, ..., MAX_STEPS **do**:
9. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Gather and store `h` experiences $(S_i, A_i, R-i, S'_i)$ using current policy
10. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; **for** b = 0, ..., B, **do**:
11. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Sample a batch of experiences from experience replay memory
12. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; **for** u = 0, ..., U, **do**:    # Do U updates with current batch
13. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; # Loop over experiences in batch
14. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;  **for** i = 0, ..., N, **do**:
15. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Calculate target Q-values for each example: $q_{tar:DQN; i} = R_i + \delta_i \gamma \max_{a'}q_{\pi}(S'_i, a'_i)$
16. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; **end for**
17. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Calculate loss between target and value of q network for current state, action, e.g. with MSE Loss
18. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Update network params, e.g. with SGD
19. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; **end for**
20. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; **end for**
21. &ensp;&ensp;&ensp;&ensp; Decay $\tau$
22. &ensp;&ensp;&ensp;&ensp; **end for**
