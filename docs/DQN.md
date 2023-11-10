1. **Initialise** learning rate $\alpha$
2. **Initialise** exploration hparam, e.g. $\tau$
3. **Initialise** number of batches per training step, B
4. **Initialise** number of updates per batch, U
5. **Initialise** batch size, N
6. **Initialise** experience replay memory, with max size K
7. **Initialise** according to init method, weights for value network $q_{\theta}$
8. **for** m = 0, ..., MAX_STEPS **do**:
9. Gather and store `h` experiences $(S_i, A_i, R-i, S'_i)$ using current policy
10. **for** b = 0, ///, B, **do**:
11. Sample a batch of experiences from experience replay memory
12. **for** u = 0, ..., U, **do**:    # Do U updates with current batch
13. **for** i = 0, ..., N, **do**:    # Loop over experiences in batch
14. Calculate target Q-values for each example: $q_{tar:DQN; i} = R_i + \delta_i \gamma \max_{a'}q_{\pi}(S'_i, a'_i)$
15. **end for**
16. Calculate loss between target and value of q network for current state, action, e.g. with MSE Loss
17. Update network params, e.g. with SGD
18. **end for**
19. **end for**
20. Decay $\tau$
21. **end for**
