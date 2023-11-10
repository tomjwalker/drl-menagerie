
1.  **Initialise** learning rate $\alpha$, discount factor $\gamma$, policy network with specified initialisation method $\pi_{\theta}$
2.  **for** episode in {0, ..., MAX_EPISODES} **do**:
3. **for** step in {0, ..., T_episode} **do**:
4. Sample action A_t according to policy $pi(a|s)$
5. Observe reward R_t from environment
6. Calculate policy gradient del(J)
7. **end for**
8. Gradient ascent: $\theta <= \theta + \alpha \cdot \grad J(\pi_{\theta})$
9. **end for**       
