<pre>
1.  **Initialise** learning rate $\alpha$, policy network with specified initialisation method $\pi_{\theta}$
2.  **for** episode in {0, ..., MAX_EPISODES} **do**:
3.  Sample a trajectory \tau = S_0, A_0, R_0, ..., S_T, A_T, R_T$
4.  Set $\nabla{J(\pi_theta)} = 0
5. **for** step in {0, ..., T_episode} **do**:
6. Calculate remaining return, G_t
8. Calculate policy gradient del(J)
9. **end for**
10. Gradient ascent: $\theta \leftarrow \theta + \alpha \cdot \nabla J(\pi_{\theta})$
11. **end for**       
</pre>
