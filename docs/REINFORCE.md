1.  &ensp;&ensp;&ensp;&ensp;**Initialise** learning rate $\alpha$, policy network with specified initialisation method $\pi_{\theta}$
2.  &ensp;&ensp;&ensp;&ensp;**for** episode in {0, ..., MAX_EPISODES} **do**:
3.  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Sample a trajectory \tau = S_0, A_0, R_0, ..., S_T, A_T, R_T$
4.  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Set $\nabla{J(\pi_theta)} = 0
5. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;**for** step in {0, ..., T_episode} **do**:
6. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Calculate remaining return, G_t
8. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Calculate policy gradient del(J)
9. &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;**end for**
10. &ensp;&ensp;&ensp;&ensp;Gradient ascent: $\theta \leftarrow \theta + \alpha \cdot \nabla J(\pi_{\theta})$
11. &ensp;&ensp;&ensp;&ensp;**end for**       
