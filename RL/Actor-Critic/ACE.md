# ACE: Off-Policy Actor-Critic with Causality-Aware Entropy Regularization

## Motivation

In different stages of task, most significant primitive behaviors are different. 
It's crucial to **emphasize the exploration of the most significant primitive behaviors** at distinct stages of policy learning.

The paper aims to answer: How can we identify the most crucial primitive behaviors at each stage of policy learning?
We need to model the **causal policy-reward structure** to identify the most crucial primitive behaviors.

## Method

Suppose the sequences of observations is $\{s_t,a_t,r_t\}_{t=1}^T$, where $s_t=(s_{1,t},...,s_{\dim S,t})$ is the state, $a_t=(a_{1,t},...,a_{\dim A,t})$ is the action, $r_t$ is the reward.

To integrate the causality into MDP, we explicitly encode causal structures over variables into reward

$$
r_t=r_{M}(B_{s\to r|a}\odot s_t,B_{a\to r|s}\odot a_t, \epsilon_t)
$$

where $B_{s\to r|a}\in\mathbb{R}^{dim S\times 1}$ is the causal influence of $s$ on $r$ given $a$, $B_{a\to r|s}\in\mathbb{R}^{dim A\times 1}$ is the causal influence of $a$ on $r$ given $s$, $\epsilon_t$ is the noise.

### Mathematical Formulation (Not Important)

Why we can formulate reward in this way? The authors make two assumptions:
- Global Markov Condition: if on the graph $G$, $A$ d-separates $S$ from $R$, then $p(S,R|A)=p(S|A)p(R|A)$. (ensures the graph represents valid conditional independencies in the data)
- Faithfulness Assumption: no independencies between variables that are not entailed by Markov condition. (ensures that the graph doesn’t miss or misrepresent any independence relationships)

Based on the two assumption, we can derive following theorem:
- There exists an edge from $a_{i,t}$ to $r_t$ if and only if $a_{i,t}$ is not independent of $r_t$ given $s_t,a_{-i,t}$.
- If $s_t,a_t,r_t$ follows the causal model, then structural vectors $B_{a\to r|s}$ are identifiable.

### Causality-aware Actor-Critic

To infuse explainable causality into policy entropy, propose the causality-aware entropy $\mathcal{H}_c$ for enhanced exploration

$$
\mathcal{H}_c(\pi(\cdot\vert s)) = -\mathbb{E}_{a\in A}\left[\sum_{i=1}^{\dim A}B_{a_i \to r|s}\pi(a_i\vert s) \log \pi(a_i\vert s) \right]
$$

We define the causality-aware Bellman operator as

$$
\mathcal{T}_c^{\pi}Q(s,a) = r(s,a)+\gamma\mathbb{E}_{s'\sim P(\cdot\vert s,a),a'\sim\pi(\cdot|s')}\left[Q(s',a')+\alpha \mathcal{H}_c(\pi(\cdot\vert s))\right]
$$

the causality-aware Bellman operator satisfies
- Policy Improvement: $\mathcal{T}_c^{\pi}Q(s,a)\geq Q(s,a)$
- Policy Evaluation: $Q$ value iteration converges to $Q^{\pi}$
- Policy Iteration: $Q^{\pi^*}\geq Q^{\pi}$ for all $\pi$

With the causality-aware Bellman operator, we can define causality-aware off-policy actor-critic algorithm

The author finds that causality-aware exploration often gets stuck in local optima, overfitting to specific primitive behaviors.

Introduce **Gradient-deomancy-guided Reset**, based on following definitions:

- For a fully connected layer in a neural network, where $N^l$ represents the number of neurons in layer $l$, the L2 norm of gradients of the weights for neuron $i$ is denoted as $n_{i}^l$. Neuron $i$ is classified as a gradient-dormant neuron if it satisfies $\frac{n_i^l(x)}{\frac{1}{N^l}\sum_{k \in l} n_k^l} \leq \tau$, where $\tau$ is a threshold.
- Denote the number of all neurons in the neural network identified as gradient-dormant neurons as $N_{\tau}^l$. The $\alpha_\tau$ for the neural network is defined as: $\alpha_\tau = \frac{\sum_{l \in \phi} N_{\tau}^l}{\sum_{l \in \phi} N^l}$

We periodically perturbing the policy network and critic network with reset factor $\eta$, $\theta_t=(1-\eta)\theta_{t-1}+\eta\phi_i$, where $\phi_i$ is the $i$-th reset, $\eta=clip(\alpha_\tau,0,\eta_{\max})$.

