# example of importing env already in mpe   

n_envs = int(n_envs)
n_steps = int(n_steps)
n_agents = 3
env = vmas.make_env(
      "simple_spread",
      device=device,
      num_envs=n_envs,
      continuous_actions=False,
      # Scenario specific config
      n_agents=n_agents,
 )
