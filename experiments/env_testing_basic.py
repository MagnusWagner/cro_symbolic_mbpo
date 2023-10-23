from simulation_env.environment_basic import CropRotationEnv

def test_print():
    # Initialize crop rotation environment
    env = CropRotationEnv()
    env.render()
    
def test_random():
    env = CropRotationEnv()
    # Generate 5 random crop rotations without training (for environment testing)
    episodes = 100
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        while not done:
            action = env.action_space.sample()
            n_state, _, reward, done, info = env.step(action)
            score+=reward
            # pp.pprint(info)

        print('Episode:{} Score:{}'.format(episode, score))