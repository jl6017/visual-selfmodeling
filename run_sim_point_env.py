from sim import ArmPybulletSim
import numpy as np

# Run a very small non-GUI simulation to produce 2 saved meshes and robot_state.json
env = ArmPybulletSim(gui_enabled=False, num_cam=1)
N = 2

# run until we have N saved steps
while True:
    env.reset_everything()
    action = {'robot': np.random.uniform(-1, 1, env._num_joints) * np.pi, 'joint': [0, 1, 2, 3]}
    obs, r, done, _ = env.step(action)
    print('n_steps:', env._n_steps)
    if env._n_steps == N:
        break

env.save_robot_state()
print('Saved robot_state.json to', env.save_mesh_folder)
