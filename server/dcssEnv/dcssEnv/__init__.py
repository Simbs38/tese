from gym.envs.registration import register

register(
	id="dcss-v0",
	entry_point ='dcssEnv.envs:DcssEnv',
)