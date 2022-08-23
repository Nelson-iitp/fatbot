
#%%
import fatbot as fb
import db6 as db
from trainer import sbalgo, sbname, total_timestepsL, delta_reward, schemes, isd_keys, permute_states
import matplotlib.pyplot as plt

for total_timesteps in total_timestepsL:
    for si,scheme in enumerate(schemes):
        for isd in isd_keys:
            id = f'{sbname}_{si}_{isd}_{total_timesteps}'
            print(f'Testing: [{id}]')
            test_env = db.envF(True, scheme, delta_reward, permute_states, isd)
            fb.TEST(
                env=            test_env, 
                model=          sbalgo.load(id), #<---- use None for random
                episodes=       1,
                steps=          0,
                deterministic=  True,
                render_as=      None, #id,
                save_dpi=       'figure',
                make_video=     False,
                video_fps=      2,
                render_kwargs=dict(local_sensors=True, reward_signal=True),
                starting_state=None,
                plot_results=0,
                cont=False,
            )
            fig=test_env.render()
            fig.savefig(f'{id}.png')
            del fig

#%%





