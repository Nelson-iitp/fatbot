
import fatbot as fb
import db6 as db
from trainer import sbalgo, sbname
import matplotlib.pyplot as plt
total_timestepsL=    [50_000, 150_000]

for total_timesteps in total_timestepsL:
    for delta_reward in (False, True):
        for si,scheme in enumerate((db.scheme_1, db.scheme_2, db.scheme_3)):
            for isd in db.isd_keys:
                id = f'{sbname}_{si}_{isd}_{total_timesteps}_{delta_reward}'
                print(f'Testing: [{id}]')
                test_env = db.envF(True, scheme, delta_reward, isd)
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





