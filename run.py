
__doc__="""

#----experiments
    |----db6    - multiple dbx,  use (--db) arg to select
        |----isd - use (--sd) arg to select the initial state distribution
            |----D.png
            |----P.png
            |----S.png
        |----config.py - there may be multiple 'config.py' files,  
                            use (--cs) arg to select the python file
                            use (--cc) arg to select the config class
        |----db.py
    |----db8
        |----isd
            |----D.png
            |----P.png
            |----S.png
        |----config.py
        |----db.py

    
"""

"""

    python38 run.py --db=db16 --cs=config --cc=C1 --sd=S --run=1000000
    python38 run.py --db=db14 --cs=config --cc=C1 --sd=S --run=1000000
    python38 run.py --db=db12 --cs=config --cc=C1 --sd=S --run=1000000
    python38 run.py --db=db10 --cs=config --cc=C1 --sd=S --run=1000000
    python38 run.py --db=db8 --cs=config --cc=C1 --sd=S --run=1000000
    python38 run.py --db=db6 --cs=config --cc=C1 --sd=S --run=1000000



"""

import os
results_dir = '__results__'
os.makedirs(results_dir, exist_ok=True)
os.chdir(results_dir)

import argparse
import importlib
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='1111111', 
        help="""
        check_base 0
        train_base 1
        test_base 2
        check_aux 3
        train_aux 4
        test_aux 5
        test_final 6
        """)

parser.add_argument('--db', type=str, default='db6',    help='data base')
parser.add_argument('--cs', type=str, default='config', help='config set')
parser.add_argument('--cc', type=str, default='C1',     help='config class')
parser.add_argument('--sd', type=str, default='S',      help='state distribution')
argc = parser.parse_args()
runs = [bool(int(i)) for i in argc.run]

cs = importlib.import_module(f'experiments.{argc.db}.{argc.cs}')
cc = getattr(cs, argc.cc)
module = cc(alias=f'{argc.db}_{argc.cs}_{argc.cc}_{argc.sd}', common_initial_states=argc.sd)



if runs[0]: module.do_check_base(episodes=5, steps=1)
if runs[1]: module.do_train_base()
if runs[2]: module.do_test_base()
if runs[3]: module.do_check_aux(episodes=5, steps=1)
if runs[4]: module.do_train_aux()
if runs[5]: module.do_test_aux()
if runs[6]: module.do_final_test()