
run => 
	base (check, test, train) 111
	aux  (check, test, train) 111
	final(test, )				1
	
python38 run.py --db=db16 --cs=config --cc=C1 --sd=S --run=1000000
python38 run.py --db=db14 --cs=config --cc=C1 --sd=S --run=1000000
python38 run.py --db=db12 --cs=config --cc=C1 --sd=S --run=1000000
python38 run.py --db=db10 --cs=config --cc=C1 --sd=S --run=1000000
python38 run.py --db=db8 --cs=config --cc=C1 --sd=S --run=1000000
python38 run.py --db=db6 --cs=config --cc=C1 --sd=S --run=1000000


> copy these:
	--> run.py
	--> setup.py
	--> experiments
	--> module

> install locally 
	python -m pip install -e .

> run scripts

	python run.py --db=db16 --cs=config --cc=C1 --sd=S
	python run.py --db=db16 --cs=config --cc=C1 --sd=S
	
nohup python run.py