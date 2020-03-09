#!/usr/bin/env bash
for i in {1..4}
do
    python run.py -sim single -n 4 -tsc maxpressure -nogui -mode test -gmin 5 
    python run.py -sim single -n 4 -tsc websters -nogui -mode test -cmax 180 -cmin 40 -f 1800 -satflow 0.44 
    python run.py -sim single -n 4 -tsc uniform -nogui -mode test -gmin 12 
    python run.py -sim single -n 4 -tsc sotl -nogui -mode test -mu 5 -omega 0 -theta 10
    python run.py -sim single -n 4 -tsc dqn -load -nogui -mode test
    python run.py -sim single -n 4 -tsc ddpg -load -nogui -mode test
done
