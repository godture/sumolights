#!/usr/bin/env bash
python hp_optimization.py -sim single -tsc maxpressure -demand linear
python hp_optimization.py -sim single -tsc sotl  -demand linear
python hp_optimization.py -sim single -tsc websters -demand linear
python hp_optimization.py -sim single -tsc uniform -demand linear
#python hp_optimization.py -sim single -tsc dqn -demand linear
#python hp_optimization.py -sim single -tsc dpg -demand linear
