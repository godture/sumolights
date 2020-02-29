import itertools, time, os, argparse, shutil

import numpy as np
import pickle

from src.picklefuncs import load_data, save_data
from src.helper_funcs import check_and_make_dir, write_lines_to_file, write_line_to_file, get_time_now

def parse_cl_args():
    parser = argparse.ArgumentParser()

    #multi proc params
    parser.add_argument("-n", type=int, default=9, dest='n', help='number of sim procs (parallel simulations) generating experiences, default: 7')
    parser.add_argument("-l", type=int, default=1, dest='l', help='number of parallel learner procs producing updates, default: 1')

    ##sumo params
    parser.add_argument("-sim", type=str, default='single', dest='sim', help='simulation scenario, default: lust, options:lust, single, double')
    parser.add_argument("-tsc", type=str, default='websters', dest='tsc', help='traffic signal control algorithm, default:websters; options:sotl, maxpressure, dqn, ddpg'  )
    
    #demand
    parser.add_argument("-demand", type=str, default='dynamic', dest='demand', help='vehicle demand generation patter, single limits vehicle network population to one, dynamic creates changing vehicle population, default:dynamic, options:single, dynamic, linear, real')

    args = parser.parse_args()
    return args

def get_hp_dict(tsc_str):
    if tsc_str == 'websters':
        return {'-cmin':[40, 60, 80], '-cmax':[160, 180, 200], '-satflow':[0.3, 0.38, 0.44], '-f':[600, 900]}
    elif tsc_str == 'maxpressure':
        return {'-gmin':np.arange(1,31)}
    elif tsc_str == 'uniform': 
        return {'-gmin':np.arange(1,53)}
    else:
        #raise not found exceptions
        assert 0, 'Error: Supplied traffic signal control argument type '+str(tsc_str)+' does not exist.'

def create_hp_cmds(args, hp_order, hp):
    hp_cmds = []
    cmd_str = 'python run.py -sim '+str(args.sim)+' -nogui -tsc '+str(args.tsc)

    #create hp string
    hp_str = ' '
    for s, v in zip(hp_order, hp):
        hp_str += str(s)+' '+str(v)+' '

    if args.tsc == 'ddpg' or args.tsc == 'dqn':
        #train cmd for rl tsc
        #need to learn before can evaluate hp
        hp_cmds.append(cmd_str+hp_str+'-mode train -save -n '+str(args.n)+' -l '+str(args.l) + ' -demand '+'dynamic')

    #test cmd runs 'n' sims to gen results
    hp_cmds.append(cmd_str+hp_str+'-mode test -n '+str(args.n+args.l))
    #test cmd for rl tsc needs to load saved/learned weights
    if args.tsc == 'ddpg' or args.tsc == 'dqn':
        hp_cmds[-1] += ' -load'

    return hp_cmds

def get_hp_results(fp):
    travel_times = []
    for f in os.listdir(fp):
        travel_times.extend(load_data(fp+f))

    return travel_times

def rank_hp(hp_fitness, hp_order, tsc_str, fp, tt_hp):
    #fitness is the mean+std of the travel time
    ranked_hp_fitness = [ (hp, hp_fitness[hp]['mean']+hp_fitness[hp]['std'], hp_fitness[hp]['n_v_pass']) for hp in hp_fitness]
    ranked_hp_fitness = sorted(ranked_hp_fitness, key=lambda x:x[1]) 
    print('Best hyperparams set for '+str(tsc_str))
    print(hp_order)
    print(ranked_hp_fitness[0])
    
    # write tt_hp of best hp to corresponding file
    tt_file = open(fp[:-4]+'.tt', 'wb')
    pickle.dump(tt_hp[ranked_hp_fitness[0][0]], tt_file)
    tt_file.close()

    #write all hps to file
    #write header line
    lines = [','.join(hp_order)+',mean,std,mean+std,n_v_pass_all_processes']
    #rest of lines are ranked hyperparams
    for hp in ranked_hp_fitness:
        hp_str = hp[0]
        lines.append( hp_str+','+str(hp_fitness[hp_str]['mean'])+','+str(hp_fitness[hp_str]['std'])+','+str(hp[1])+','+str(hp[2]))
    
    write_lines_to_file(fp, 'a+', lines)

def write_temp_hp(hp, results, fp):
    write_line_to_file(fp, 'a+', hp+','+str(results['mean'])+','+str(results['std'])+','+str(results['mean']+results['std']))

def save_hp_performance(data, path, hp_str): 
    check_and_make_dir(path)
    #name the return the unique hp string
    save_data(path+hp_str+'.p', data)

def main():
    start = time.time()

    args = parse_cl_args()

    #get hyperparams for command line supplied tsc
    tsc_str = args.tsc
    hp_dict = get_hp_dict(tsc_str)
    hp_order = sorted(list(hp_dict.keys()))

    hp_list = [hp_dict[hp] for hp in hp_order]
    #use itertools to produce cartesian product of hyperparams
    hp_set = list(itertools.product(*hp_list))
    print(str(len(hp_set))+' total hyper params')

    #where to find metrics
    hp_travel_times = {}
    metrics_fp = 'metrics/'+tsc_str

    #where to print hp results
    path = 'hyperparams/'+tsc_str+'/'
    check_and_make_dir(path)
    
    # hp_optimize for real cycle
    fname = 'real'
    hp_fp = path+fname+'.csv'
    write_line_to_file(hp_fp, 'a+', ','.join(hp_order)+',mean,std,mean+std' )
    tt_hp = {} # store all travel_times for corresponding hp
    for hp in hp_set:    
        hp_cmds = create_hp_cmds(args, hp_order, hp)
        #print(hp_cmds)
        travel_times = []

        if args.demand == 'linear':
            assert False, 'demand should be real!!'
        elif args.demand == 'dynamic':
            assert False, 'demand should be real!!'
        elif args.demand == 'real':
            cmd_test = hp_cmds[-1] + ' -demand ' + 'real'
        else:
            assert False, 'Please only give demand: real'
        os.system(cmd_test)
        #read travel times, store mean and std for determining best hp set
        hp_str = ','.join([str(h) for h in hp])
        travel_times += get_hp_results(metrics_fp+'/traveltime/')
        tt_hp[hp_str] = travel_times
        # !!!!!!!!!! the travel_times is cumulated in 10 processes


        n_v_pass = len(travel_times)
        #remove all metrics for most recent hp
        shutil.rmtree(metrics_fp)
        hp_travel_times[hp_str] = {'mean':int(np.mean(travel_times)), 'std':int(np.std(travel_times)), 'n_v_pass':n_v_pass}
        write_temp_hp(hp_str, hp_travel_times[hp_str], hp_fp)
        #generate_returns(tsc_str, 'metrics/', hp_str)
        save_hp_performance(travel_times, 'hp/'+tsc_str+'/', hp_str) 

    #remove temp hp and write ranked final results
    os.remove(hp_fp)
    rank_hp(hp_travel_times, hp_order, tsc_str, hp_fp, tt_hp)
    
    print('All hyperparamers performance can be viewed at: '+str(hp_fp))

    print('TOTAL HP SEARCH TIME')
    secs = time.time()-start
    print(str(int(secs/60.0))+' minutes ')
    
    
    '''
    # hp_optimize for each linear cycle
    for idx_cycle in range(30):
        fname = 'cycle_'+str(idx_cycle).zfill(2)
        hp_fp = path+fname+'.csv'
        write_line_to_file(hp_fp, 'a+', ','.join(hp_order)+',mean,std,mean+std' )
        #run each set of hp from cartesian product
        tt_hp = {} # store all travel_times for corresponding hp
        for hp in hp_set:    
            hp_cmds = create_hp_cmds(args, hp_order, hp)
            #print(hp_cmds)
            travel_times = []
            # only train the ddpg and dqn with the predefined sine wave
            if args.tsc == 'ddpg' or args.tsc == 'dqn':
                os.system(hp_cmds[0])

            if args.demand == 'linear':
                cmd_test = hp_cmds[-1] + ' -demand ' + 'linear_' + str(idx_cycle).zfill(2)
            elif args.demand == 'dynamic':
                cmd_test = hp_cmds[-1] + ' -demand ' + 'dynamic'
            elif args.demand == 'real':
                cmd_test = hp_cmds[-1] + ' -demand ' + 'real'
            else:
                assert False, 'Please only give demand: linear, dynamic or real'
            os.system(cmd_test)
            #read travel times, store mean and std for determining best hp set
            hp_str = ','.join([str(h) for h in hp])
            travel_times += get_hp_results(metrics_fp+'/traveltime/')
            tt_hp[hp_str] = travel_times
            # !!!!!!!!!! the travel_times is cumulated in 8 processes
            
            
            n_v_pass = len(travel_times)
            #remove all metrics for most recent hp
            shutil.rmtree(metrics_fp)
            hp_travel_times[hp_str] = {'mean':int(np.mean(travel_times)), 'std':int(np.std(travel_times)), 'n_v_pass':n_v_pass}
            write_temp_hp(hp_str, hp_travel_times[hp_str], hp_fp)
            #generate_returns(tsc_str, 'metrics/', hp_str)
            save_hp_performance(travel_times, 'hp/'+tsc_str+'/', hp_str) 

        #remove temp hp and write ranked final results
        os.remove(hp_fp)
        rank_hp(hp_travel_times, hp_order, tsc_str, hp_fp, tt_hp)
        
        
        
        # ??? 记得把simlen 调到3600
        
        # ??? 根据30 cycle 的结果调整hyper parameter 的范围，主要是maxpressure 和 uniform
        
        # ?????? n_vehilce_passed 
        # ?????? n_vehilce_passed
        # ?????? n_vehilce_passed
        # ?????? n_vehilce_passed
        # ?????? n_vehilce_passed
        # ?????? n_vehilce_passed
        # ?????? n_vehilce_passed
        
        
        
        # ???? 找个地方把每个cycle 的best hyperparameter， mean, std, n_vehilce_passed 存起来
        

        
        
        
        print('All hyperparamers performance can be viewed at: '+str(hp_fp))

        print('TOTAL HP SEARCH TIME')
        secs = time.time()-start
        print(str(int(secs/60.0))+' minutes ')
    '''
    
    '''
    #run each set of hp from cartesian product
    for hp in hp_set:    
        hp_cmds = create_hp_cmds(args, hp_order, hp)
        #print(hp_cmds)
        if args.demand == 'linear':
            n_repeat = 30
        else:
            n_repeat = 1
        travel_times = []
        # only train the ddpg and dqn with the predefined sine wave
        if args.tsc == 'ddpg' or args.tsc == 'dqn':
            os.system(hp_cmds[0])
        for i in range(n_repeat):
            if args.demand == 'linear':
                cmd_test = hp_cmds[-1] + ' -demand ' + 'linear_' + str(i).zfill(2)
            elif args.demand == 'dynamic':
                cmd_test = hp_cmds[-1] + ' -demand ' + 'dynamic'
            elif args.demand == 'real':
                cmd_test = hp_cmds[-1] + ' -demand ' + 'real'
            else:
                assert False, 'Please only give demand: linear, dynamic or real'
            os.system(cmd_test)
            #read travel times, store mean and std for determining best hp set
            hp_str = ','.join([str(h) for h in hp])
            travel_times += get_hp_results(metrics_fp+'/traveltime/')
            #remove all metrics for most recent hp
            shutil.rmtree(metrics_fp)
        hp_travel_times[hp_str] = {'mean':int(np.mean(travel_times)), 'std':int(np.std(travel_times))}
        write_temp_hp(hp_str, hp_travel_times[hp_str], hp_fp)
        #generate_returns(tsc_str, 'metrics/', hp_str)
        save_hp_performance(travel_times, 'hp/'+tsc_str+'/', hp_str) 
        
    #remove temp hp and write ranked final results
    os.remove(hp_fp)
    rank_hp(hp_travel_times, hp_order, tsc_str, hp_fp)
    print('All hyperparamers performance can be viewed at: '+str(hp_fp))

    print('TOTAL HP SEARCH TIME')
    secs = time.time()-start
    print(str(int(secs/60.0))+' minutes ')
    
    '''

if __name__ == '__main__':
    main()
