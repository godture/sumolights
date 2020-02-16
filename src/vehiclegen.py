import os, sys
import pickle as pk
from scipy.stats import johnsonsb
from scipy.stats import johnsonsu
F_RATES = [60., 200., 500., 800., 1100., 1400., 1700.] # typical flow rates (v/h) with headway distribution models

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

import numpy as np

class VehicleGen:
    def __init__(self, netdata, sim_len, demand, scale, mode, conn):
        np.random.seed()
        self.conn = conn
        self.v_data = None
        self.vehicles_created = 0
        self.netdata = netdata
        ###for generating vehicles
        self.origins = self.netdata['origin']
        self.destinations = self.netdata['destination'] 
        self.add_origin_routes()
        self.scale = scale
        self.sim_len = sim_len
        self.t = 0
        self.stop_gen = False
        self.demand = demand

        ###determine what function we run every step to 
        ###generate vehicles into sim
        if demand == 'single':
            self.gen_vehicles = self.gen_single
        elif demand == 'dynamic' or mode =='train':  # use the sine wave to train rl-tsc in framework
            self.v_schedule = self.gen_dynamic_demand(mode)
            self.gen_vehicles = self.gen_dynamic
        elif demand[:6] == 'linear':
            self.gen_vehicles = self.gen_linear_cycle
        elif demand == 'real':
            self.gen_vehicles = self.gen_real_cycle

    def run(self):
        if not self.stop_gen:
            self.gen_vehicles()
        self.t += 1

    def gen_dynamic(self):
        ###get next set of edges from v schedule, use them to add new vehicles
        ###this is batch vehicle generation
        try:
            new_veh_edges = next(self.v_schedule)
            self.gen_veh( new_veh_edges  )
        except StopIteration:
            print('no vehicles left')
#############################################################################

    def headway_j(self,flow_rate):
        assert flow_rate >= F_RATES[0]/2, f'Input flow rate ({flow_rate}) should not be smaller than {F_RATES[0]/2}v/h for calculating the headway!'
        #assert flow_rate <= F_RATES[-1], f'Input flow rate ({flow_rate}) should not be bigger than {F_RATES[-1]}v/h, the saturation rate of one lane!'
        headway = {F_RATES[0]: lambda: johnsonsb.rvs(0.3, 1.8, loc=0.85, scale=130),
                   F_RATES[1]: lambda: johnsonsb.rvs(0.9, 0.71, loc=0.85, scale=60), 
                   F_RATES[2]: lambda: johnsonsb.rvs(2.49, 0.71, loc=0.85, scale=104.57), 
                   F_RATES[3]: lambda: johnsonsb.rvs(3.71, 0.98, loc=0.85, scale=104.57), 
                   F_RATES[4]: lambda: johnsonsu.rvs(-2.18, 1.15, loc=0.8, scale=0.52), 
                   F_RATES[5]: lambda: johnsonsu.rvs(-2.54, 1.31, loc=0.47, scale=0.46), 
                   F_RATES[6]: lambda: johnsonsu.rvs(-2.49, 1.49, loc=0.55, scale=0.49)}
        if flow_rate <= F_RATES[0]:
            result = min(900, headway[F_RATES[0]]() * F_RATES[0] / flow_rate) # headyway should be smaller than 900s, which is the normal measure period of traffic flow
            return result
        for i in range(1, len(F_RATES)):
            if flow_rate <= F_RATES[i]:
                ratio = (flow_rate-F_RATES[i-1])/(F_RATES[i]-F_RATES[i-1])
                result = (1-ratio) * headway[F_RATES[i-1]]() + ratio * headway[F_RATES[i]]()
                result = max(result, 0)
                return result
        result = max(headway[F_RATES[-1]]() * F_RATES[-1] / flow_rate, 0) # headyway should be bigger than 0
        return result
    
    def addVehicle(self, id_route, id_vehicle_type, time_depart, depart_speed='desired'): # the option 'desired' for departSpeed is only available since ubuntu 18.04, use 'max' for ubuntu version 16.04
        id_v = 'v_' + id_route + '_' + str(self.vehicles_created)	# !!! naming rule for vehicle id: v_<route id>_<vehicle id>
        self.conn.vehicle.add(id_v, id_route, 'car_a', depart=time_depart, departLane="best")
        self.vehicles_created += 1
        self.conn.vehicle.setColor(id_v, self.color_routes[id_route])
    
    # only for test mode, not for training  
    def gen_fr_cycle(self, type):
        self.routes = self.conn.route.getIDList()
        self.routes = [route for route in self.routes if route[0]=='r']
        self.color_routes = {self.routes[0]: (0,255,0), self.routes[1]: (255,255,0), self.routes[2]: (255,0,0), self.routes[3]: (0,255,0), 
                             self.routes[4]: (255,255,0), self.routes[5]: (255,0,0), self.routes[6]: (0,255,0), self.routes[7]: (255,255,0), 
                             self.routes[8]: (255,0,0),self.routes[9]: (0,255,0), self.routes[10]: (255,255,0), self.routes[11]: (255,0,0)} # colors
        v_gen_file = None
        if type == "linear":
            list_files = os.listdir("/home/yan/work_spaces/sumolights/tf_test/linear")
            list_files = [item for item in list_files if item[-3:]=='.vg']
            tf_level = list_files[0][:4]
            v_gen_file = open("/home/yan/work_spaces/sumolights/tf_test/linear/" + tf_level + '_' + self.demand[-2:]+'.vg', 'rb')
            #tf_file = open("/home/yan/work_spaces/sumolights/tf_test/linear/" + np.random.choice(list_files), 'rb')
        elif type == "real":
            list_files = os.listdir("/home/yan/work_spaces/sumolights/tf_test/real")
            list_files = [item for item in list_files if item[-3:]=='.vg']
            v_gen_file = open("/home/yan/work_spaces/sumolights/tf_test/real/" + np.random.choice(list_files), 'rb')
        start_time_routes = pk.load(v_gen_file)
        print(f'################# v_gen_file is:\n {v_gen_file}')
        print(f'@@@@@@@@@@@@@ length start_time_routes is: \n{len(start_time_routes)}')
        print(f'$$$$$$$$$$$$$ routs are:\n{self.routes}')
        '''
        for i in range(len(self.routes)):
            route = self.routes[i]
            t_generate = 0
            while t_generate < len(flow_rate_routes):
                if flow_rate_routes[t_generate][i] < F_RATES[0]:
                    t_generate += 1
                elif len(self.start_time_routes[route]) == 0:
                    self.start_time_routes[route].append(t_generate + self.headway_j(flow_rate_routes[t_generate][i]))
                    t_generate = int(round(self.start_time_routes[route][-1]))
                else:
                    self.start_time_routes[route].append(self.start_time_routes[route][-1] + self.headway_j(flow_rate_routes[t_generate][i]))
                    t_generate = int(round(self.start_time_routes[route][-1]))
        '''
        
        for route in self.routes:
            [self.addVehicle(route, 'car_a', start_time) for start_time in start_time_routes[route]]
        self.stop_gen = True
        
    def gen_linear_cycle(self):
        self.gen_fr_cycle("linear")
        
    def gen_real_cycle(self):
        self.gen_fr_cycle("real")
        
        


##############################################################################
    def gen_dynamic_demand(self, mode):
        ###use sine wave as rate parameter for dynamic traffic demand
        t = np.linspace(1*np.pi, 2*np.pi, self.sim_len)                                          
        sine = np.sin(t)+1.55
        ###create schedule for number of vehicles to be generated each second in sim
        v_schedule = []
        second = 1.0
        for t in range(int(self.sim_len)):
            n_veh = 0.0
            while second > 0.0:
                headway = np.random.exponential( sine[t], size=1)
                second -= headway
                if second > 0.0:
                    n_veh += 1
            second += 1.0
            v_schedule.append(int(n_veh))
                                                                                            
        ###randomly shift traffic pattern as a form of data augmentation
        v_schedule = np.array(v_schedule)
        if mode == 'test':
            random_shift = 0
        else:
            random_shift = np.random.randint(0, self.sim_len)
        v_schedule = np.concatenate((v_schedule[random_shift:], v_schedule[:random_shift]))
        ###zero out the last minute for better comparisons because of random shift
        v_schedule[-60:] = 0
        ###randomly select from origins, these are where vehicles are generated
        v_schedule = [ np.random.choice(self.origins, size=int(self.scale*n_veh), replace = True) 
                       if n_veh > 0 else [] for n_veh in v_schedule  ]
        print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ v number is:\n{sum([len(item) for item in v_schedule])}')
        ###fancy iterator, just so we can call next for sequential access
        return v_schedule.__iter__() 

    def add_origin_routes(self):
        for origin in self.origins:
            self.conn.route.add(origin, [origin] )

    def gen_single(self):
        if self.conn.vehicle.getIDCount() == 0:
            ###if no vehicles in sim, spawn 1 on random link
            veh_spawn_edge = np.random.choice(self.origins)
            self.gen_veh( [veh_spawn_edge] )

    def gen_veh( self, veh_edges ):
        for e in veh_edges:
            vid = e+str(self.vehicles_created)
            self.conn.vehicle.addFull( vid, e, departLane="best" )
            self.set_veh_route(vid)
            self.vehicles_created += 1

    def set_veh_route(self, veh):
        current_edge = self.conn.vehicle.getRoute(veh)[0]
        route = [current_edge]
        while current_edge not in self.destinations:
            next_edge = np.random.choice(self.netdata['edge'][current_edge]['outgoing'])
            route.append(next_edge)
            current_edge = next_edge
        self.conn.vehicle.setRoute( veh, route )    
