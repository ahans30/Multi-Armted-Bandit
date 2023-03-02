import numpy as np

class MultiArmedBandit(object):
    
    def __init__(self, 
                 k = 10, 
                 # initial_values = 'peek_once', # optimistic and zero  
                 stationery = True,
                ):
        '''
        k = number of bandits
        stationery = indicates is underlying actual dsitribution of arms's rewards function (q*) is stable / non-moving
        'loc' and 'scale' are mean and standard distribution of gaussian distribution (NumPy naming convention used)
         we numerically analyze learning algorithm’s average behavior over N random experiments.
        '''
        self.k = k
        self.stationery = stationery
        self.actual_distribution_parameters = None
        
    def _initialize_actual_distribution(self):
        self.actual_distribution_parameters = dict(zip(list(range(self.k)), [{'loc': np.random.uniform(-0.25,2), 'scale': np.random.uniform(2,5)} for _ in range(self.k)]))
        if not self.stationery:
            self.distribution_scaling_parameters = dict(zip(list(range(self.k)), \
                                                            [{'loc_multiplier': np.random.choice([1,-1]) + 0.005, \
                                                              'scale_multiplier': np.random.choice([1,1-0.0001]) + 0.0001} for _ in range(self.k)]))
            
    def get_immediate_reward(self, arm_selected):
        return np.random.normal(self.actual_distribution_parameters[arm_selected]['loc'], \
                                self.actual_distribution_parameters[arm_selected]['scale'])
    
    def get_optimal_arm(self):
        '''Optimal arm = arm with largest mean/expected value. This is static in case of 
        stationery problems but changes as function of time for non-stationery.
        
        Please note, this requires Python >=3.7 dict as it holds the keys in the 
        order of insertion and hence np.argmax would yield the arm with largest mean (loc)'''
        
        
        return np.argmax([self.actual_distribution_parameters[i]['loc'] for i in range(self.k)])
    
    def _mutate_underlying_distribution(self, arm = None):
        if arm is None:
            # We scale all arms
            for arm in range(self.k):
                self.actual_distribution_parameters[arm]['loc'] *=  self.distribution_scaling_parameters[arm]['loc_multiplier']
                self.actual_distribution_parameters[arm]['scale'] *=  self.distribution_scaling_parameters[arm]['scale_multiplier']
        
        else:
            # We only scale specific arm. This can be the arm that has been selected for payout perhaps (ideally unknown to learning algorithm.)
            self.actual_distribution_parameters[arm]['loc'] *=  self.distribution_scaling_parameters[arm]['loc_multiplier']
            self.actual_distribution_parameters[arm]['scale'] *=  self.distribution_scaling_parameters[arm]['scale_multiplier']
            
    def get_initial_values(self, method):
        if method == 'zeros':
            return np.array([0] * self.k)
        elif method == 'optimistic':
            return np.array([5] * self.k)
        elif method == 'peek_once':
            return np.array([self.get_immediate_reward(i) for i in range(self.k)])      
        
    def select_arm(self, strategy, epsilon):
        
        arm_selected = None
        
        if strategy== 'epsilon_greedy':
            choice = np.random.choice(['Explore', 'Exploit'], p = [epsilon, 1-epsilon])
            
            if choice == 'Exploit':
                arm_selected = np.argmax(self.estimated_values)
            elif choice == 'Explore':
                arm_selected = np.random.choice(range(0, self.k))
            
        elif strategy== 'UCB':
            arm_selected = np.argmax(self.ucb_score)
            
        return arm_selected
    
    def update_estimate(self, strategy, arm_selected, payout, step_size, arms_pulled_frequency, t, c = 2):
            self.estimated_values[arm_selected] = self.estimated_values[arm_selected] + step_size * (payout - self.estimated_values[arm_selected])
            
            if strategy == 'UCB':
                self.ucb_score[arm_selected] = self.estimated_values[arm_selected] + c * ((np.log(t) / 1 + arms_pulled_frequency[arm_selected]) ** 0.5)
            
        
            
    def run(self, 
            strategy = 'epsilon_greedy', 
            epsilon = 0.1,
            T = 1000, 
            initial_values_method = 'peek_once', 
            estimation_method = 'sample_average', #can be exponential recently weighted average with constant step_size
            stepsize = None,
            c = 2):
        
        self._initialize_actual_distribution()
        self.estimated_values = self.get_initial_values(initial_values_method)
        if strategy == 'UCB': self.ucb_score = self.estimated_values.copy()
        
        arms_pulled_frequency = [1] * self.k #if initial_values_method == 'zeros' else [1] * self.k 
        
        total_reward_collected = 0
        average_reward = []
        
        optimal_action_sum = 0
        optimal_action = []
        
        total_regret = 0
        regret_vector = []
        
        for t in range(1, T+1):
            arm_selected = self.select_arm(strategy, epsilon)
            arms_pulled_frequency[arm_selected] += 1
            
            if arm_selected == self.get_optimal_arm():
                optimal_action_sum += 1
                
            optimal_action.append(optimal_action_sum / t)

            payout = self.get_immediate_reward(arm_selected)
            total_reward_collected += payout
            average_reward.append(total_reward_collected / t)
            
            if not self.stationery: self._mutate_underlying_distribution()
            if estimation_method == 'sample_average': step_size = 1 / arms_pulled_frequency[arm_selected]
            
            self.update_estimate(strategy, arm_selected, payout, step_size, arms_pulled_frequency, t, c)
            
            total_regret += self.actual_distribution_parameters[self.get_optimal_arm()]['loc'] - self.estimated_values[arm_selected]
            regret_vector.append(total_regret)
            
        return {'average_reward': np.array(average_reward),  \
                'optimal_action': np.array(optimal_action),
                'total_regret': np.array(regret_vector)}
    
    
    def average_behavior(self, 
                         N = 100,
                         strategy = 'epsilon_greedy', 
                         epsilon = 0.1,
                         T = 1000, 
                         initial_values_method = 'peek_once', 
                         estimation_method = 'sample_average', #can be exponential recently weighted average with constant step_size
                         stepsize = None):
        
        
        average_reward = np.zeros(T)
        optimal_action = np.zeros(T)
        total_regret = np.zeros(T)
        
        for n in range(N):
            output = self.run(strategy, epsilon, T, initial_values_method, estimation_method, stepsize)
            average_reward += output['average_reward']
            optimal_action += output['optimal_action']
            total_regret += output['total_regret']
            
        average_reward /= N
        optimal_action /= N
        total_regret /= N
        
        return {'average_reward': average_reward, 'optimal_action': optimal_action, 'total_regret': total_regret}
            
            
        

        
        
        
        
            
            
    

        
        
                 