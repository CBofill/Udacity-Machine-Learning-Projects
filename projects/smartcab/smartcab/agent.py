import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
	"""An agent that learns to drive in the smartcab world."""

	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
		self.Q = {None:{None:2.}}
		self.alpha = 0.1
		self.gamma = 0.01
		self.epsilon =  0.5
		self.decay_rate = 0.005
		self.last_update = (None,None,0.0)

	def reset(self, destination=None):
		self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
		inputs = self.env.sense(self)
		
		deadline = self.env.get_deadline(self)
        
		# TODO: Update state
		oncoming = "None"
		left = "None"
		right = "None"
		if inputs['oncoming']:
			oncoming = inputs['oncoming']
		if inputs['left']:
			left = inputs['left']
		if inputs['right']:
			right = inputs['right']
		
		self.state = self.next_waypoint+inputs['light']+oncoming+right+left
		
		""" This code updates the state in the simplest possible fashion
		self.state = self.next_waypoint + " " + inputs['light']
		if self.next_waypoint == 'right' and inputs['light'] == 'red' and inputs['left'] == 'forward':
			self.state += ' cannot_turn_right'
		if self.next_waypoint == 'left' and inputs['light'] == 'green':
			if inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right':
				self.state += ' cannot_turn_left'
		"""
		
		# Initialize new states in Q
		if self.state not in self.Q:
			self.Q[self.state] = {None:2., 'forward':2.,'left':2.,'right':2.}
		
		# TODO: Select action according to your policy
		max_Q = max(self.Q[self.state],key=self.Q[self.state].get)
		
		# Take a random action with 1-ε probability
		if random.random() > self.epsilon:
			action = max_Q
		else:
			action = [None, 'forward','left','right'][random.randint(0,3)]
        
		# Execute action and get reward 
		reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

		# Determine Q estimate; Q(s,a) = (1-α)*Q(s,a) + α*(reward + γ*Q(ŝ,â))
		X = self.last_update[2] + self.gamma * self.Q[self.state][max_Q] # X = reward + γ*Q(ŝ,â)
		self.Q[self.last_update[0]][self.last_update[1]] = (1-self.alpha)*self.Q[self.last_update[0]][self.last_update[1]] + self.alpha*X # Q(s,a) = (1-α)*Q(s,a) + α*X
		
		# Store s,a,reward for next Q estimate
		self.last_update = (self.state,action,reward)
		
		# Decay exploration rate
		self.epsilon -= self.decay_rate
		
		print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
		#print "State is {}, maxQ is {}, Q table is {}".format(self.state,max_Q,self.Q[self.state])
	
	# Print every state and the corresponding Q values of each action
	def print_Q(self):
		for key in self.Q:
			print key, " ",self.Q[key]
		

def run():
	"""Run the agent for a finite number of trials."""

    # Set up environment and agent
	e = Environment()  # create environment (also adds some dummy traffic)
	a = e.create_agent(LearningAgent)  # create agent
	e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
	sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

	sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
	
	# Print final Q table
	#a.print_Q()


if __name__ == '__main__':
    run()
