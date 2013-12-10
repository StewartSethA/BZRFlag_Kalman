# This agent is supposed to defy the Kalman Filter's predictions, making an agent that is hard to hit.

import sys
import math
import time
import random

from bzrc import BZRC, Command

class Agent(object):
    """Class handles all command and control logic for a teams tanks."""

    def __init__(self, bzrc):
        self.bzrc = bzrc
        self.constants = self.bzrc.get_constants()
        self.commands = []
        self.movedir = 1
        self.whenMove = random.randint(1,7)
        self.whenAccel = random.randint(0,4)
       	self.moveX = random.randint(-350,350)
      	self.moveY = random.randint(-350,350)

        self.clock = 0
        self.clock1 = 0
        self.last_time_diff = 0
        self.acceleration = random.random() * 25

    def tick(self, time_diff):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks
        self.othertanks = othertanks
        self.flags = flags
        self.shots = shots
        self.enemies = [tank for tank in othertanks if tank.color !=
                        self.constants['team']]
        
        tank = self.mytanks[0]
        
        #print self.clock , self.last_time_diff
        self.clock = self.clock + (time_diff - self.last_time_diff)
        if self.clock > self.whenMove:
            self.clock = 0
            self.whenMove = random.randint(0,5)
            self.moveX = random.randint(-350,350)
            self.moveY = random.randint(-350,350)
            print "New target: ", self.moveX, ",", self.moveY

        self.clock1 = self.clock1 + (time_diff - self.last_time_diff)
        if self.clock1 > self.whenAccel:
        	self.clock1 = 0
        	self.acceleration = random.random() * 25
        	self.whenAccel = random.randint(0,4)
        	trickstop = random.random()
        	if trickstop < 0.10:
        		self.acceleration = 0
        		print "HAHA!"
        	if trickstop > 0.6:
        		self.acceleration = 25
        		print "Speedy Goose!"
        	elif trickstop > 0.4:
        		self.acceleration += 10

        self.last_time_diff = time_diff
        self.move_to_position(tank, self.moveX, self.moveY)

        results = self.bzrc.do_commands(self.commands)

    def move_to_position(self, tank, target_x, target_y):
        """Set command to move to given coordinates."""
        target_angle = math.atan2(target_y - tank.y,
                                  target_x - tank.x)
        relative_angle = self.normalize_angle(target_angle - tank.angle)
        command = Command(tank.index, self.acceleration, relative_angle, False)
        self.commands.append(command)

    def normalize_angle(self, angle):
        """Make any angle be between +/- pi."""
        angle -= 2 * math.pi * int (angle / (2 * math.pi))
        if angle <= -math.pi:
            angle += 2 * math.pi
        elif angle > math.pi:
            angle -= 2 * math.pi
        return angle


def main():
    # Process CLI arguments.
    try:
        execname, host, port = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print >>sys.stderr, '%s: incorrect number of arguments' % execname
        print >>sys.stderr, 'usage: %s hostname port' % sys.argv[0]
        sys.exit(-1)

    # Connect.
    bzrc = BZRC(host, int(port))

    agent = Agent(bzrc)

    prev_time = time.time()

    # Run the agent
    try:
        while True:
			time_diff = time.time() - prev_time
			agent.tick(time_diff)
    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()


if __name__ == '__main__':
    main()