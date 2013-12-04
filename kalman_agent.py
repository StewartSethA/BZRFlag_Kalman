# This agent is stationary (cannot move) but can change its angle to track and shoot at enemy tanks.
# It implements a Kalman Filter to track and predict the movements of other tanks
# And also creates plots (via either GNUPlot or OpenGL) of the probability distributions in order to visualize the efficacy of the Kalman Filter.

import sys
import math
import time
import numpy
from numpy import ones
import random
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import zeros

from bzrc import BZRC, Command

class Agent(object):
    """Class handles all command and control logic for a teams tanks."""

    def __init__(self, bzrc):
        self.bzrc = bzrc
        self.constants = self.bzrc.get_constants()
        self.commands = []
        self.mytankdata = []
        mytanks = self.bzrc.get_mytanks()
        for tank in mytanks:
            self.mytankdata.append((0.0, 0.0)) # push initial speed_error and angle_error onto list for each tank
        
    def tick(self, time_diff):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks
        self.othertanks = othertanks
        self.flags = flags
        self.shots = shots
        self.enemies = [tank for tank in othertanks if tank.color !=
                        self.constants['team']]
        self.commands = []
        self.mu_t = numpy.zeros(6)
        self.sigma_t = 100.0 * numpy.identity(6, float)
        self.sigma_t[1][1] = 0.1
        self.sigma_t[2][2] = 0.1
        self.sigma_t[4][4] = 0.1
        self.sigma_t[5][5] = 0.1
        self.physics = numpy.identity(6, float)
        self.state_noise = 0.1 * numpy.identity(6, float)
        self.state_noise[2][2] = 100
        self.state_noise[5][5] = 100
        self.observation_matrix = numpy.zeros((2, 6), float)
        self.observation_matrix[0][0] = 1
        self.observation_matrix[1][3] = 1
        self.observation_noise = 25 * numpy.identity(2, float)
        self.sigma_x = 70
        self.sigma_y = 100
        self.rho = 0.3
        self.worldsize = int(self.constants['worldsize'])
        self.grid_color_normalizer = 1
        self.update_estimates()
        self.update_estimate_plot()

        # Move
        mytanks = self.bzrc.get_mytanks()
        tank = mytanks[0]
        if tank.status != 'alive':
        	self.do_move(tank)
        	results = self.bzrc.do_commands(self.commands)

    def update_estimates(self):
    	F = self.physics
    	Et = self.sigma_t
    	Ex = self.state_noise
    	H = self.observation_matrix
    	Ez = self.observation_noise
    	ut = self.mu_t
    	FT = F.transpose()
    	HT = H.transpose()

    	second_term_inverse = numpy.linalg.inv(H.dot(F.dot(Et.dot(FT)) + Ex).dot(HT) + Ez)
    	kalman_gain = (F.dot(Et).dot(FT) + Ex).dot(HT).dot(second_term_inverse)
        zt = (self.enemies[0].x, self.enemies[0].y)
    	self.mu_t = F.dot(ut) + kalman_gain.dot(zt -H.dot(F.dot(ut)))
    	self.sigma_t = (numpy.identity(6, float) - kalman_gain.dot(H)).dot(F.dot(Et.dot(FT)) + Ex)

    	for item in self.mu_t:
    		print item
    	for item in self.sigma_t:
    		print item

    def update_estimate_plot(self):
    	maxvalue = 0
    	self.sigma_x = math.sqrt(self.sigma_t[0][0])**2
    	self.sigma_y = math.sqrt(self.sigma_t[1][1])**2
    	self.rho = self.sigma_t[0][1] / (self.sigma_x * self.sigma_y)

    	for x in range(0, self.worldsize - 1):
    		for y in range(0, self.worldsize - 1):
    			grid[y][x] = 1.0/(2.0 * math.pi * self.sigma_x * self.sigma_y * math.sqrt(1 - self.rho **2)) \
    				* math.exp(-1.0/2.0 * ((x-self.worldsize/2)**2 / self.sigma_x**2 + (y-self.worldsize/2)**2 / self.sigma_y**2 \
    				-2.0*self.rho*(x-self.worldsize/2)*(y-self.worldsize/2)/(self.sigma_x*self.sigma_y)))
    			if grid[y][x] > 1:
    				print grid[y][x]
    			if grid[y][x] > maxvalue:
    				maxvalue = grid[y][x]
    	self.grid_color_normalizer = maxvalue
    	for x in range(0, self.worldsize - 1):
    		for y in range(0, self.worldsize - 1):
    			grid[x][y] = grid[x][y] / self.grid_color_normalizer
    			if x % 100 == 0:
    				grid[x][y] = (1 - float (abs((x - self.worldsize / 2))) / float(self.worldsize / 2))**2
    			if y % 100 == 0:
    				grid[x][y] = (1 - float (abs((y - self.worldsize / 2))) / float(self.worldsize / 2))**2
    			if y % 100 == 0 and x % 100 == 0:
    				grid[x][y] = 1
    			if x == self.worldsize / 2 and y == self.worldsize / 2:
    				grid[x][y] = 0
    	grid[int(self.mu_t[1]) + self.worldsize / 2][int(self.mu_t[0]) + self.worldsize / 2] = 0

    def do_move(self, tank):
        """Compute and follow the potential field vector"""
        print(self.get_potential_field_vector(tank))
        v, theta = self.get_potential_field_vector(tank)
        self.commands.append(self.pd_controller_move(tank, v, theta))

    def move_to_position(self, tank, target_x, target_y):
        """Set command to move to given coordinates."""
        target_angle = math.atan2(target_y - tank.y,
                                  target_x - tank.x)
        relative_angle = self.normalize_angle(target_angle - tank.angle)
        command = Command(tank.index, 1, 2 * relative_angle, True)
        self.commands.append(command)

    def normalize_angle(self, angle):
        """Make any angle be between +/- pi."""
        angle -= 2 * math.pi * int (angle / (2 * math.pi))
        if angle <= -math.pi:
            angle += 2 * math.pi
        elif angle > math.pi:
            angle -= 2 * math.pi
        return angle

    def draw_grid(self):
        # This assumes you are using a numpy array for your grid
        width, height = grid.shape
        glRasterPos2f(-1, -1)
        glDrawPixels(width, height, GL_LUMINANCE, GL_FLOAT, grid)
        glFlush()
        glutSwapBuffers()
        glutPostRedisplay()

    def update_grid(self, new_grid):
        global grid
        grid = new_grid

    def init_window(self, width, height):
        global window
        global grid
        grid = zeros((width, height))
        glutInit(())
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("Grid filter")
        glutDisplayFunc(self.draw_grid)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

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
    agent.init_window(int(800),int(800))
    # Run the agent
    try:
        while True:
            time_diff = time.time() - prev_time
            agent.tick(time_diff)
            glutMainLoopEvent()
    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()


if __name__ == '__main__':
    main()