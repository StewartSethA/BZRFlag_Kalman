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
        self.worldsize = int(self.constants['worldsize'])
        self.grid_color_normalizer = 1
        self.last_time_diff = 0

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
        
        dt = time_diff - self.last_time_diff
        self.last_time_diff = time_diff

        self.physics[0][1] = dt
        self.physics[0][2] = dt**2 / 2
        self.physics[1][2] = dt
        self.physics[3][4] = dt
        self.physics[3][5] = dt**2 / 2
        self.physics[4][5] = dt

        self.update_estimates()
        self.update_estimate_plot()

        # Move
        mytanks = self.bzrc.get_mytanks()
        tank = mytanks[0]
        if tank.status != 'alive':
        	self.do_move(tank)
        	results = self.bzrc.do_commands(self.commands)

    def update_estimates(self):
        # Initialize equation variables
    	F = self.physics
    	Et = self.sigma_t
    	Ex = self.state_noise
    	H = self.observation_matrix
    	Ez = self.observation_noise
    	ut = self.mu_t
    	FT = F.transpose()
    	HT = H.transpose()

        # Kalman update equations
    	second_term_inverse = numpy.linalg.inv(H.dot(F.dot(Et.dot(FT)) + Ex).dot(HT) + Ez)
    	kalman_gain = (F.dot(Et).dot(FT) + Ex).dot(HT).dot(second_term_inverse)
        self.zt = (self.enemies[0].x, self.enemies[0].y)
    	self.mu_t = F.dot(ut) + kalman_gain.dot(self.zt -H.dot(F.dot(ut)))
    	self.sigma_t = (numpy.identity(6, float) - kalman_gain.dot(H)).dot(F.dot(Et.dot(FT)) + Ex)

        # Debug statements
        print "mu_t:\n"
    	for item in self.mu_t:
    		print item
        print "observation:\n"
        print self.zt[0], self.zt[1]
        print "sigma:\n"
    	for item in self.sigma_t:
    		print item

    def update_estimate_plot(self):
        # sigma x, y, and rho are used to plot the PDF
    	self.sigma_x = self.sigma_t[0][0]
    	self.sigma_y = self.sigma_t[3][3]
    	self.rho = self.sigma_t[0][3] / (math.sqrt(self.sigma_x) * math.sqrt(self.sigma_y))
        print self.sigma_x, self.sigma_y, self.rho

        # Create a mini 2D sigma matrix and a 1D mu vector for plotting the distribution
        pos_mu = (self.mu_t[0] + self.worldsize/2, self.mu_t[3] + self.worldsize/2)
        pos_sigma = ((self.sigma_t[0][0], self.sigma_t[0][3]),(self.sigma_t[3][0], self.sigma_t[3][3]))
        E_inv = numpy.linalg.inv(pos_sigma)
        rank = 2
        coefficient = (1.0 / math.sqrt(numpy.linalg.det(pos_sigma) * (2.0 * math.pi)**rank))

        # Clear display grid
        for x in range(0, self.worldsize - 1):
            for y in range(0, self.worldsize - 1):
                grid[x][y] = 0

        step = 1
        extent = 2
        maxvalue = 0
        for x in range(int(self.mu_t[0] - extent * self.sigma_t[0][0] + self.worldsize/2), int(self.mu_t[0] + extent * self.sigma_t[0][0] + self.worldsize/2), step):
            if x < 0 or x >= self.worldsize:
                continue
            for y in range(int(self.mu_t[3] - extent * self.sigma_t[3][3] + self.worldsize/2), int(self.mu_t[3] + extent * self.sigma_t[3][3] + self.worldsize/2), step):
                if y < 0 or y >= self.worldsize:
                    continue

                gx = x #/ step
                gy = y #/ step
                '''
                x_center = x - self.mu_t[0] - self.worldsize/2
                y_center = y - self.mu_t[3] - self.worldsize/2
                small1 = 1.0/(2.0 * math.pi * self.sigma_x * self.sigma_y * math.sqrt(1 - self.rho **2))
                small2 = math.exp(-1.0/2.0 * (x_center**2 / self.sigma_x**2 + y_center**2 / self.sigma_y**2 \
                    -2.0*self.rho*x_center*y_center/(self.sigma_x*self.sigma_y)))


                # Plot the Gaussian distribution
                grid[gy][gx] = 1.0/(2.0 * math.pi * self.sigma_x * self.sigma_y * math.sqrt(1 - self.rho **2)) \
    				* math.exp(-1.0/2.0 * (x_center**2 / self.sigma_x**2 + y_center**2 / self.sigma_y**2 \
    				-2.0*self.rho*x_center*y_center/(self.sigma_x*self.sigma_y)))
                '''

                # Plotting Try 2: See if we can dispense with the Rho
                plot_x = (gx, gy)
                plot_vector = numpy.subtract(plot_x, pos_mu)
                #print "plot vector ", plot_vector, "\n"
                exponent = (-1.0/2.0) * (plot_vector.transpose().dot(E_inv.dot(plot_vector)))
                #print coefficient, " c\n"
                #print exponent, " e\n"
                grid[gy][gx] = coefficient * math.exp(exponent)
                #print grid[gy][gx], " g\n"

                # Set the normalizing constant
                if grid[gy][gx] > maxvalue:
                    maxvalue = grid[gy][gx]

                # DEBUG
                if grid[gy][gx] > 1:
                    print grid[gy][gx]

        # Apply normalizing constant
        for x in range(int(self.mu_t[0] - extent * self.sigma_t[0][0] + self.worldsize/2), int(self.mu_t[0] + extent * self.sigma_t[0][0] + self.worldsize/2), step):
            if x < 0 or x >= self.worldsize:
                continue
            for y in range(int(self.mu_t[3] - extent * self.sigma_t[3][3] + self.worldsize/2), int(self.mu_t[3] + extent * self.sigma_t[3][3] + self.worldsize/2), step):
                if y < 0 or y >= self.worldsize:
                    continue
                gx = x #/ step
                gy = y #/ step
                grid[gy][gx] = (grid[gy][gx] / maxvalue) + 0

                # Gridline drawing code: Uncomment to see gridlines drawn
                '''
    			if x % 100 == 0:
    				grid[x][y] = (1 - float (abs((x - self.worldsize / 2))) / float(self.worldsize / 2))**2
    			if y % 100 == 0:
    				grid[x][y] = (1 - float (abs((y - self.worldsize / 2))) / float(self.worldsize / 2))**2
    			if y % 100 == 0 and x % 100 == 0:
    				grid[x][y] = 1
    			if x == self.worldsize / 2 and y == self.worldsize / 2:
    				grid[x][y] = 0
                '''

        # Put a white dot at the observation site
        if self.zt[1] >= 0 and self.zt[1] < self.worldsize:
            if self.zt[0] >= 0 and self.zt[0] < self.worldsize:
                grid[int(self.zt[1]) - self.worldsize/2][int(self.zt[0]) - self.worldsize/2] = 1
    	#grid[int(self.mu_t[1]) + self.worldsize / 2][int(self.mu_t[0]) + self.worldsize / 2] = 0
        

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