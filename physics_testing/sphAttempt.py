import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import gamma
import taichi as ti
ti.init(arch=ti.cpu)

#TODO if beyond boundary, set it's position back to the boundary and then inverse it's velocity 
# + multiple vecolity by a component perp to the wall with a damping factor

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""

def W( x, y, z, h ):
	"""
	Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w
	
	
def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
	
	
def getPairwiseSeparations( ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	M = ri.shape[0]
	N = rj.shape[0]
	
	# positions ri = (x,y,z)
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))
	riz = ri[:,2].reshape((M,1))
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1))
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T
	
	return dx, dy, dz
	

def getDensity( r, pos, m, h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of accelerations
	"""
	
	M = r.shape[0]
	
	dx, dy, dz = getPairwiseSeparations( r, pos )
	
	rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1))
	
	return rho
	
	
def getPressure(rho, k, n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	P = k * rho**(1+1/n)
	
	return P
	

def getAcc( pos, vel, m, h, k, n, lmbda, nu ):
	"""
	Calculate the acceleration on each SPH particle
	pos   is an N x 3 matrix of positions
	vel   is an N x 3 matrix of velocities
	m     is the particle mass
	h     is the smoothing length
	k     equation of state constant
	n     polytropic index
	lmbda external force constant
	nu    viscosity
	a     is N x 3 matrix of accelerations
	"""
	
	N = pos.shape[0]
	
	# Calculate densities at the position of the particles
	rho = getDensity( pos, pos, m, h )
	
	# Get the pressures
	P = getPressure(rho, k, n)
	
	# Get pairwise distances and gradients
	dx, dy,dz = getPairwiseSeparations( pos, pos )
	dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))
	ay = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))
	az = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))
	
	# Add external potential force
	a -= lmbda * pos
	
	# Add viscosity
	a -= nu * vel
	
	return a
	


def main():
	""" N-body simulation """
	
	# Simulation parameters
	N         = 1000   # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 100     # time at which simulation ends
	dt        = 0.04   # timestep
	h         = 0.1    # smoothing length
	k         = 0.1    # equation of state constant
	n         = 2      # polytropic index
	nu        = .3      # damping
	m     = 1                    # single particle mas
	g = 9.81			# gravity
	lmbda = np.array([[0.0, g, 0.0]]) # external force constant

	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	np.random.seed(42)            # set the random number generator seed
	
	pos = np.zeros(shape=(N, 3))         # particle positions
	# pos_field = ti.field(float, shape=(N, 2))
	pos_field = ti.Vector.field(3, float, shape=(N,))
	for i in range(0, N):
		posi = np.array([[np.random.uniform(1, 3), np.random.uniform(1, 3), np.random.uniform(1,3) ]])
		pos[i] = posi
	vel   = np.zeros(pos.shape)
	
	# calculate initial gravitational accelerations
	acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# prep figure
	# fig = plt.figure(figsize=(4,5), dpi=80)
	# grid = plt.GridSpec(6, 1, wspace=0.0, hspace=0.3)
	# ax1 = plt.subplot(grid[0:5,0])
	# ax2 = plt.subplot(grid[5,0])
	rr = np.zeros((100,3))
	rlin = np.linspace(0,1,100)
	rr[:,0] =rlin

	window = ti.ui.Window('N-Body', (1280, 720), vsync=True)
	canvas = window.get_canvas()
	camera = ti.ui.Camera()
	scene = ti.ui.Scene()
	
	# Simulation Main Loop
	while window.running:
		# (1/2) kick
		vel += acc * dt/2
		
		# drift
		pos += vel * dt
		
		# update accelerations
		acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		vel += acc * dt/2
		
		# update time
		t += dt
		
		# get density for plottiny
		rho = getDensity( pos, pos, m, h )

		out_of_bottom_boundary = pos[:, 1] < 0
		vel[out_of_bottom_boundary, 1] *= -0.9
		pos[out_of_bottom_boundary, 1] = 0

		out_of_top_boundary = pos[:, 1] > 5
		vel[out_of_top_boundary, 1] *= -0.9
		pos[out_of_top_boundary, 1] = 5

		out_of_left_boundary = pos[:, 0] < 0
		vel[out_of_left_boundary, 0] *= -0.9
		pos[out_of_left_boundary, 0] = 0

		out_of_right_boundary = pos[:, 0] > 5
		vel[out_of_right_boundary, 0] *= -0.9
		pos[out_of_right_boundary, 0] = 5


		pos_field.from_numpy(pos)
		camera.track_user_inputs(window)
		scene.ambient_light((1, 1, 1))
		scene.particles(pos_field, radius=0.05, color=(1, 0, 0))
		scene.set_camera(camera)
		canvas.scene(scene)
		window.show()

		
		# rendering
		# plot in real time
		# if plotRealTime:
		# 	plt.sca(ax1)
		# 	plt.cla()
		# 	cval = np.minimum((rho-3)/3,1).flatten()
		# 	plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
		# 	ax1.set(xlim=(0, 5), ylim=(0, 5))
		# 	ax1.set_aspect('equal', 'box')
		# 	ax1.set_facecolor('black')
		# 	ax1.set_facecolor((.1,.1,.1))
			
		# 	plt.sca(ax2)
		# 	plt.cla()
		# 	ax2.set(xlim=(0, 1), ylim=(0, 3))
		# 	ax2.set_aspect(0.1)
		# 	rho_radial = getDensity( rr, pos, m, h )
		# 	plt.pause(0.001)
	


  
if __name__== "__main__":
  main()
