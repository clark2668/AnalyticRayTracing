import numpy as np
from scipy import integrate, optimize
#from scipy.optimize import fsolve, minimize, basinhopping, root
import scipy.constants

# define model parameters (SPICE 2015/southpole)
n_ice = 1.78
b = 2 * n_ice
z_0 = 71.
delta_n = 0.427
speed_of_light = scipy.constants.c

def n(z):
	"""
	refractive index as a function of depth
	"""
	res = n_ice - delta_n * np.exp(z / z_0)
	return res
	
	
def get_gamma(z):
	return delta_n * np.exp(z / z_0)
	
def get_turning_point(c):
	"""
	calculate the turning point, i.e. the maximum of the ray tracing path
	"""
	gamma2 = b * 0.5 - (0.25 * b ** 2 - c) ** 0.5  # first solution discarded
	z2 = np.log(gamma2 / delta_n) * z_0
	return gamma2, z2
	
def get_y(gamma, C_0, C_1):
	"""
	analytic form of the ray tracing part given an exponential index of refraction profile
	Parameters
	-------
	gamma: (float or array)
		gamma is a function of the depth z
	C_0: (float)
		first parameter
	C_1: (float)
		second parameter
	"""
	c = n_ice ** 2 - C_0 ** -2
	root = np.abs(gamma ** 2 - gamma * b + c)  # we take the absolute number here but we only evaluate the equation for positive outcome. This is to prevent rounding errors making the root negative
	logargument = gamma / (2 * c ** 0.5 * (root) ** 0.5 - b * gamma + 2 * c)
	result = z_0 * (n_ice ** 2 * C_0 ** 2 - 1) ** -0.5 * np.log(logargument) + C_1
	return result
	
	
def get_y_with_z_mirror(z, C_0, C_1=0):
	"""
	analytic form of the ray tracing part given an exponential index of refraction profile
	this function automatically mirrors z values that are above the turning point,
	so that this function is defined for all z
	Parameters
	-------
	z: (float or array)
		depth z
	C_0: (float)
		first parameter
	C_1: (float)
		second parameter
	"""
	c = n_ice ** 2 - C_0 ** -2
	gamma_turn, z_turn = get_turning_point(c)
	if(z_turn >= 0):
		# signal reflected at surface
		z_turn = 0
		gamma_turn = get_gamma(0)
	y_turn = get_y(gamma_turn, C_0, C_1)
	if(type(z) == float or (type(z) == int) or (type(z) == np.float64)):
		if(z < z_turn):
			gamma = get_gamma(z)
			return get_y(gamma, C_0, C_1)
		else:
			gamma = get_gamma(2 * z_turn - z)
			return 2 * y_turn - get_y(gamma, C_0, C_1)
	else:
		mask = z < z_turn
		res = np.zeros_like(z)
		zs = np.zeros_like(z)
		gamma = get_gamma(z[mask])
		zs[mask] = z[mask]
		res[mask] = get_y(gamma, C_0, C_1)
		gamma = get_gamma(2 * z_turn - z[~mask])
		res[~mask] = 2 * y_turn - get_y(gamma, C_0, C_1)
		zs[~mask] = 2 * z_turn - z[~mask]
	return res, zs

def get_C_1(x1, C_0):
	"""
	calculates constant C_1 for a given C_0 and start point x1
	"""
	return x1[0] - get_y_with_z_mirror(x1[1], C_0)
	
def get_c(C_0):
	return n_ice ** 2 - C_0 ** -2
	
	
def get_C0_from_log(logC0):
	"""
	transforms the fit parameter C_0 so that the likelihood looks better
	"""
	return np.exp(logC0) + 1. / n_ice

def get_z_unmirrored(z, C_0):
	"""
	calculates the unmirrored z position
	"""
	c = n_ice ** 2 - C_0 ** -2
	gamma_turn, z_turn = get_turning_point(c)
	if(z_turn >= 0):
		# signal reflected at surface
		z_turn = 0

	z_unmirrored = z
	if(z > z_turn):
		z_unmirrored = 2 * z_turn - z
	return z_unmirrored

def get_y_diff(z_raw, C_0):
	"""
	derivative dy(z)/dz
	"""
	z = get_z_unmirrored(z_raw, C_0)
	c = n_ice ** 2 - C_0 ** -2
	res = (-np.sqrt(c) * np.exp(z / z_0) * b * delta_n + 0.2e1 * np.sqrt(-b * delta_n * np.exp(z / z_0) + delta_n ** 2 * np.exp(0.2e1 * z / z_0) + c) * c + 0.2e1 * c ** (0.3e1 / 0.2e1)) / (0.2e1 * np.sqrt(c) * np.sqrt(-b * delta_n * np.exp(z / z_0) + delta_n ** 2 * np.exp(0.2e1 * z / z_0) + c) - b * delta_n * np.exp(z / z_0) + 0.2e1 * c) * (-b * delta_n * np.exp(z / z_0) + delta_n ** 2 * np.exp(0.2e1 * z / z_0) + c) ** (-0.1e1 / 0.2e1) * ((n_ice ** 2 * C_0 ** 2 - 1) ** (-0.1e1 / 0.2e1))

	if(z != z_raw):
		res *= -1
	return res

def ds(t, C_0):
	"""
	helper to calculate line integral
	"""
	return (get_y_diff(t, C_0) ** 2 + 1) ** 0.5
	
	
def get_z_mirrored(x1, x2, C_0):
	"""
	calculates the mirrored x2 position so that y(z) can be used as a continuous function
	"""
	c = n_ice ** 2 - C_0 ** -2
	C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0)
	gamma_turn, z_turn = get_turning_point(c)
	if(z_turn >= 0):
		# signal reflected at surface
		z_turn = 0
		gamma_turn = get_gamma(0)
	y_turn = get_y(gamma_turn, C_0, C_1)
	zstart = x1[1]
	zstop = x2[1]
	if(y_turn < x2[0]):
		zstop = zstart + np.abs(z_turn - x1[1]) + np.abs(z_turn - x2[1])
	x2_mirrored = [x2[0], zstop]
	return x2_mirrored
	

def get_path_length(x1, x2, C_0):
	x2_mirrored = get_z_mirrored(x1, x2, C_0)
	path_length = integrate.quad(ds, x1[1], x2_mirrored[1], args=(C_0))
	return path_length
	
def get_travel_time(x1, x2, C_0):
	x2_mirrored = get_z_mirrored(x1, x2, C_0)

	def dt(t, C_0):
		z = get_z_unmirrored(t, C_0)
		return ds(t, C_0) / speed_of_light * n(z)

	travel_time = integrate.quad(dt, x1[1], x2_mirrored[1], args=(C_0))
	return travel_time
	
def get_temperature(z):
	return (-51.5 + z * (-4.5319e-3 + 5.822e-6 * z))
	
	
def get_attenuation_length(z, frequency):
	t = get_temperature(z)
	f0 = 0.0001
	f2 = 3.16
	w0 = np.log(f0)
	w1 = 0.0
	w2 = np.log(f2)
	w = np.log(frequency)
	b0 = -6.74890 + t * (0.026709 - t * 0.000884)
	b1 = -6.22121 - t * (0.070927 + t * 0.001773)
	b2 = -4.09468 - t * (0.002213 + t * 0.000332)
	if((type(frequency) == float) or (type(frequency) == np.float64)):
		if (frequency < 1.):
			a = (b1 * w0 - b0 * w1) / (w0 - w1)
			bb = (b1 - b0) / (w1 - w0)
		else:
			a = (b2 * w1 - b1 * w2) / (w1 - w2)
			bb = (b2 - b1) / (w2 - w1)
	else:
		a = np.ones_like(frequency) * (b2 * w1 - b1 * w2) / (w1 - w2)
		bb = np.ones_like(frequency) * (b2 - b1) / (w2 - w1)
		a[frequency < 1.] = (b1 * w0 - b0 * w1) / (w0 - w1)
		bb[frequency < 1.] = (b1 - b0) / (w1 - w0)

	return 1. / np.exp(a + bb * w)
	
def get_attenuation_along_path(x1, x2, C_0, frequency):
	x2_mirrored = get_z_mirrored(x1, x2, C_0)

	def dt(t, C_0, frequency):
		z = get_z_unmirrored(t, C_0)
		return ds(t, C_0) / get_attenuation_length(z, frequency)

	tmp = np.array([integrate.quad(dt, x1[1], x2_mirrored[1], args=(C_0, f))[0] for f in frequency])
	attenuation = np.exp(-1 * tmp)
	return attenuation

def get_angle(x, x_start, C_0):
	z = get_z_mirrored(x_start, x, C_0)[1]
	dy = get_y_diff(z, C_0)
	angle = np.arctan(dy)
	if(angle < 0):
		angle = np.pi + angle
	return angle

def get_launch_angle(x1, C_0):
	return get_angle(x1, x1, C_0)

def get_receive_angle(x1, x2, C_0):
	return np.pi - get_angle(x2, x1, C_0)

def get_delta_y(C_0, x1, x2, C0range=[1. / n_ice, np.inf]):
	"""
	calculates the difference in the y position between the analytic ray tracing path
	specified by C_0 at the position x2
	"""
	if(not(type(C_0) == np.float64 or type(C_0) == float)):
		C_0 = C_0[0]
	if((C_0 < C0range[0]) or(C_0 > C0range[1])):
		return np.inf
	c = n_ice ** 2 - C_0 ** -2
	# determine y translation first
	C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0)

	# for a given c_0, 3 cases are possible to reach the y position of x2
	# 1) direct ray, i.e., beofre the turning point
	# 2) refracted ray, i.e. after the turning point but not touching the surface
	# 3) reflected ray, i.e. after the ray reaches the surface
	gamma_turn, z_turn = get_turning_point(c)
	if(z_turn > 0):
		z_turn = 0  # a reflection is just a turning point at z = 0, i.e. cases 2) and 3) are the same
		gamma_turn = get_gamma(z_turn)
	y_turn = get_y(gamma_turn, C_0, C_1)
	if(z_turn < x2[1]):  # turning points is deeper that x2 positions, can't reach target
		return -np.inf
	if(y_turn > x2[0]):  # we always propagate from left to right
		# direct ray
		y2_fit = get_y(get_gamma(x2[1]), C_0, C_1)  # calculate y position at get_path position
		diff = (x2[0] - y2_fit)
		return diff
	else:
		# now it's a bit more complicated. we need to transform the coordinates to be on the mirrored part of the function
		z_mirrored = x2[1]
		gamma = get_gamma(z_mirrored)
		y2_raw = get_y(gamma, C_0, C_1)
		y2_fit = 2 * y_turn - y2_raw
		diff = (x2[0] - y2_fit)
		return -1 * diff
	
def obj_delta_y_square(logC_0, x1, x2):
	"""
	objective function to find solution for C0
	"""
	C_0 = get_C0_from_log(logC_0)
	return get_delta_y(C_0, x1, x2) ** 2

def obj_delta_y(logC_0, x1, x2):
	"""
	function to find solution for C0, returns distance in y between function and x2 position
	result is signed! (important to use a root finder)
	"""
	C_0 = get_C0_from_log(logC_0)
	return get_delta_y(C_0, x1, x2)

def determine_solution_type(x1, x2, C_0):
	c = n_ice ** 2 - C_0 ** -2
	C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0)
	gamma_turn, z_turn = get_turning_point(c)

	if(z_turn >= 0):
		z_turn = 0
		gamma_turn = get_gamma(0)
	y_turn = get_y(gamma_turn, C_0, C_1)
	if(x2[0] < y_turn):
		return 'direct'
	else:
		if(z_turn == 0):
			return 'reflected'
		else:
			return 'refracted'
		
def find_solutions(x1, x2):
	"""
	this function finds all ray tracing solutions
	prerequesite is that x2 is above and to the right of x1, this is not a violation of universality
	because this requirement can be achieved with a simple coordinate transformation
	returns an array of the C_0 paramters of the solutions (the array might be empty)
	"""
	tol = 1e-4
	results = []
	C0s = []  # intermediate storage of results
	print ('starting optimization with x0 = {:.2f} -> C0 = {:.3f}'.format(-1, get_C0_from_log(-1)))
	result = optimize.root(obj_delta_y_square, x0=-1, args=(x1, x2), tol=tol)
	if(result.fun < 1e-5):
		if(np.round(result.x[0], 3) not in np.round(C0s, 3)):
			C_0 = get_C0_from_log(result.x[0])
			C0s.append(C_0)
			solution_type = determine_solution_type(x1, x2, C_0)
			print ("found {} solution C0 = {:.2f}, logC0 = {:.8f}".format(solution_type, C_0,result.x[0]))
			results.append({'type': solution_type, 'C0': C_0, 'C1': get_C_1(x1, C_0)})

	# check if another solution with higher logC0 exists
	logC0_start = result.x[0] + 0.0001
	logC0_stop = 100
	delta_start = obj_delta_y(logC0_start, x1, x2)
	delta_stop = obj_delta_y(logC0_stop, x1, x2)
	#print "Delta start ",delta_start
	#print "Delta stop ",delta_stop
	if(np.sign(delta_start) != np.sign(delta_stop)):
		print("solution with logC0 > {:.3f} exists".format(result.x[0]))
		result2 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(x1, x2))
		if(np.round(result2, 3) not in np.round(C0s, 3)):
			C_0 = get_C0_from_log(result2)
			C0s.append(C_0)
			solution_type = determine_solution_type(x1, x2, C_0)
			print("found {} solution C0 = {:.2f}, logC0 = {:.8f}".format(solution_type, C_0,result2))
			results.append({'type': solution_type, 'C0': C_0, 'C1': get_C_1(x1, C_0)})
	else:
		print("no solution with logC0 > {:.3f} exists".format(result.x[0]))

	logC0_start = -100
	logC0_stop = result.x[0] - 0.0001
	delta_start = obj_delta_y(logC0_start, x1, x2)
	delta_stop = obj_delta_y(logC0_stop, x1, x2)
	if(np.sign(delta_start) != np.sign(delta_stop)):
		print("solution with logC0 < {:.3f} exists".format(result.x[0]))
		result3 = optimize.brentq(obj_delta_y, logC0_start, logC0_stop, args=(x1, x2))
		if(np.round(result3, 3) not in np.round(C0s, 3)):
			C_0 = get_C0_from_log(result3)
			C0s.append(C_0)
			solution_type = determine_solution_type(x1, x2, C_0)
			print ("found {} solution C0 = {:.2f}, logC0 = {:.8f}".format(solution_type, C_0,result.x[0]))
			results.append({'type': solution_type, 'C0': C_0, 'C1': get_C_1(x1, C_0)})
	else:
		print ("no solution with logC0 < {:.3f} exists".format(result.x[0]))
	return results
	
def get_path(x1, x2, C_0, n_points=1000):
	"""
	for plotting purposes only,  returns the ray tracing path between x1 and x2
	the result is only valid if C_0 is a solution to the ray tracing problem
	Parameters
	-------
	x1: array
		start position (y, z)
	x2: array
		stop position (y, z)
	C_0: (float)
		first parameter
	n_points: integer (optional)
		the number of coordinates to calculate
	Returns
	-------
	yy: array
		the y coordinates of the ray tracing path
	zz: array
		the z coordinates of the ray tracing path
	"""
	c = n_ice ** 2 - C_0 ** -2
	C_1 = x1[0] - get_y_with_z_mirror(x1[1], C_0)
	gamma_turn, z_turn = get_turning_point(c)
	if(z_turn >= 0):
		# signal reflected at surface
		#print('signal reflects off surface')
		z_turn = 0
		gamma_turn = get_gamma(0)
	y_turn = get_y(gamma_turn, C_0, C_1)
	zstart = x1[1]
	zstop = get_z_mirrored(x1, x2, C_0)[1]
	z = np.linspace(zstart, zstop, n_points)
	mask = z < z_turn
	res = np.zeros_like(z)
	zs = np.zeros_like(z)
	gamma = get_gamma(z[mask])
	zs[mask] = z[mask]
	res[mask] = get_y(gamma, C_0, C_1)
	gamma = get_gamma(2 * z_turn - z[~mask])
	res[~mask] = 2 * y_turn - get_y(gamma, C_0, C_1)
	zs[~mask] = 2 * z_turn - z[~mask]

	#print('turning points for C_0 = {:.2f}, b= {:.2f}, gamma = {:.4f}, z = {:.1f}, y_turn = {:.0f}'.format(C_0, b, gamma_turn, z_turn, y_turn))
	return res, zs