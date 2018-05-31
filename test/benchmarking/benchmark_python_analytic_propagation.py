import raytrace_tools as ray
import numpy as np
import time


def main():
	
	x1 = [478.,-149.]
	x2 = [635., -5.] #direct/reflected
	x3 = [1000., -90.] #refracted/reflected
	x4 = [700., -149.] #refracted/reflected
	x5 = [1000., -5.] #no solution
	
	x = x4
	
	t = time.time()
	results = ray.find_solutions(x1,x)
	ray.get_path(x1, x, results[0]['C0'])
	ray.get_path(x1, x, results[1]['C0'])
	
	dt = time.time()-t
	
	print"\n\n"

	print "Time used ", dt
	
main()