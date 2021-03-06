//basically a c++ recasting of the python code written by c glaser

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>
#include <stdio.h>

//for gsl numerical integration
#include <gsl/gsl_integration.h>

//for gsl root finding
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>


using namespace std;

double n_ice = 1.78;
double b = 2.*n_ice;
double z_0 = 71.; //meters
double delta_n = 0.427;
double speed_of_light = 299792458; //meters/second
double pi = atan(1.)*4.; //compute and store pi
//double inf = std::numeric_limits<double>::infinity();
//double inf = std::numeric_limits<double>::max()-2;
double inf = 1e130; //infinity for all practical purposes...

double index_vs_depth(double z){
	//return the index of refraction at a given depth
	double index = n_ice - (delta_n*exp(z/z_0));
	return index;
}

double get_gamma(double z){
	return delta_n * exp(z/z_0);
}

void get_turning_point(double c, double &gamma2, double &z2){
	//calculate the turning point (the maximum of the ray tracing path)
	gamma2 = b*0.5 - sqrt(0.25 * pow(b,2.) - c);
	z2 = log(gamma2/delta_n) * z_0;
}

double get_y(double gamma, double C0, double C1){
	//parameters
	//gamma: gamma is a function of the depth z
	//c0: first parameter
	//c1: second paramter
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double root = abs( pow(gamma,2.) - gamma*b + c);
	double logargument = gamma / ( 2.*sqrt(c) * sqrt(root) - b*gamma + 2.*c);
	double result = z_0 * 1./sqrt((pow(n_ice,2.) * pow(C0,2.) - 1)) * log(logargument) + C1;
	return result;
}

double get_y_with_z_mirror(double z, double C0, double C1=0.){
	//parameters
	//z: arrays of depths
	//c0: first parameter
	//c1: second parameter
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn);
	if(z_turn >=0.){ //signal is reflected at surface
		z_turn=0.; //we've hit the surface
		gamma_turn=get_gamma(0.); //get gamma at the surface
	}
	double y_turn = get_y(gamma_turn,C0,C1);
	double result=0.;
	if(z < z_turn){
		double gamma = get_gamma(z);
		result=get_y(gamma,C0,C1);
	}
	else{
		double gamma = get_gamma(2*z_turn - z);
		result = 2*y_turn - get_y(gamma,C0,C1);
	}
	return result;
}

double get_C1(double pos[2], double C0){
	//calculates C1 for a given C0 and starting point X1
	return pos[0] - get_y_with_z_mirror(pos[1],C0);
}

double get_c(double C0){
	return pow(n_ice,2.)-pow(C0,-2.);
}

double get_C0_from_log(double logC0){
	//transforms fit parameter C0 so that the likelihood looks better
	return exp(logC0) + 1./n_ice;
}

double get_z_unmirrored(double z, double C0){
	//calculates the unmirrored z position
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn);
	if(z_turn >=0.){ //signal is reflected at surface
		z_turn=0.; //we've hit the surface
		gamma_turn=get_gamma(0.); //get gamma at the surface
	}
	double z_unmirrored = z;
	if(z > z_turn) z_unmirrored = 2*z_turn - z;
	return z_unmirrored;
}

double get_y_diff(double z_raw, double C0){
	//derivative dy(z)/dz
	double z = get_z_unmirrored(z_raw, C0);
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double term1 = 
			-sqrt(c) * exp(z/z_0) * b * delta_n
			+ 2.*sqrt(-b * delta_n * exp(z/z_0) + pow(delta_n,2.)*exp(2.*z/z_0) + c) * c
			+ 2.*pow(c,3./2.);
	double term2 =
			2. * sqrt(c) * sqrt(-b * delta_n * exp(z/z_0) + pow(delta_n,2.) * exp(2.*z/z_0) + c)
			-b*delta_n*exp(z/z_0)+
			+ 2.*c;
	double term3 =
			-b * delta_n * exp(z/z_0)
			+ pow(delta_n,2.) * exp(2.*z/z_0)
			+ c;
	double term4 = pow(n_ice,2.)*pow(C0,2.)-1;
	
	double res = term1 / term2 * 1./sqrt(term3) * 1./sqrt(term4);
	if(z != z_raw) res*=-1.;
	return res;
}

void get_z_mirrored(double pos[2], double pos2[2], double C0, double (&x2_mirrored)[2]){
	//calculates the mirrored x2 position so that y(z) can be used as a continuous function
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double C1 = pos[0] - get_y_with_z_mirror(pos[1],C0);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn);
	if(z_turn >=0.){ //signal is reflected at surface
		z_turn=0.; //we've hit the surface
		gamma_turn=get_gamma(0.); //get gamma at the surface
	}
	double y_turn = get_y(gamma_turn,C0,C1);
	double z_start = pos[1];
	double z_stop = pos2[1];
	if(y_turn <pos2[0]){
			z_stop = z_start + abs(z_turn - pos[1]) + abs(z_turn - pos2[1]);
	}
	x2_mirrored[0] = pos2[0];
	x2_mirrored[1] = z_stop;
}

//this function is explicity prepared for gsl integration in get_path_length
double ds (double t, void *params){
	//helper to calculate line integral
	double C0 = *(double *) params;
	return sqrt((pow(get_y_diff(t,C0),2.)+1));
}

double get_path_length(double pos[2], double pos2[2], double C0){
	double x2_mirrored[2]={0.};
	get_z_mirrored(pos,pos2,C0,x2_mirrored);
	
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
	gsl_function F;
	F.function = &ds;
	F.params=&C0;
	
	double result, error;
	
	gsl_integration_qags(&F, pos[1], x2_mirrored[1],0,1e-7,1000,w,&result,&error);
	gsl_integration_workspace_free(w);
	return result;
}

//this function is explicitly prepared for gsl integration in get_travel_time
double dt (double t, void *params){
	double C0 = *(double *) params;
	double z = get_z_unmirrored(t,C0);
	return sqrt((pow(get_y_diff(t,C0),2.)+1)) / speed_of_light * index_vs_depth(z);
}

double get_travel_time(double pos[2], double pos2[2], double C0){
	double x2_mirrored[2]={0.};
	get_z_mirrored(pos,pos2,C0,x2_mirrored);
	
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
	gsl_function F;
	F.function = &dt;
	F.params=&C0;
	
	double result, error;
	
	gsl_integration_qags(&F, pos[1], x2_mirrored[1],0,1e-7,1000,w,&result,&error);
	gsl_integration_workspace_free(w);
	return result;
}

double get_temperature(double z){
	//return temperature as a function of depth
	return (-51.5 + z * (-4.5319e-3 + 5.822e-6 * z));
}

double get_attenuation_length(double z, double frequency){
	double t = get_temperature(z);
	double f0 = 0.0001;
	double f2 = 3.16;
	double w0 = log(f0);
	double w1 = 0.0;
	double w2 = log(f2); 
	double w = log(frequency); //units of GHz
	double b0 = -6.74890 + t * (0.026709 - t * 0.000884);
	double b1 = -6.22121 - t * (0.070927 + t * 0.001773);
	double b2 = -4.09468 - t * (0.002213 + t * 0.000332);
	double a, bb;
	if(frequency<1.){
		a = (b1 * w0 - b0 * w1) / (w0 - w1);
		bb = (b1 - b0) / (w1 - w0);
	}
	else{
		a = (b2 * w1 - b1 * w2) / (w1 - w2);
		bb = (b2 - b1) / (w2 - w1);
	}
	return 1./exp(a +bb*w);
}

//this function is explicitly prepared for gsl integration in get_attenuation_along_path
struct dt_freq_params{ double a; double c;}; //a=C0, c=freq
double dt_freq (double t, void *p){
	struct dt_freq_params *params = (struct dt_freq_params *)p;
	double C0 = (params->a);
	double freq = (params->c);
	double z = get_z_unmirrored(t,C0);
	return sqrt((pow(get_y_diff(t,C0),2.)+1)) / get_attenuation_length(z,freq);
}

double get_attenuation_along_path(double pos[2], double pos2[2], double C0, double frequency){
	double x2_mirrored[2]={0.};
	get_z_mirrored(pos,pos2,C0,x2_mirrored);
	
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
	gsl_function F;
	F.function = &dt_freq;
	struct dt_freq_params params = {C0,frequency};
	F.params=&params;

	double result, error;
	
	gsl_integration_qags(&F, pos[1], x2_mirrored[1],0,1e-7,1000,w,&result,&error);
	gsl_integration_workspace_free(w);
	double attenuation = exp(-1 * result);
	return attenuation;
}

double get_angle(double x[2], double x_start[2], double C0){
	double result[2]={0.};
	get_z_mirrored(x_start,x,C0,result);
	double z = result[1];
	double dy = get_y_diff(z,C0);
	double angle = atan(dy);
	if(angle < 0.) angle += pi;
	return angle;
}

double get_launch_angle(double x1[2], double C0){
	return get_angle(x1,x1,C0);
}

double get_receive_angle(double x1[2], double x2[2], double C0){
	return pi - get_angle(x2,x1,C0);
}

double get_delta_y(double C0, double x1[2], double x2[2]){
	//calculates the difference in the y position between the analytic ray tracing path
	//specified by C0 at the position x2
	
	double lower_bound = 1./n_ice;
	double upper_bound = inf; 
	if(C0<lower_bound || C0>upper_bound) {return inf;}
	double c = pow(n_ice,2.) - pow(C0,-2.);
	//determine y translation
	double C1 = x1[0] - get_y_with_z_mirror(x1[1],C0);
	
	//for a given C0, 3 cases are possible to reach the position of x2
	//1: Direct ray--before the turning point
	//2: Refracted ray--after the turning point but not touching surface
	//3: Reflected ray--after the ray reaches the surface
	
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn);
	if(z_turn > 0.){
		z_turn = 0.; //a reflection is just a turning point at z=0, ie case 2 and 3 are the same
		gamma_turn = get_gamma(z_turn);
	}
	double y_turn = get_y(gamma_turn,C0,C1);
	if(z_turn < x2[1]){ //turning points is deeper than x2 positions, can't reach target
		return -inf;
	}
	if(y_turn > x2[0]){//always propagate from left to right
		//direct ray
		double y2_fit = get_y(get_gamma(x2[1]),C0,C1); //calculate the y position at get_path position
		double diff = (x2[0] - y2_fit);
		return diff;
	}
	else{
		//now it's a bit more complicated; we need to transform the coordinates to be on the mirrored part of the function
		double z_mirrored = x2[1];
		double gamma = get_gamma(z_mirrored);
		double y2_raw = get_y(gamma, C0, C1);
		double y2_fit = 2 * y_turn-y2_raw;
		double diff = x2[0] - y2_fit;
		return -1*diff;
	}
}

int determine_solution_type(double x1[2], double x2[2], double C0){
	//return 1 for direct solution
	//return 2 for reflected
	//return 3 for refraced
	
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double C1 = x1[0] - get_y_with_z_mirror(x1[1],C0);
	
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn);
	
	if(z_turn >= 0.){
		z_turn=0.;
		gamma_turn = get_gamma(0);
	}
	double y_turn = get_y(gamma_turn,C0,C1);
	if(x2[0] < y_turn) return 1; //direct
	else{
		if(abs(z_turn-0.0)<1e-6) return 2; //reflected, trying to do z_turn==0, but == is bad with doubles
		else return 3; //refracted
	}
}

//deprecated, but here for posterity
/*
double obj_delta_y_square(double logC0, double x1[2], double x2[2]){
	//objective function to find solution for C0
	double C0 = get_C0_from_log(logC0);
	return pow(get_delta_y(C0,x1,x2),2.);
}
*/

//this function is explicitly prepared for gsl root finding in find_solutions
struct obj_delta_y_square_params{double x1_x; double x1_z; double x2_x; double x2_z;}; //x1_x=x1[0] and so forth
double obj_delta_y_square(double logC0, void *p){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double C0 = get_C0_from_log(logC0);
	return pow(get_delta_y(C0,x1,x2),2.);
}

//this function is explicity prepared for gsl root finding in find_solutions
//it is the "derivative"
//the derivative is f(x+h) - f(x) / h for small h
//so let's just hard code that
double obj_delta_y_square_df(double logC0, void *p){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double C0 = get_C0_from_log(logC0);
	double increment_size = C0/10000.;	//our small h
	return (pow(get_delta_y(C0+increment_size,x1,x2),2.)-pow(get_delta_y(C0,x1,x2),2.))/increment_size; //definition of derivative
}

//this function is explicity prepared for gsl root finding in find_solutions
//it is the "f * derivative"
void obj_delta_y_square_fdf(double logC0, void *p, double *y, double *dy){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double C0 = get_C0_from_log(logC0);
	double increment_size = C0/10000.;	//our small h
	*y = pow(get_delta_y(C0,x1,x2),2.);
	*dy = (pow(get_delta_y(C0+increment_size,x1,x2),2.)-pow(get_delta_y(C0,x1,x2),2.))/increment_size; //definition of derivative
}
/*
double obj_delta_y(double logC0, double x1[2], double x2[2]){
	//function to find solution for C0, returns distance in y between function and x2 position
	//result is signed! (important to use root finder)
	double C0 = get_C0_from_log(logC0);
	return get_delta_y(C0,x1,x2);
}
*/


//this function is explicitly prepared for gsl root finding in find_solutions
double obj_delta_y(double logC0, void *p){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double C0 = get_C0_from_log(logC0);
	return get_delta_y(C0,x1,x2);
}

vector <vector <double> > find_solutions(double x1[2], double x2[2]){
	//function finds all ray tracing solutions
	//we assume that x2 is above and to the right of x2_mirrored
	//this is perfectly general, as a coordinate transform can put any system in this configuration
	
	//returns a vector of vectors of the C0 solutions
	//entry 0 will be logC0
	//entry 1 will be CO
	//entry 2 will be C1
	//entry 3 will be type
	vector < vector <double> > results;
	
	struct obj_delta_y_square_params params = {x1[0],x1[1],x2[0],x2[1]};
	
	
	/////////
	////Find root 1: a first solution; check around logC0=-1
	/////////
	
	int status;
	int iter=0, max_iter=20000;
	double x_guess = -1;
	bool found_root_1=false;
	double root_1=-10000000; //some insane value we'd never believe
	

		
	const gsl_root_fdfsolver_type *Tfdf;
	gsl_root_fdfsolver *sfdf;
	gsl_function_fdf FDF;
	FDF.f = &obj_delta_y_square;
	FDF.df = &obj_delta_y_square_df;
	FDF.fdf = &obj_delta_y_square_fdf;
	FDF.params = &params;
	Tfdf = gsl_root_fdfsolver_secant;
	sfdf = gsl_root_fdfsolver_alloc(Tfdf);
	int num_badfunc_tries = 0;
	int max_badfunc_tries = 50;
	gsl_root_fdfsolver_set(sfdf,&FDF,x_guess);
		gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
		do{
			iter++;
			//cout<<"Got to iter "<<iter<<" val is "<<GSL_FN_FDF_EVAL_F(&FDF,x_guess)<<endl;
			status = gsl_root_fdfsolver_iterate(sfdf);
			//we need to manually protect against the function blowing up, which *is an error*, but will casue GSL to fail
			//so, if we get a GSL_EBADFUNC, we want to manually say skip this, but re-enable the continue flag
			if(status==GSL_EBADFUNC) {status=GSL_CONTINUE; num_badfunc_tries++; continue;} 
			root_1 = x_guess;
			x_guess = gsl_root_fdfsolver_root(sfdf);
			status = gsl_root_test_delta(x_guess,root_1,0,0.0000001);
			if(status == GSL_SUCCESS){
				//printf("Converged on root 1! Iteration %d\n",iter);
				found_root_1=true;
			}
		} while (status == GSL_CONTINUE && iter < max_iter && num_badfunc_tries<max_badfunc_tries);
		gsl_set_error_handler (myhandler); //restore original error handler
	gsl_root_fdfsolver_free (sfdf);
	
	if(found_root_1){
		vector <double> sol1;
		sol1.push_back(root_1);
		double C0 = get_C0_from_log(root_1);
		sol1.push_back(C0);
		sol1.push_back(get_C1(x1,C0));
		sol1.push_back(ceil(double(determine_solution_type(x1,x2,C0))));
		
		printf("Solution 1 [logC0, C0, C1, type]: [%.4f, %.4f, %.4f, %f]]\n",sol1[0],sol1[1],sol1[2],sol1[3]);
		
		results.push_back(sol1);
	}
	
	//reset this counter
	num_badfunc_tries = 0;
	
	/////////
	////Find root 2: a second solution
	/////////
	
	//now to check if another solution with higher logC0 exists
	//if the above algorithm failed, then we have to be more brute-forcy in the next go around
	double logC0_start;
	if(!found_root_1)  logC0_start=0.;
	else logC0_start = root_1+0.0001;
	
	gsl_function F;
	F.function = &obj_delta_y;
	F.params = &params;
		
	double logC0_stop = 100.;
	double delta_start = GSL_FN_EVAL(&F,logC0_start);
	double delta_stop = GSL_FN_EVAL(&F,logC0_stop);
	bool found_root_2 = false;
	double root_2 = -10000000;
	if(signbit(delta_start)!=signbit(delta_stop)){
		//printf("Solutions with logc0 > %.3f exist\n",root_1);
		
		//now we must solve again
		//let's use Brent's method, which should be faster
		//now that we know there is a y=0 (x-axis) crossing
		int status2;
		const gsl_root_fsolver_type *T;
		gsl_root_fsolver *s;

		T = gsl_root_fsolver_brent;
		s = gsl_root_fsolver_alloc(T);
		gsl_root_fsolver_set(s, &F, logC0_start, logC0_stop);
			gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
			iter=0;
			do{
				iter++;
				status2 = gsl_root_fsolver_iterate(s);
				logC0_start = gsl_root_fsolver_x_lower(s);
				logC0_stop = gsl_root_fsolver_x_upper(s);
				//printf("[Iter, Xlo, Xhi]: [%d, %.8f, %.8f] \n",iter,logC0_start,logC0_stop);
				status2 = gsl_root_test_interval(logC0_start,logC0_stop,0,0.000001);
				if(status2==GSL_EBADFUNC) {status2=GSL_CONTINUE; num_badfunc_tries++; continue;} 
				if(status2 == GSL_SUCCESS){
					//printf("Converged on root 2! Iteration %d\n",iter);
					found_root_2=true;
					root_2 = gsl_root_fsolver_root(s);
				}
			} while (status2 == GSL_CONTINUE && iter < max_iter && num_badfunc_tries<max_badfunc_tries);
			gsl_set_error_handler (myhandler); //restore original error handler
		gsl_root_fsolver_free (s);
		
		if(found_root_2){
			vector <double> sol2;
			sol2.push_back(root_2);
			double C0 = get_C0_from_log(root_2);
			sol2.push_back(C0);
			sol2.push_back(get_C1(x1,C0));
			sol2.push_back(ceil(double(determine_solution_type(x1,x2,C0))));
		
			printf("Solution 2 [logC0, C0, C1, type]: [%.4f, %.4f, %.4f, %f]]\n",sol2[0],sol2[1],sol2[2],sol2[3]);
		
			results.push_back(sol2);
		}
	}
	else printf("No solution with logc0 > %.3f exist\n",logC0_start);
	
	//reset this counter
	num_badfunc_tries = 0;
	
	/////////
	////Find root 3: a third solution
	/////////
	
	//now to check if another solution with lower logC0 exists
	//if the above algorithm failed, then we have to be more brute-forcy in the next go around
	
	if(!found_root_1)  logC0_stop=0.0001;
	else logC0_stop = root_1-0.0001;
		
	logC0_start = -100.;
	delta_start = GSL_FN_EVAL(&F,logC0_start);
	delta_stop = GSL_FN_EVAL(&F,logC0_stop);
	bool found_root_3 = false;
	double root_3 = -10000000;
	if(signbit(delta_start)!=signbit(delta_stop)){
		//printf("Solutions with logc0 > %.3f exist\n",root_1);
		
		//now we must solve again
		//let's use Brent's method, which should be faster
		//now that we know there is a y=0 (x-axis) crossing
		int status3;
		const gsl_root_fsolver_type *T;
		gsl_root_fsolver *s;
		gsl_function F;
		F.function = &obj_delta_y;
		F.params = &params;
		T = gsl_root_fsolver_brent;
		s = gsl_root_fsolver_alloc(T);
			gsl_root_fsolver_set(s, &F, logC0_start, logC0_stop);
			gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
			iter=0;
			do{
				iter++;
				status3 = gsl_root_fsolver_iterate(s);
				logC0_start = gsl_root_fsolver_x_lower(s);
				logC0_stop = gsl_root_fsolver_x_upper(s);
				//printf("[Iter, Xlo, Xhi]: [%d, %.8f, %.8f] \n",iter,logC0_start,logC0_stop);
				status3 = gsl_root_test_interval(logC0_start,logC0_stop,0,0.000001);
				if(status3==GSL_EBADFUNC) {status3=GSL_CONTINUE; num_badfunc_tries++; continue;} 
				if(status3 == GSL_SUCCESS){
					//printf("Converged on root 3! Iteration %d\n",iter);
					found_root_3=true;
					root_3 = gsl_root_fsolver_root(s);
				}
			} while (status3 == GSL_CONTINUE && iter < max_iter && num_badfunc_tries<max_badfunc_tries);
			gsl_set_error_handler (myhandler); //restore original error handler
		gsl_root_fsolver_free (s);
		
		if(found_root_3){
			vector <double> sol3;
			sol3.push_back(root_3);
			double C0 = get_C0_from_log(root_3);
			sol3.push_back(C0);
			sol3.push_back(get_C1(x1,C0));
			sol3.push_back(ceil(double(determine_solution_type(x1,x2,C0))));
		
			printf("Solution 3 [logC0, C0, C1, type]: [%.4f, %.4f, %.4f, %f]]\n",sol3[0],sol3[1],sol3[2],sol3[3]);
		
			results.push_back(sol3);
		}
	}
	else printf("No solution with logc0 < %.3f exist\n",logC0_stop);
	
	return results;
}

void get_path(double x1[2], double x2[2], double C0, vector<double> &res, vector<double> &zs, int n_points=100){
	
	//will return the ray tracing path between x1 and x2
	//this is only true if C0 is a solution to the ray tracing problem
	
	//parameters
	//x1: start position (y,z)
	//x2: stop position (y,z)
	//C0: first parameter
	//n_points: number of points to calcuate
	
	//returns two vectors by reference
	//res are y coordinates
	//zs are z coordinates
	
	
	double c = pow(n_ice,2.) - pow(C0,-.2);
	double C1 = x1[0] - get_y_with_z_mirror(x1[1],C0);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn);
	if(z_turn >=0.){
		//signal reflects at surface
		z_turn=0.;
		gamma_turn = get_gamma(0);
	}
	double y_turn = get_y(gamma_turn, C0, C1);
	double zstart = x1[1];
	double result[2];
	get_z_mirrored(x1,x2,C0,result);
	double zstop = result[1];
	double step_size = (zstop-zstart)/double(n_points-1); //do n-1 so that the bounds are actually the bounds
	vector<double> z; //vector to hold z's
	for(int i=0; i<n_points; i++){
		z.push_back(zstart+i*step_size);
	}
	
	//c++ has no clever "masking" tools like python
	//so we have to do this the old fashioned way
		
	//some temporary stuff
	for(int i=0; i<n_points; i++){
		double gamma_temp;
		if(z[i]<z_turn){
			gamma_temp = get_gamma(z[i]);
			res.push_back(get_y(gamma_temp,C0,C1));
			zs.push_back(z[i]);
		}
		else{
			gamma_temp = get_gamma(2 * z_turn - z[i]);
			res.push_back(2*y_turn - get_y(gamma_temp,C0,C1));
			zs.push_back(2*z_turn - z[i]);
		}
	}
}
	
int main(int argc, char **argv){
	
	//Test 1, n(z)
	double test_nofz = index_vs_depth(-10.);
	printf("Test 1: Index %f\n",test_nofz);
	
	//Test 2, gamma
	double test_gamma = get_gamma(-10.);
	printf("Test 2: Gamma %f\n",test_gamma);
		
	//Test 3, turning point
	double gamma2, z2;
	get_turning_point(2.8,gamma2,z2);
	printf("Test 3: Turning Point %f, %f\n",gamma2, z2);
	
	//Test 4, gety
	double y = get_y(0.37,0.61,10);
	printf("Test 4: Y %f\n",y);	
	
	//Test 5, GetYWithZMirro
	double y_mirror = get_y_with_z_mirror(-10,0.61);
	printf("Test 5: Y with Z Mirror %f\n",y_mirror);	
	
	//Test 6, GetC1
	double pos[2] = {20,-10};
	double C1 = get_C1(pos,0.61);
	printf("Test 6: C1 %f\n",C1);	
	
	//Test 7, GetC
	double c = get_c(0.61);
	printf("Test 7: c %f\n",c);
	
	//Test 8, GetC0FromLog
	double c0_from_log = get_C0_from_log(-0.5);
	printf("Test 8: C0 From Log %f\n",c0_from_log);
	
	//Test 9, GetZUnmirrored
	double z_unmirrored = get_z_unmirrored(-10.,0.61);
	printf("Test 9: Z Unmirorred %f\n",z_unmirrored);
	
	//Test 10, GetYDiff
	double y_diff = get_y_diff(-10.,0.61);
	printf("Test 10: Y Diff %f\n",y_diff);
	
	//Test 11, Ds
	//double DS = ds(100.,0.61);
	//printf("Test 11: Ds %f\n",DS);
	
	//Test 12, Z Mirrored
	double test_pos[2]={0,-100.};
	double test_pos2[2]={10,-90.};
	double result[2]={0};
	get_z_mirrored(test_pos, test_pos2,0.61,result);
	printf("Test 12: Mirrored z %f, %f\n",result[0],result[1]);	
	
	//Test 13, Path Length
	double path_length = get_path_length(test_pos, test_pos2, 0.61);
	printf("Test 13: Path Length %f\n",path_length);

	//Test 14, Travel Time
	double travel_time = get_travel_time(test_pos, test_pos2, 0.61);
	printf("Test 14: Travel Time %.10f\n",travel_time);
	
	//Test 15, Attenuation Length
	double att_length = get_attenuation_length(-10,0.1);
	printf("Test 15: Att Length %f\n",att_length);

	//Test 16, Attenuation Length Along Path
	double att_length_along_path = get_attenuation_along_path(test_pos,test_pos2,0.61,0.1);
	printf("Test 16: Att Along path %f\n",att_length_along_path);
	
	//Test 17, Get Angle
	double angle = get_angle(test_pos,test_pos2,0.61);
	printf("Test 17: Angle %f\n",angle);
	
	//Test 18, Launch and Receive
	double launch = get_launch_angle(test_pos,0.61);
	double receive = get_receive_angle(test_pos,test_pos2,0.61);
	printf("Test 18: Launch and Receive %f, %f\n",launch, receive);
	
	//Test 19, GetDeltaY
	double deltay = get_delta_y(0.61,test_pos,test_pos2);
	printf("Test 19: Delta Y %f\n",deltay);
	
	//Test 20, DeltaYSquare
	//double obj_deltaysquare = obj_delta_y_square(-0.49,test_pos,test_pos2);
	//double obj_deltay = obj_delta_y(-0.49,test_pos,test_pos2);
	//printf("Test 20: Obj DeltYSquare, Obj DeltY %f, %f\n",obj_deltaysquare, obj_deltay);
	
	//Test 21, Solution Type
	int solution = determine_solution_type(test_pos,test_pos2,0.61);
	printf("Test 21: Solution Type %d\n",solution);
	
	//Test 22, Find Root
	vector <vector <double> > solutions = find_solutions(test_pos, test_pos2);
	printf("Test 22: Solutions %d\n",int(solutions.size()));
	
	
	//Test 23, Print Path
	vector<double> res;
	vector<double> zs;
	get_path(test_pos, test_pos2,0.8479,res,zs,10);
	printf("Test 23: Print Path\n");
	for(int i=0; i<int(res.size());i++) printf("Element num, z, y: [%d, %f, %f]\n",i,res[i],zs[i]);
	
	
	//okay, now let's try and get a ray
	double x1[2] = {478., -149.};
	double x2[2] = {635., -5.}; //a direct ray solution
	vector<vector<double> > direct_solutions = find_solutions(x1,x2);
	vector <double> direct_res;
	vector <double> direct_zs;
	get_path(x1,x2,direct_solutions[0][1],direct_res,direct_zs,10);
	for(int i=0; i<int(direct_res.size());i++) printf("Element num, z, y: [%d, %f, %f]\n",i,direct_res[i],direct_zs[i]);

	
	
	return 0;
}

