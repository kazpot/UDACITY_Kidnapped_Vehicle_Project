/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

#define M_PI 3.14159265358979323846

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[0]);
	std::normal_distribution<double> dist_theta(theta, std[0]);

	for (int i=0; i<num_particles; ++i){
		Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
		particles.push_back(p);
	}
	weights.resize(num_particles);
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_yaw(0, std_pos[2]);

	for (int i=0; i<num_particles; ++i){
		Particle p = particles[i];
		if(yaw_rate != 0){
		   p.x = p.x + velocity/yaw_rate * ((sin(p.theta) + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
		   p.y = p.y + velocity/yaw_rate * ((cos(p.theta) - cos(p.theta + yaw_rate * delta_t))) + dist_y(gen);
		   p.theta = p.theta + yaw_rate * delta_t + dist_yaw(gen);
		}else{
			p.x = p.x + velocity * delta_t * cos(p.theta) + dist_x(gen);
			p.y = p.y + velocity * delta_t * sin(p.theta) + dist_y(gen);
			p.theta = p.theta + dist_yaw(gen);
		}
		particles[i] = p;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i<observations.size(); ++i){
		double dist = 0.0;
		double min_dist = std::numeric_limits<double>::infinity();
		for (int j=0; i<predicted.size(); ++i){
			double x_dist = observations[i].x - predicted[j].x;
			double y_dist = observations[i].y - predicted[j].y;
			dist = sqrt(x_dist * x_dist + y_dist * y_dist);
			if(dist < min_dist){
				min_dist = dist;
				observations[i].id = predicted[j].id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	
	for (int i=0; i < num_particles; ++i){
		Particle p = particles[i];

		double px = p.x;
		double py = p.y;
		double ptheta = p.theta;

        std::vector<LandmarkObs> transformed_obs;
		for (int j=0; j < observations.size(); ++j){
            double tx = observations[j].x * cos(ptheta) - observations[j].y * sin(ptheta) + px;
            double ty = observations[j].x * sin(ptheta) + observations[j].y * cos(ptheta) + py;
            transformed_obs.push_back(LandmarkObs{observations[j].id, tx, ty});
		}

		std::vector<LandmarkObs> predicted;
		for (int k=0; k < map_landmarks.landmark_list.size(); ++k){
		   int id = map_landmarks.landmark_list[k].id_i;
		   float landx = map_landmarks.landmark_list[k].x_f;
		   float landy =map_landmarks.landmark_list[k].y_f;
	       if (fabs(landx - px) <= sensor_range && fabs(landy - py) <= sensor_range){
		      predicted.push_back(LandmarkObs{id, landx, landy});
		   }
		}

		dataAssociation(predicted, transformed_obs);

		double weight = 1.0;
		for (int l=0; l < transformed_obs.size(); ++l){
           double prx = predicted[transformed_obs[l].id].x;
           double pry = predicted[transformed_obs[l].id].y;

		   double obsx = transformed_obs[l].x;
           double obsy = transformed_obs[l].y;

           double sigmax = std_landmark[0];
           double sigmay = std_landmark[1];

           double a = 1 / (2 * M_PI * sigmax * sigmay);
           double b = ((obsx - prx) * (obsx - prx))/(2 * sigmax * sigmax); 
           double c = ((obsy - pry) * (obsy - pry))/(2 * sigmay * sigmay); 
           
           double weight = a * exp(-(b + c));

           particles[i].weight *= weight;
		}
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> new_particles;
    
    double maxw = *max_element(begin(weights), end(weights));

    std::mt19937 generator (123);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    double randomRealBetweenZeroAndOne = dis(generator);

    int index = int(randomRealBetweenZeroAndOne * (num_particles - 1));
    double beta = 0.0;

    //resample wheel
    for (int i=0; i < num_particles; ++i){
    	beta += randomRealBetweenZeroAndOne * 2.0 * maxw;
    	while(beta > weights[index]){
    		beta -= weights[index];
    		index = (index + 1) % num_particles;
    	}
    	new_particles.push_back(particles[index]);
    }
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
