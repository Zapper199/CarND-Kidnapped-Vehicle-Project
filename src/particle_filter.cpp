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
#include <cfloat>


#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	particles.resize(num_particles);
	weights.resize(num_particles, 0);

	// Initialize all particles to this position
	std::default_random_engine gen;
  	std::normal_distribution<double> x_dist(x, std[0]);
  	std::normal_distribution<double> y_dist(y, std[1]);
  	std::normal_distribution<double> theta_dist(theta, std[2]);

  	for (int i = 0; i < num_particles; i++) {
  		particles[i].id = i;
  		particles[i].x = x_dist(gen);
  		particles[i].y = y_dist(gen);
  		particles[i].theta = theta_dist(gen);
  		particles[i].weight = 1;
  	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;
	for (Particle &particle : particles) {
		double x_f = particle.x + (velocity/yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
		double y_f = particle.y + (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
		double theta_f = particle.theta + yaw_rate * delta_t;
		std::normal_distribution<double> x_dist(x_f, std_pos[0]);
  		std::normal_distribution<double> y_dist(y_f, std_pos[1]);
  		std::normal_distribution<double> theta_dist(theta_f, std_pos[2]);

  		particle.x = x_dist(gen);
  		particle.y = y_dist(gen);
  		particle.theta = theta_dist(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (LandmarkObs &observation : observations) {
		int closestLandmark;
		double smallestDistance = DBL_MAX;
		for (LandmarkObs &prediction : predicted) {
			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
			if (distance <= smallestDistance) {
				smallestDistance = distance;
				closestLandmark = prediction.id;
			}
		}

		observation.id = closestLandmark;
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
	//   http://planning.cs.uiuc.edu/node99.html

	// For each particle
	for (int i = 0; i < particles.size(); i++) {
		double sum = 0;
		Particle particle = particles[i];

		// Make a list of predicted measurements for this particle
		std::vector<LandmarkObs> predicted(map_landmarks.landmark_list.size());
		for (int j = 0; j < predicted.size(); j++) {
			Map::single_landmark_s landmark = map_landmarks.landmark_list[j];

			double x_diff = landmark.x_f - particle.x;
			double y_diff = landmark.y_f - particle.y;

			double x_pred = y_diff * sin(particle.theta) + x_diff * cos(particle.theta);
			double y_pred = y_diff * cos(particle.theta) - x_diff * sin(particle.theta);

			predicted[j].x = x_pred;
			predicted[j].y = y_pred;
			predicted[j].id = landmark.id_i;
		}

		// Find closest landmarks for each observation
		dataAssociation(predicted, observations);

		// For each observation, find the error
		for (LandmarkObs obs : observations) {
			// Find the probability of the observation

			// First transform observation to map coordinates
			double obs_x = particle.x + obs.x * cos(particle.theta);
			double obs_y = particle.y = obs.y * sin(particle.theta);

			// What's the probability that I make this observation?
			double true_x = map_landmarks.landmark_list[obs.id-1].x_f;
			double true_y = map_landmarks.landmark_list[obs.id-1].y_f;

			double obs_prob = (1/(2*M_PI*std_landmark[0]*std_landmark[1]))
				* exp(-(((true_x - obs_x)*(true_x-obs_x))/(2*std_landmark[0]*std_landmark[0]))
					+ (((true_y - obs_y)*(true_y-obs_y))/(2*std_landmark[1]*std_landmark[1])));
			weights[i] *= obs_prob;
		}
	}

	// Normalize weights
	double sum = 0;
	for (int i = 0; i < weights.size(); i++) sum += weights[i];
	for (int i = 0; i < weights.size(); i++) weights[i] /= sum;

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine gen;
	std::discrete_distribution<int> dist(weights.begin(), weights.end());

	std::vector<Particle> new_particles;
	new_particles.reserve(particles.size());

	for (int i = 0; i < particles.size(); i++) {
		int sampled_index = dist(gen);
		new_particles.push_back(particles[sampled_index]);
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
