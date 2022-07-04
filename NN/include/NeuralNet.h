#pragma once

#include "Neuron.h"
#include <vector>

typedef std::vector<Neuron> Layer;

class NeuralNet
{
public:
	NeuralNet(const std::vector<size_t> &topology);

	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals) const;
	double getRecentAverageError(void) { return _recentAverageError; };

	void mutate();

private:
	std::vector<Layer> _layers;

	double _error;
	double _recentAverageError;
	static double _recentAverageSmoothingFactor;
};