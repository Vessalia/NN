#pragma once

#include "NeuralNet.h"

class Agent
{
public:
	Agent(NeuralNet brain);

	double train(const std::vector<double>& inputVals, const std::vector<double>& targetVals);
	
	float checkFitness();
	Agent reproduce(float mutationRate);
	void mutate(float mutationRate);

private:
	NeuralNet brain;
};
