#include "Agent.h"

Agent::Agent(NeuralNet brain)
	: brain(brain) { }

double Agent::train(const std::vector<double>& inputVals, const std::vector<double>& targetVals)
{
	brain.feedForward(inputVals);
	brain.backProp(targetVals);

	return brain.getRecentAverageError();
}

float Agent::checkFitness()
{
	return 0.0f;
}

Agent Agent::reproduce(float mutationRate)
{
	Agent child = *this;
	child.mutate(mutationRate);
	return child;
}

void Agent::mutate(float mutationRate)
{
	float mutationChance = rand() / RAND_MAX;
	if (mutationChance > mutationRate)
	{
		brain.mutate();
	}
}
