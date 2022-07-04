#pragma once

#include <vector>
#include <cstdlib>

class Neuron
{
	typedef std::vector<Neuron> Layer;

public:
	Neuron(size_t numOutputs, size_t index);

	void setOutputVal(double val) { _outputVal = val; };
	double getOutputVal(void) const { return _outputVal; }

	void feedForward(const Layer& prevLayer);

	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

private:
	struct Connection
	{
		double weight;
		double deltaWeight;
	};

	static double eta;
	static double alpha;

	static unsigned int slope;
	static unsigned int leakySlope;

	size_t _index;
	
	double _gradient;
	double _outputVal;
	std::vector<Connection> _outputWeights;

	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }

	double sumDOW(const Layer& nextLayer) const;
};
