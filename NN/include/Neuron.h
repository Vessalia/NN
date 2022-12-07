#pragma once

#include <vector>
#include <cstdlib>

class Neuron
{
private:
	friend class NeuralNet;

	typedef std::vector<Neuron> Layer;

	Neuron(size_t numOutputs, size_t index, bool doLeaky);

	void setOutputVal(double val) { m_outputVal = val; };
	double getOutputVal(void) const { return m_outputVal; }

	double getOutputWeight(size_t index) const { return m_outputWeights[index].weight; }

	void feedForward(const Layer& prevLayer);

	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

	struct Connection
	{
		double weight;
		double deltaWeight;
	};

	static double epsilon;
	static double alpha;

	static unsigned int slope;
	static unsigned int leakySlope;

	bool m_doLeaky;

	size_t m_index;
	
	double m_gradient;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;

	static double leakyTransferFunction(double x);
	static double leakyTransferFunctionDerivative(double x);
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return randomWeight(0, 1); }
	static double randomWeight(double min, double max) { return rand() / double(RAND_MAX) * (max - min) + min; }

	double sumWeightedDerivatives(const Layer& nextLayer) const;
};
