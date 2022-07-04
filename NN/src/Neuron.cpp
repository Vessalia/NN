#include "Neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(size_t numOutputs, size_t index)
	: _index(index)
{
	for (size_t conn = 0; conn < numOutputs; ++conn)
	{
		_outputWeights.push_back(Connection());
		_outputWeights.back().weight = Neuron::randomWeight();
	}
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;

	for (size_t n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n]._outputWeights[_index].weight;
	}

	_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - _outputVal;
	_gradient = delta * Neuron::transferFunctionDerivative(_outputVal);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	_gradient = dow * Neuron::transferFunctionDerivative(_outputVal);
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	for (size_t n = 0; n < prevLayer.size(); ++n)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron._outputWeights[_index].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * _gradient
			+ alpha * oldDeltaWeight;

		neuron._outputWeights[_index].deltaWeight = newDeltaWeight;
		neuron._outputWeights[_index].weight += newDeltaWeight;
	}
}

unsigned int Neuron::slope = 1;
unsigned int Neuron::leakySlope = 100;

double Neuron::transferFunction(double x)
{
	return x > 0 ? slope * x : x / leakySlope;
}

double Neuron::transferFunctionDerivative(double x)
{
	return x > 0 ? slope : 1.0 / leakySlope;
}

/*
* Sums derivative of weights from next layer to approx suitable delta error
*/
double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	for (size_t n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += _outputWeights[n].weight * nextLayer[n]._gradient;
	}

	return sum;
}
