/*
Credit for the development of this neural network is attributed to Dave Miller
url: https://www.millermattson.com/dave/?p=54
*/

#include "Neuron.h"

double Neuron::epsilon = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(size_t numOutputs, size_t index, bool doLeaky)
	: m_index(index)
	, m_doLeaky(doLeaky)
{
	for (size_t conn = 0; conn < numOutputs; ++conn)
	{
		m_outputWeights.emplace_back(Connection());
		m_outputWeights.back().weight = Neuron::randomWeight(-1, 1);
	}
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;

	for (size_t n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_index].weight;
	}

	if (m_doLeaky) m_outputVal = Neuron::leakyTransferFunction(sum);
	else m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	if(m_doLeaky) m_gradient = delta * Neuron::leakyTransferFunctionDerivative(m_outputVal);
	else m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	double sum = sumWeightedDerivatives(nextLayer);
	if(m_doLeaky) m_gradient = sum * Neuron::leakyTransferFunctionDerivative(m_outputVal);
	else m_gradient = sum * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	for (size_t n = 0; n < prevLayer.size(); ++n)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

		double newDeltaWeight = epsilon * neuron.getOutputVal() * m_gradient
			+ alpha * oldDeltaWeight;

		neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_index].weight += newDeltaWeight;
	}
}

unsigned int Neuron::slope = 1;
unsigned int Neuron::leakySlope = 10;

double Neuron::leakyTransferFunction(double x)
{
	return x > 0 ? slope * x : x / leakySlope;
}

double Neuron::leakyTransferFunctionDerivative(double x)
{
	return x > 0 ? slope : 1.0 / leakySlope;
}

double Neuron::transferFunction(double x)
{
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return 1 - tanh(x) * tanh(x);
}

double Neuron::sumWeightedDerivatives(const Layer& nextLayer) const
{
	double sum = 0.0;

	for (size_t n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}
