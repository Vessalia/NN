#include "NeuralNet.h"
#include <assert.h>

double NeuralNet::m_recentAverageSmoothingFactor = 100.0;

NeuralNet::NeuralNet(const std::vector<size_t>& topology, bool doLeaky)
	: m_error(0.0)
	, m_recentAverageError(0.0)
	, m_numNodes(0)
{
	size_t numLayers = topology.size();
	for (size_t layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.emplace_back(Neuron::Layer());
		size_t numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (size_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().emplace_back(Neuron(numOutputs, neuronNum, doLeaky));
			m_numNodes++;
		}

		m_layers.back().back().setOutputVal(1.0);
	}
}

void NeuralNet::feedForward(const std::vector<double>& inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	for (size_t i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	for (size_t layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		auto &prevLayer = m_layers[layerNum - 1];
		for (size_t n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void NeuralNet::backProp(const std::vector<double>& targetVals)
{
	auto& outputLayer = m_layers.back();
	m_error = 0.0;

	for (size_t n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}

	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error);

	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);



	for (size_t n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	for (size_t layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		auto& hiddenLayer = m_layers[layerNum];
		auto& nextLayer = m_layers[layerNum + 1];

		for (size_t n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	for (size_t layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		auto& layer = m_layers[layerNum];
		auto& prevLayer = m_layers[layerNum - 1];

		for (size_t n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void NeuralNet::getResults(std::vector<double>& resultVals) const
{
	resultVals.clear();

	auto outputNeurons = m_layers.back();

	for (size_t n = 0; n < outputNeurons.size() - 1; ++n)
	{
		resultVals.push_back(outputNeurons[n].getOutputVal());
	}
}

size_t NeuralNet::getNumConnections(void) const
{
	{
		size_t num = 0;
		for (size_t i = 0; i < m_layers.size() - 1; ++i)
		{
			num += m_layers[i].size() * m_layers[i + 1].size();
		}

		return num;
	}
}

double NeuralNet::getNodeWeight(size_t index) const
{
	if (index > getNumNodes() - 1)
	{
		exit(1);
	}

	size_t visited = 0;
	size_t layer = 0;
	size_t node = 0;
	for (size_t i = 0; i < m_layers.size(); ++i)
	{
		if (visited + m_layers[i].size() - 1 < index)
		{
			visited += m_layers[i].size();
			layer++;
		}
		else
		{
			for (int j = 0; j < m_layers[i].size(); ++j)
			{
				if (node + visited < index)
				{
					node++;
				}
				else
				{
					break;
				}
			}

			break;
		}
	}
	return m_layers[layer][node].getOutputVal();
}

bool NeuralNet::isBiasNodeIndex(size_t index) const
{
	bool isBias = false;
	size_t visited = 0;
	for (size_t i = 0; i < m_layers.size() && !isBias; ++i)
	{
		visited += m_layers[i].size();
		if (visited - 1 > index)
		{
			break;
		}
		else if (visited - 1 == index)
		{
			isBias = true;
		}
	}

	return isBias;
}
