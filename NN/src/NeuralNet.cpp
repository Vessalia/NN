#include "NeuralNet.h"
#include <assert.h>

double NeuralNet::_recentAverageSmoothingFactor = 100.0;

NeuralNet::NeuralNet(const std::vector<size_t> &topology)
	: _error(0.0)
	, _recentAverageError(0.0)
{
	size_t numLayers = topology.size();
	for (size_t layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		_layers.push_back(Layer());
		size_t numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (size_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}

		_layers.back().back().setOutputVal(1.0);
	}
}

void NeuralNet::feedForward(const std::vector<double>& inputVals)
{
	assert(inputVals.size() == _layers[0].size() - 1);

	for (size_t i = 0; i < inputVals.size(); ++i)
	{
		_layers[0][i].setOutputVal(inputVals[i]);
	}

	for (size_t layerNum = 1; layerNum < _layers.size(); ++layerNum)
	{
		Layer &prevLayer = _layers[layerNum - 1];
		for (size_t n = 0; n < _layers[layerNum].size() - 1; ++n)
		{
			_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void NeuralNet::backProp(const std::vector<double>& targetVals)
{
	Layer& outputLayer = _layers.back();
	_error = 0.0;

	for (size_t n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		_error += delta * delta;
	}

	_error /= outputLayer.size() - 1;
	_error = sqrt(_error);

	_recentAverageError = (_recentAverageError * _recentAverageSmoothingFactor + _error) / (_recentAverageSmoothingFactor + 1.0);

	for (size_t n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	for (size_t layerNum = _layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer& hiddenLayer = _layers[layerNum];
		Layer& nextLayer = _layers[layerNum + 1];

		for (size_t n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	for (size_t layerNum = _layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer& layer = _layers[layerNum];
		Layer& prevLayer = _layers[layerNum - 1];

		for (size_t n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void NeuralNet::getResults(std::vector<double>& resultVals) const
{
	resultVals.clear();

	Layer outputNeurons = _layers.back();

	for (size_t n = 0; n < outputNeurons.size() - 1; ++n)
	{
		resultVals.push_back(outputNeurons[n].getOutputVal());
	}
}

void NeuralNet::mutate()
{
}
