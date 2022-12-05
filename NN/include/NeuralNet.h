#pragma once

#include "Neuron.h"
#include <vector>

class NeuralNet
{
public:
	NeuralNet(const std::vector<size_t> &topology, bool doLeaky);

	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; };

	//code for drawing
	size_t getNumNodes(void) const { return m_numNodes; };
	size_t getNumConnections(void) const;
	size_t numLayers(void) const { return m_layers.size(); }
	size_t getLayerSize(size_t index) const { return m_layers[index].size(); }
	double getNodeWeight(size_t index) const;
	double getConnectionWeight(size_t i, size_t j, size_t k) const { return m_layers[i][j].getOutputWeight(k); }
	bool isBiasNodeIndex(size_t index) const;

private:
	std::vector<Neuron::Layer> m_layers;

	size_t m_numNodes;

	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};