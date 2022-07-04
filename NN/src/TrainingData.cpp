#include "TrainingData.h"

#include <sstream>

TrainingData::TrainingData(const std::string& filename)
{
	_trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(std::vector<size_t>& topology)
{
	std::string line;
	std::string label;

	getline(_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		size_t n;
		ss >> n;
		topology.push_back(n);
	}
}

size_t TrainingData::getNextInputs(std::vector<double>& inputVals)
{
	inputVals.clear();

	std::string line;
	getline(_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

size_t TrainingData::getTargetOutputs(std::vector<double>& targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	getline(_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}
