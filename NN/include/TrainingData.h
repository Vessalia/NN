#pragma once

#include <string>
#include <vector>
#include <fstream>

class TrainingData
{
public:
	TrainingData(const std::string& filename);

	~TrainingData();

	bool isEof(void) const { return m_trainingDataFile.eof(); }

	void getTopology(std::vector<size_t>& topology);
	size_t getNextInputs(std::vector<double>& inputVals);
	void getTargetOutputs(std::vector<double>& targetOutputVals);

private:
	std::ifstream m_trainingDataFile;
};
