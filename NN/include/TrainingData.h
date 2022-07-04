#pragma once

#include <string>
#include <vector>
#include <fstream>

class TrainingData
{
public:
	TrainingData(const std::string& filename);
	bool isEof(void) { return _trainingDataFile.eof(); }
	void getTopology(std::vector<size_t>& topology);

	size_t getNextInputs(std::vector<double>& inputVals);
	size_t getTargetOutputs(std::vector<double>& targetOutputVals);

private:
	std::ifstream _trainingDataFile;
};
