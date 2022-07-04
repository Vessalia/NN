#include <iostream>

#include "NeuralNet.h"
#include "Agent.h"
#include "TrainingData.h"
#include <cstdlib>
#include <assert.h>
#include <vector>
#include <algorithm>

#define DATA_SET_SIZE 10000

void createTrainingFile()
{
    std::ofstream file;
    file.open("assets/trainingData.txt");
    file << "topology: 2 4 1\n";
    for (size_t i = 0; i < DATA_SET_SIZE; ++i)
    {
        int firstIn = rand() % 2;
        int secondIn = rand() % 2;
        int out = 0;
        file << "in: " <<  firstIn << ".0 " << secondIn << ".0\n";

        if ((firstIn && !secondIn) || (!firstIn && secondIn))
        {
            out = 1;
        }
        file << "out: " << out << ".0";
        if (i < DATA_SET_SIZE - 1)
        {
            file << "\n";
        }
    }
    file.close();
}

void showVectorVals(const std::string& label, const std::vector<double>& vals)
{
    std::cout << label << " ";
    for (size_t i = 0; i < vals.size(); ++i)
    {
        std::cout << vals[i] << " ";
    }

    std::cout << std::endl;
}

int numDigits(int value)
{
    int digits = 1;
    while (value > 9)
    {
        value /= 10;
        digits++;
    }

    return digits;
}

int main()
{
    createTrainingFile();
    TrainingData trainData("assets/trainingData.txt");

    std::vector<size_t> topology;
    trainData.getTopology(topology);
    NeuralNet net(topology);

    std::vector<double> inputVals, targetVals, resultVals;
    size_t trainingPass = 0;

    while (!trainData.isEof())
    {
        bool showValues = false;
        ++trainingPass;

        if (trainData.getNextInputs(inputVals) != topology[0])
        {
            break;
        }
        if (!(trainingPass % (int)(DATA_SET_SIZE / (int)std::pow(10, std::max(numDigits(DATA_SET_SIZE) - 5, 0)))))
        {
            showValues = true;
        }

        if (showValues)
        {
            std::cout << "Pass: " << trainingPass << std::endl;
            showVectorVals("Inputs:", inputVals);
        }

        net.feedForward(inputVals);

        net.getResults(resultVals);
        if (showValues)
        {
            showVectorVals("Outputs:", resultVals);
        }

        trainData.getTargetOutputs(targetVals);
        if (showValues)
        {
            showVectorVals("Targets:", targetVals);
        }
        assert(targetVals.size() == topology.back());

        net.backProp(targetVals);

        if (showValues)
        {
            std::cout << "Net recent average error: " << net.getRecentAverageError() << std::endl;
        }
    }

    net.feedForward({ 1, 0 });
    net.getResults(resultVals);
    showVectorVals("Test:", resultVals);

    std::cout << "Done!" << std::endl;
}
