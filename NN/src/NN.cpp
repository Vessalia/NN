#include <iostream>
#define SDL_MAIN_HANDLED

#include <SDL.h>
#include "NeuralNet.h"
#include "TrainingData.h"
#include <cstdlib>
#include <assert.h>
#include <vector>
#include <algorithm>

#define DATA_SET_SIZE 10000
#define LEAKY true

void createTrainingFile()
{
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    std::cout << "Current seed: " << seed << std::endl;
    std::ofstream file;
    file.open("assets/trainingData.txt");
    file << "topology: 2 4 1\n";
    for (size_t i = 0; i < DATA_SET_SIZE; ++i)
    {
        int firstInput = rand() % 2;
        int secondInput = rand() % 2;
        int out = 0;
        file << "in: " <<  firstInput << ".0 " << secondInput << ".0\n";

        if (firstInput ^ secondInput)
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

SDL_Window* gWindow = NULL;
SDL_Renderer* gRenderer = NULL;

const size_t SCREEN_WIDTH = 640;
const size_t SCREEN_HEIGHT = 640;

bool initSDL()
{
    bool success = true;

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        success = false;
    }
    else
    {
        gWindow = SDL_CreateWindow("Neural Net", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
        if (gWindow == NULL)
        {
            printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
            success = false;
        }
        else
        {
            gRenderer = SDL_CreateRenderer(gWindow, -1, SDL_RENDERER_ACCELERATED);
            if (gRenderer == NULL)
            {
                printf("Renderer could not be created! SDL Error: %s\n", SDL_GetError());
                success = false;
            }
            else
            {
                SDL_SetRenderDrawColor(gRenderer, 0xFF, 0xFF, 0xFF, 0xFF);
            }
        }
    }

    return success;
}

void closeSDL()
{
    SDL_DestroyRenderer(gRenderer);
    SDL_DestroyWindow(gWindow);
    gWindow = NULL;
    gRenderer = NULL;
    SDL_Quit();
}

void setNodeRects(std::vector<SDL_Rect>& nodes, const NeuralNet& net)
{
    int w = 10;
    int h = 10;
    for (size_t i = 0; i < net.numLayers(); ++i)
    {
        size_t layerSize = net.getLayerSize(i);
        for (size_t j = 0; j < layerSize; ++j)
        {
            SDL_Rect rect = { (int)((i + 0.5) * (double)SCREEN_WIDTH / net.numLayers()), (int)((j + 0.5) * (double)SCREEN_HEIGHT / layerSize), w, h };
            nodes.push_back(rect);
        }
    }
}

double clamp(double num, double low, double high)
{
    return num < high ? (num > low ? num : low) : high;
}

double lerp(double start, double end, double t)
{
    return start + t * (end - start);
}

unsigned int lerpRed(double weight)
{
    double clamped = clamp(weight, 0, 1);
    return (unsigned int)lerp(0x00, 0xFF, 1 - clamped);
}

unsigned int lerpGreen(double weight)
{
    double clamped = clamp(weight, 0, 1);
    return (unsigned int)lerp(0x00, 0xFF, clamped);
}

unsigned int lerpRedConnection(double weight)
{
    double clamped = (clamp(weight, -1, 1) + 1) / 2;
    return (unsigned int)lerp(0x00, 0xFF, 1 - clamped);
}

unsigned int lerpGreenConnection(double weight)
{
    double clamped = (clamp(weight, -1, 1) + 1) / 2;
    return (unsigned int)lerp(0x00, 0xFF, clamped);
}

void setNodeWeights(std::vector<unsigned int>& nodeWeights, const NeuralNet& net)
{
    for (size_t i = 0; i < net.getNumNodes(); ++i)
    {
        if (net.isBiasNodeIndex(i))
        {
            nodeWeights.push_back(0x00);
            nodeWeights.push_back(0x00);
            nodeWeights.push_back(0xFF);
        }
        else
        {
            double weight = net.getNodeWeight(i);

            nodeWeights.push_back(lerpRed(weight));
            nodeWeights.push_back(lerpGreen(weight));
            nodeWeights.push_back(0x00);
        }
    }
}

void setConnectionWeights(std::vector<unsigned int>& connectionWeights, const NeuralNet& net)
{
    for (size_t i = 0; i < net.numLayers(); ++i)
    {
        size_t layerSize = net.getLayerSize(i);
        for (size_t j = 0; j < layerSize; ++j)
        {
            if (i < net.numLayers() - 1)
            {
                size_t nextLayerSize = net.getLayerSize(i + 1);
                for (size_t k = 0; k < nextLayerSize - 1; ++k)
                {
                    double weight = net.getConnectionWeight(i, j, k);

                    connectionWeights.push_back(lerpRedConnection(weight));
                    connectionWeights.push_back(lerpGreenConnection(weight));
                }
            }
        }
    }
}

void drawNet(const NeuralNet& net)
{
    std::vector<SDL_Rect> nodes;
    std::vector<unsigned int> nodeWeights;
    std::vector<unsigned int> connectionWeights;

    nodes.reserve(net.getNumNodes());
    nodeWeights.reserve(3 * net.getNumNodes());
    connectionWeights.reserve(2 * net.getNumConnections());

    setNodeRects(nodes, net);
    setNodeWeights(nodeWeights, net);
    setConnectionWeights(connectionWeights, net);

    if (!initSDL())
    {
        printf("Failed to initialize!\n");
    }
    SDL_UpdateWindowSurface(gWindow);
    SDL_Event e; 
    bool quit = false; 
    while (!quit) 
    { 
        while (SDL_PollEvent(&e)) 
        { 
            if (e.type == SDL_QUIT) quit = true; 
        }
        SDL_SetRenderDrawColor(gRenderer, 0xFF, 0xFF, 0xFF, 0xFF);
        SDL_RenderClear(gRenderer);

        size_t visited = 0;
        for (size_t i = 0; i < net.numLayers(); ++i)
        {
            size_t layerSize = net.getLayerSize(i);
            for (size_t j = 0; j < layerSize; ++j)
            {
                SDL_SetRenderDrawColor(gRenderer, nodeWeights[3 * (j + visited)], nodeWeights[3 * (j + visited) + 1], nodeWeights[3 * (j + visited) + 2], 0xFF);
                SDL_RenderFillRect(gRenderer, &nodes[j + visited]);
                if (i < net.numLayers() - 1)
                {
                    size_t nextLayerSize = net.getLayerSize(i + 1);
                    for (size_t k = 0; k < nextLayerSize - 1; ++k)
                    {
                        size_t connectionIndex = 2 * ((layerSize * i) + (nextLayerSize * j) + k);
                        size_t nextNodeIndex = visited + layerSize + k;
                        SDL_SetRenderDrawColor(gRenderer, connectionWeights[connectionIndex], connectionWeights[connectionIndex + 1], 0x00, 0xFF);
                        SDL_RenderDrawLine(gRenderer, nodes[j + visited].x, nodes[j + visited].y, nodes[nextNodeIndex].x, nodes[nextNodeIndex].y);
                    }
                }
            }

            visited += layerSize;
        }

        SDL_RenderPresent(gRenderer);
    }

    closeSDL();
}

int main()
{
    createTrainingFile();
    TrainingData trainingData("assets/trainingData.txt");

    std::vector<size_t> topology;
    trainingData.getTopology(topology);
    NeuralNet net(topology, LEAKY);

    std::vector<double> inputVals, targetVals, resultVals;
    size_t trainingPass = 0;

    while (!trainingData.isEof())
    {
        bool showValues = false;
        ++trainingPass;

        size_t inputSize = trainingData.getNextInputs(inputVals);
        if (inputSize != topology[0])
        {
            std::cout << "Input data is of wrong dimensions.\nInput size: " << inputSize << "\nRequired: " << topology[0] << std::endl;
            break;
        }
        if (!(trainingPass % ((int)std::ceil(DATA_SET_SIZE / 10.0))))
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

        trainingData.getTargetOutputs(targetVals);
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

    net.feedForward({ 1, 1 });
    net.getResults(resultVals);
    showVectorVals("Test:", resultVals);

    drawNet(net);

    std::cout << "Done!" << std::endl;
}
