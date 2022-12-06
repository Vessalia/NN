#include <iostream>
#define SDL_MAIN_HANDLED

#include <SDL.h>
#include "NeuralNet.h"
#include "TrainingData.h"
#include <cstdlib>
#include <assert.h>
#include <vector>
#include <string.h>
#include <algorithm>

#define DATA_SET_SIZE 10000
#define DISPLAY_FACTOR 10.0
#define USE_SAVED_DATA true
#define LEAKY true
#define OPERATOR XOR
#define TOPOLOGY "2 4 8 6 5 1"
// draw dimensions per neuron
#define NEURON_WIDTH 30
#define NEURON_HEIGHT 30

enum boolOperator
{
    AND, NAND, OR, NOR, XOR, NXOR
};

void createTrainingFile()
{
    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);
    std::cout << "Current seed: " << seed << std::endl;
    std::ofstream file;
    file.open("assets/trainingData.txt");
    file << "topology: " << TOPOLOGY <<"\n";
    for (size_t i = 0; i < DATA_SET_SIZE; ++i)
    {
        int firstInput = rand() % 2;
        int secondInput = rand() % 2;
        int out = 0;
        file << "in: " <<  firstInput << ".0 " << secondInput << ".0\n";

        switch (OPERATOR)
        {
            case(AND): if (firstInput & secondInput) out = 1; break;
            case(NAND): if (!(firstInput & secondInput)) out = 1; break;
            case(OR): if (firstInput | secondInput) out = 1; break;
            case(NOR): if (!(firstInput | secondInput)) out = 1; break;
            case(XOR): if (firstInput ^ secondInput) out = 1; break;
            case(NXOR): if (!(firstInput ^ secondInput)) out = 1; break;
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

SDL_Window* _window = NULL;
SDL_Renderer* _renderer = NULL;

const size_t SCREEN_WIDTH = 640;
const size_t SCREEN_HEIGHT = 640;

/*This piece of code was originally from Lazy Foo' Productions
(http://lazyfoo.net/)*/
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
        _window = SDL_CreateWindow("Neural Net", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
        if (_window == NULL)
        {
            printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
            success = false;
        }
        else
        {
            _renderer = SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED);
            if (_renderer == NULL)
            {
                printf("Renderer could not be created! SDL Error: %s\n", SDL_GetError());
                success = false;
            }
            else
            {
                SDL_SetRenderDrawColor(_renderer, 0xFF, 0xFF, 0xFF, 0xFF);
            }
        }
    }

    return success;
}

void closeSDL()
{
    SDL_DestroyRenderer(_renderer);
    _renderer = NULL;

    SDL_DestroyWindow(_window);
    _window = NULL;

    SDL_Quit();
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
    double clamped = clamp(weight, -1, 1);
    if (weight < 0)
    {
        return 0xFF;
    }

    return (unsigned int)lerp(0xFF, 0x00, clamped);
}

unsigned int lerpGreen(double weight)
{
    double clamped = clamp(weight, -1, 1);
    if (weight >= 0)
    {
        return 0xFF;
    }

    return (unsigned int)lerp(0xFF, 0x00, -clamped);
}

void getNodeRects(std::vector<SDL_Rect>& nodes, const NeuralNet& net)
{
    int w = NEURON_WIDTH;
    int h = NEURON_HEIGHT;
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

void getNodeOutputs(std::vector<unsigned int>& nodeWeights, const NeuralNet& net)
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

void getConnectionWeights(std::vector<unsigned int>& connectionWeights, const NeuralNet& net)
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

                    connectionWeights.push_back(lerpRed(weight));
                    connectionWeights.push_back(lerpGreen(weight));
                }
            }
        }
    }
}

void buildDraw(std::vector<SDL_Rect>& nodes, std::vector<unsigned int>& nodeOutputs, std::vector<unsigned int>& connectionWeights, const NeuralNet& net)
{
    nodes.clear();
    nodeOutputs.clear();
    connectionWeights.clear();

    nodes.reserve(net.getNumNodes());
    nodeOutputs.reserve(3 * net.getNumNodes());
    connectionWeights.reserve(2 * net.getNumConnections());

    getNodeRects(nodes, net);
    getNodeOutputs(nodeOutputs, net);
    getConnectionWeights(connectionWeights, net);
}

const std::vector<std::vector<double>> testInputs = { { 1.0, 1.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 0.0, 0.0 } };
void drawNet(NeuralNet& net, std::vector<double>& resultVals)
{
    std::vector<SDL_Rect> nodes;
    std::vector<unsigned int> nodeOutputs;
    std::vector<unsigned int> connectionWeights;

    buildDraw(nodes, nodeOutputs, connectionWeights, net);

    if (!initSDL())
    {
        printf("Failed to initialize!\n");
    }
    SDL_UpdateWindowSurface(_window);
    SDL_Event e; 
    bool quit = false; 
    while (!quit) 
    { 
        while (SDL_PollEvent(&e)) 
        { 
            if (e.type == SDL_QUIT) quit = true; 
            if (e.type == SDL_KEYDOWN)
            {
                size_t index = 0;
                switch (e.key.keysym.sym)
                {
                case SDLK_UP: index = 0; net.feedForward(testInputs[index]); break;
                case SDLK_LEFT: index = 1; net.feedForward(testInputs[index]); break;
                case SDLK_RIGHT: index = 2; net.feedForward(testInputs[index]); break;
                case SDLK_DOWN: index = 3; net.feedForward(testInputs[index]); break;
                }

                net.getResults(resultVals);
                showVectorVals("Inputs:", testInputs[index]);
                showVectorVals("Outputs:", resultVals);
                buildDraw(nodes, nodeOutputs, connectionWeights, net);
            }
        }
        SDL_SetRenderDrawColor(_renderer, 0xA5, 0xA5, 0xA5, 0xFF);
        SDL_RenderClear(_renderer);

        size_t visited = 0;
        for (size_t i = 0; i < net.numLayers(); ++i)
        {
            size_t layerSize = net.getLayerSize(i);
            for (size_t j = 0; j < layerSize; ++j)
            {
                SDL_SetRenderDrawColor(_renderer, nodeOutputs[3 * (j + visited)], nodeOutputs[3 * (j + visited) + 1], nodeOutputs[3 * (j + visited) + 2], 0xFF);
                SDL_RenderFillRect(_renderer, &nodes[j + visited]);
                if (i < net.numLayers() - 1)
                {
                    size_t nextLayerSize = net.getLayerSize(i + 1);
                    for (size_t k = 0; k < nextLayerSize - 1; ++k)
                    {
                        size_t connectionIndex = 2 * ((layerSize * i) + (nextLayerSize * j) + k);
                        size_t nextNodeIndex = visited + layerSize + k;
                        SDL_SetRenderDrawColor(_renderer, connectionWeights[connectionIndex], connectionWeights[connectionIndex + 1], 0x00, 0xFF);
                        SDL_RenderDrawLine(_renderer, nodes[j + visited].x + nodes[j + visited].w, nodes[j + visited].y + nodes[j + visited].h / 2,
                                                      nodes[nextNodeIndex].x, nodes[nextNodeIndex].y + nodes[nextNodeIndex].h / 2);
                    }
                }
            }

            visited += layerSize;
        }

        SDL_RenderPresent(_renderer);
    }

    closeSDL();
}

std::string filePath;
int main()
{
    if (USE_SAVED_DATA)
    {
        filePath = "assets/trainingDataSavedData.txt";
    }
    else
    {
        filePath = "assets/trainingData.txt";
    }
    createTrainingFile();
    TrainingData trainingData(filePath);

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
        if (!(trainingPass % ((int)std::ceil(DATA_SET_SIZE / DISPLAY_FACTOR))))
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

    net.feedForward(testInputs[0]);
    net.getResults(resultVals);
    showVectorVals("Inputs:", testInputs[0]);
    showVectorVals("Test:", resultVals);

    drawNet(net, resultVals);

    std::cout << "Done!" << std::endl;
}
