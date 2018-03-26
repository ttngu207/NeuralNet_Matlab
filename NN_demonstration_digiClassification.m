close all, clear all, clc

%% EXAMPLE - hand-written digit classification - by Thinh Nguyen
% This example script demonstrates how to use the
% "NeuralNetworkFunction_TN.m" routine to perform multi-class
% classification using a fully-connected neural network architecture

%% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

datadir = 'C:\Users\hongh\OneDrive\ADAsolution\matlab_functions\TN_Functions\MachineLearning_TNfunctions\Neural_Network_2\';
data = load([datadir,'ex4data1.mat']);
X = data.X;
y = data.y;

%% Randomly select 100 data points to display
m = size(X, 1);
sel = randperm(size(X, 1));
sel = sel(1:100);

displayImageData(X(sel, :));

%% Implement "NeuralNetworkFunction_TN.m"

nn_layer = [400,100,25,10];
options.lambda = [0,0.5,1];
options.repNum = 2;
options.trainPortion = .75;
options.featureOmit = [];
options.maxIter = 100;

output = NeuralNetworkFunction_TN( nn_layer,y,X,options );


















