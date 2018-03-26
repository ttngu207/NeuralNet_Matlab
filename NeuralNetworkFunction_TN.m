function [ output ] = NeuralNetworkFunction_TN( nn_layer,label,featureset,options )
% function [ output ] = NeuralNetworkFunction_TN( nn_layer,label,featureset,options )
%
% This routine employs neural network as a classification scheme, with
% user-defined network architecture
% by Thinh Nguyen (latest update: Oct 20, 2016)
%
% + nn_layer: Size of each layer (number of node), in a vector, starting with the
%  input layer, ending with the output layer
%  (e.g., nn_layer = [400,100,25,10] means 400 features as input layer, 2
%  hidden layers: i) 1st hidden layer with 100 units - ii) 2nd hidden layer
%  with 25 units, output layer with 10 unit -> 10 labels to classify to.
% + label: mx1 label vector (m samples) -> the y 
% + featureset: mxn featureset vector (n features) -> the X
% + options:
%       options.lambda -> regularization parameter lambda ( de s
%       options.trainPortion -> ratio of cross-validation splitting 
%       options.featureOmit -> index of feature to omit (default [])
%       options.maxIter -> maximum number of iterations (default: 100)
% + output:
%       output.CLASSIFIER -> theta parameters in each layer
%       output.accuracyTable -> [train_acc,test_acc;train_cost;test_cost] x #Lambda x #traiPortion
%       output.lambvec 
%       output.trainingSize
%
% Note: this script requires the following functions: 
% randInitializeWeights.m
% nnCostFunction.m (backpropagation)
% fmincg.m         (parameters optimization)
% feed_forward.m   (feed forward & make prediction)
% PrecisionRecallCalc.m
% sigmoid.m
% sigmoidGradient.m


%==========================================================================
%============================= CLASSIFICATION =============================
%==========================================================================


% ======================= Specification ================================
Xall = featureset;
yall = label;

try, LAMBDA = options.lambda; catch, LAMBDA = 0;end % Regularization parameter lambda
try, Nrep = options.repNum; catch, Nrep = 1;end     % Reshuffling train and test set
try, training_portion = options.trainPortion; catch, training_portion = 0.6;end     % Percentage of data used as training set
try, f_omit = options.featureOmit;
    if not(isempty(f_omit)), nn_layer(1) = nn_layer(1) - numel(f_omit); end
catch, f_omit = [];end  % Choosing features to omit
try, maxIteration = options.maxIter; catch, maxIteration = 100;end % Number of iteration

if isempty(training_portion); training_portion = [0.3,0.5,0.7,0.9];end
totalNumberOfRun = numel(training_portion) * numel(LAMBDA);

%% ====================== Create loops ==========================
Xall(:,f_omit) = [];
Nsubj = length(unique(yall));
[Nsamples,Nfeature] = size(Xall);
accuracyTable = zeros(length(LAMBDA),4,length(training_portion));
for kk = 1:length(training_portion)

%% DO Classification N repetition to get mean accuracy
for l = 1:length(LAMBDA)
    lambda = LAMBDA(l);
meantrain_accuracy = [];
meantest_accuracy = [];
mean_cost4train = [];
mean_cost4test = [];
for rep = 1: Nrep

%% ====================== PROCESS ==========================

% Cross-validation - Split up data ========================================
y = []; X = []; ytest = []; Xtest = [];
for subj = 1 : Nsubj
subjIndex = find(yall==subj);
y_tmp = yall(subjIndex);
X_tmp = Xall(subjIndex,:);

shuff = randperm(length(y_tmp));
y_tmp = y_tmp(shuff);
X_tmp = X_tmp(shuff,:);

% Get training set
y = [y;y_tmp(1:floor(training_portion(kk)*length(y_tmp)))];
X = [X;X_tmp(1:floor(training_portion(kk)*length(y_tmp)),:)];
% Get test set
ytest = [ytest;y_tmp(floor(training_portion(kk)*length(y_tmp))+1:end,:)];
Xtest = [Xtest;X_tmp(floor(training_portion(kk)*length(y_tmp))+1:end,:)];
end

shuff = randperm(length(y));y = y(shuff);X = X(shuff,:);
shuff = randperm(length(ytest));ytest = ytest(shuff);Xtest = Xtest(shuff,:);

traintestSize = [numel(y),numel(ytest)];

% Generate initial Neural Network Parameters (Theta) ======================
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta = [];
for k = 1:length(nn_layer)-1
temp = randInitializeWeights(nn_layer(k), nn_layer(k+1)); 
fprintf('Theta %d: %dx%d\n',k,size(temp,1),size(temp,2));
ini_Theta{k} = temp;
initial_Theta = [initial_Theta; temp(:)];
end

% Training Neural Network =================================================
fprintf('\nTraining Neural Network... \n')

options2 = optimset('MaxIter', maxIteration);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p,nn_layer, X, y, lambda);

% Cost function optimization
[ThetaRolled, cost] = fmincg(costFunction, initial_Theta, options2);

% Reroll trained parameters back into Theta ===============================
% Reroll ThetaRolled back into Theta
clear Theta
% Layer 1
Theta{1} = reshape(ThetaRolled(1:nn_layer(2) * ( nn_layer(1) + 1)), ...
                 nn_layer(2), (nn_layer(1) + 1));
previouslayer = numel(Theta{1});
% Subsequence layer
for k = 2:length(nn_layer)-1
    Theta{k} = reshape( ThetaRolled([1: nn_layer(k+1) * (nn_layer(k) + 1)]+previouslayer), nn_layer(k+1), (nn_layer(k) + 1) );
    previouslayer = previouslayer +  nn_layer(k+1) * (nn_layer(k) + 1);
end

% Implement Prediction =================

% Train Set
pred = feed_forward(Theta, X); 
[accTrain,trainPRF]  = PrecisionRecallCalc(y,pred);
costTrain = nnCostFunction(ThetaRolled,nn_layer, X, y, lambda);

% Test Set
predtest = feed_forward(Theta, Xtest); 
[accTest,testPRF]  = PrecisionRecallCalc(ytest,predtest);
costTest = nnCostFunction(ThetaRolled,nn_layer, Xtest, ytest, lambda);

meantrain_accuracy = [meantrain_accuracy,accTrain];
meantest_accuracy = [meantest_accuracy,accTest];
mean_cost4train = [mean_cost4train,costTrain];
mean_cost4test = [mean_cost4test,costTest];
end

accuracyTable(l,:,kk) = [ mean(meantrain_accuracy), mean(meantest_accuracy),mean(mean_cost4train),mean(mean_cost4test)];
end
end


fprintf('\nTraining Set -- Size: %d - lambda: %d - Accuracy: %.2f - Cost: %4.4e \n',traintestSize(1),lambda,accTrain,costTrain) ;
disp('    Label - Precision - Recall - F score');disp(trainPRF)
fprintf('\nTest Set --  Size: %d - lambda: %d - Accuracy: %.2f - Cost: %4.4e \n',traintestSize(2),lambda,accTest,costTest) ;
disp('    Label - Precision - Recall - F score');disp(testPRF)

output.CLASSIFIER = Theta;
output.accuracyTable = accuracyTable;
output.lambvec = LAMBDA;
output.trainingSize = floor(training_portion * Nsamples);


end

