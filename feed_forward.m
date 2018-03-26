function [p,a_ALL] = feed_forward(Theta, X,thres)
%function [p,a_ALL] = feed_forward(Theta, X)
%PREDICT Predict the label of an input given a trained neural network
%   outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%   Theta is a cell with all Theta1, Theta2, ... ThetaL
%   Output: 
%       p: prediction based on the largest p value in each class in each trial
%       a_ALL: values of all nodes in all layers with the given Theta, the
%       output of the feed forward process
%%%%%%%%%%%%%% NOTE %%%%%%%%%%%%%%%%%%%
%%%  X     : [nfeaturel x   mtrial]  %%
%%%  theta : [nfeature  x     1   ]  %%
%%%  Y     : [   1      x   mtrial]  %%
%%%                                  %%
%%%         Y = theta' * X           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Useful values
m = size(X, 1);
L = length(Theta);
try, decisionThreshold = thres; catch, decisionThreshold = 0.5; end
% Insert Dummy for Input Layer (a1)
a1 = X; a_ALL{1} = a1;
a1 = [ones(m, 1) a1]; 

% Looping forward through all hidden layers
a_previous = a1;
for k = 1:L-1
%fprintf('For layer %d, a(%d) is %dx%d - Theta(%d) is %dx%d\n',k+1,k, size(a_previous,1),size(a_previous,2),k,size(Theta{k},1),size(Theta{k},2));
% Compute values for Hidden Layer (k+1) using Theta_k and a_k
a_next = sigmoid( a_previous * Theta{k}'  ); a_ALL{end+1} = a_next;
% Insert Dummy for Hidden Layer (k+1) 
a_previous = [ones(m,1) a_next];
end

% Compute values for Output Layer (a3) using Theta2 and a2
aL = sigmoid( a_previous * Theta{end}'  ); a_ALL{end+1} = aL;

if size(aL,2)>1
% Make prediction oneVSall
[~,p] = max(aL,[],2); 
else
    p = zeros(m,1);
    tmp = find(aL > decisionThreshold);
    p(tmp) = 1;
end
% =========================================================================


end
