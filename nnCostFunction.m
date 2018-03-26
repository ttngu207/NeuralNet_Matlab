function [J, grad] = nnCostFunction(nn_params, nn_layer, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, layer, X, y, lambda) computes the cost and 
%   gradient of the neural network, given parameter Theta (rolled in nn_params) 
%   
%   Input:  
%       nn_params: unrolled version of Theta (a vector, not matrix)
%       layer: Size of each layer (number of node), in a vector, starting with the
%       input layer, ending with the output layer
%       X: training data (m training sample  x  n features)
%       y: labels of X   (m lables x 1)
%       lambda: regularization parameter
%   Output:
%       J: cost
%       grad: gradient of the Theta

% Inspect input arguments
try lambda = lambda; catch, lambda = 0; end

% Reroll nn_para into Theta
clear Theta
% Layer 1
Theta{1} = reshape(nn_params(1:nn_layer(2) * ( nn_layer(1) + 1)), ...
                 nn_layer(2), (nn_layer(1) + 1));
%    fprintf('Size Theta(%d): %d x %d\n', 1, size(Theta{1},1),size(Theta{1},2)) 
previouslayer = numel(Theta{1});
% Subsequence layer
for k = 2:length(nn_layer)-1
%     Theta{k} = reshape(nn_params( (1 + ( nn_layer(k) * (nn_layer(k-1) + 1) )) : ( nn_layer(k) * (nn_layer(k-1) + 1) )+( nn_layer(k+1) * (nn_layer(k) + 1) ) ), ...
%                  nn_layer(k+1), (nn_layer(k) + 1) ) ;
    Theta{k} = reshape( nn_params([1: nn_layer(k+1) * (nn_layer(k) + 1)]+previouslayer), nn_layer(k+1), (nn_layer(k) + 1) );
    previouslayer = previouslayer +  nn_layer(k+1) * (nn_layer(k) + 1);
%    fprintf('Size Theta(%d): %d x %d\n', k, size(Theta{k},1),size(Theta{k},2))
end
% Setup some useful variables
L = length(nn_layer);
m = size(X, 1);
Y = zeros( m, length(unique(y)) );
for k = 1:m, Y(k,y(k)) = 1; end

% Part 1: Feedforward the neural network and return the cost in the
%         variable J.

% Cost function without regularization
[~,a_output] = feed_forward(Theta,X); a_output = a_output{end};
J = (-1*Y).*(log(a_output)) -  (1 - Y).*(log(1-a_output));
J = sum(J,2); J = sum(J); J = J/m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

%fprintf('Feed Forward and Back Propagation...\n') 

Theta_grad = feedforward_backprob(Theta,X,Y,L); 

% Theta_grad{1},Theta_grad{2},Theta_grad{3},Theta_grad{4}
% Average over m trainning sets
for k = 1:length(Theta_grad)
Theta_grad{k} = Theta_grad{k}/m;
end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Regularization of cost
J_reg = 0;
for k = 1:length(Theta)
    tmpTheta = Theta{k};
    J_reg = J_reg + (sum( sum(tmpTheta(:,2:end).^2) ));
end   
J = J + (lambda/(2*m))*J_reg;

% Regularization of gradient
for k = 1:length(Theta)
    tempgrad = Theta_grad{k}; temptheta = Theta{k};
    tempgrad(:,2:end) = tempgrad(:,2:end) + (lambda/(m))*temptheta(:,2:end);
    Theta_grad{k} = tempgrad ;
end

% -------------------------------------------------------------

% =========================================================================
% Unroll gradients
grad = [];
for k = 1:length(Theta)
tempgrad = Theta_grad{k};
grad = [grad ; tempgrad(:)];
end


end

function Theta_grad = feedforward_backprob(Theta,X,Y,L)
%%%%
% Initialize theta gradient structure
    for k = 1:length(Theta)
    Theta_grad{k} = zeros(size(Theta{k}));
    end
    %%%% Feed forward
    [~,a_ALL] = feed_forward(Theta,X);
    %%%% Back propagation
    % Compute delta_L
    deltaL = a_ALL{end} - Y;
    
    % Accumulate gradient for theta(last)
    a_Lminus = a_ALL{end-1}; a_Lminus = [ ones(size(a_Lminus,1), 1), a_Lminus];
    Theta_grad{end} = Theta_grad{end} + deltaL' * a_Lminus;
    %     fprintf('BckProp layer(%d) - GradTheta(%d) - Training: %d\n',L,L-1,t)
    % Looping backward through all hidden layers
    delta_kplus = deltaL;
    for k = L-1:-1:2
        %fprintf('BckProp layer(%d) - GradTheta(%d) \n',k,k-1)
        % Compute delta(k)
        a_k = a_ALL{k}; a_k = [ ones(size(a_k,1), 1), a_k];
        a_kminus = a_ALL{k-1}; a_kminus = [ ones(size(a_kminus,1), 1), a_kminus];
        delta_k = delta_kplus * Theta{k} .* (a_k .* (ones(size(a_k))-a_k)); delta_k = delta_k(:,2:end);
        % Accumulate gradient for theta(k-1) and update
        Theta_grad{k-1} = Theta_grad{k-1} + delta_k' * a_kminus;         
        delta_kplus = delta_k;
    end    
%%%%
end


