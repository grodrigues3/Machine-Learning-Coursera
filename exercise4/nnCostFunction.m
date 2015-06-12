function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables

m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

X = [ones(m,1) X];
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%%Feedforward Part1
z2 = X*Theta1';
a2 = sigmoid(z2);
l = size(a2,1);
a2 = [ones(l,1), a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;

numLayers = size(h,2);
newY = zeros(m, numLayers);
for index = 1:m
	newY(index,y(index)) = 1;
end

cost = zeros(1,m);

for example = 1:m
	ex_cost = -newY(example,:).*log(h(example,:))-(1-newY(example,:)).*log(1-h(example,:));
	cost(example) = sum(ex_cost);
end

%regularized
regularPart1 = sum(sum(Theta1(:,2:end).^2));
regularPart2 = sum(sum(Theta2(:,2:end).^2));

totalCost = 1/m*sum(cost) + lambda/(2*m) * (regularPart1 + regularPart2);
J = totalCost;
% -------------------------------------------------------------

% =========================================================================



%Backprop
%===========================================================================


tri1 = zeros(size(Theta1));
tri2 = zeros(size(Theta2));
for i = 1:m
	a_1 = X(i,:); %1x401
	z2 = a_1*Theta1'; %1x401 * 401*25 = 1 x25
	a2 = sigmoid(z2);
	l = size(a2,1);
	a2 = [ones(l,1), a2];
	z3 = a2*Theta2'; %1x26 * 26x10 = 1x10
	a3 = sigmoid(z3); %1x10
	
	delta_3  = a3 - newY(i,:);%1x10
	delta_2 = delta_3*Theta2; %1x25
	delta_2 = delta_2(:,2:end).*sigmoidGradient(z2);
	
	tri1 = tri1 + delta_2'*a_1;
	tri2 = tri2 + delta_3'*a2;
end


Theta1_grad = 1/m * tri1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+ lambda/m*Theta1(:,2:end);
Theta2_grad = 1/m * tri2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+ lambda/m*Theta2(:,2:end);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

	