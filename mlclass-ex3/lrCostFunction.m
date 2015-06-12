function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%size(X) %5000x401
%size(y) 500 x 1
%size(theta) 10 x 401
theta = theta';
h = sigmoid(X*theta);
size(h) %5000x10
[val, index]  = max(h'); %val, index are both a 1x5000
h_prob = val; % 5000x1
biny = y == index'; %5000x1
J = 1/m*( log(h_prob)*biny - log(1-h_prob)*(1-biny))

%% Gradient Part

dummy_y = repmat(1:10, 5000, 1);
for jj = 1:length(dummy_y)
	other_y(jj,:) = dummy_y(jj,:) == y(jj);
end
size(other_y)% %5000x10
grad = zeros(size(theta));
grad = 1/m * X'*(h-other_y);


% =============================================================


grad = grad(:);

end
