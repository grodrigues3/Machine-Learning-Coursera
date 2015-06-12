function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
t1 = theta(1);
t2 = theta(2);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	h = X*[t1;t2]; %gives predicted values for each training set [m x 1] matrix
	%theta = theta - alpha*dJ(theta)/d(theta)
	% dJ/Dtheta = (h - y) * x_i
	temp1 = t1 - alpha/m * sum((h-y));
	temp2 = t2 - alpha/m * sum((h-y).*X(:,2));
	t1 = temp1;
	t2 = temp2;
    % ============================================================

    % Save the cost J in every iteration    
    
	J_history(iter) = computeCost(X, y, theta);
	
	
end
theta = [t1;t2];
end
