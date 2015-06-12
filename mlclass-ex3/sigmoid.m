function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
numColumns = size(z)(2);

for index = 1:numColumns
	g(:,index) = 1./(1+exp(-1*z(:,index)));	
end


% =============================================================

end
