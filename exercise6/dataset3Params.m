function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_List = [.01, .03, .1, .3, 1, 3, 10, 30];
Sigma_List = [.01, .03, .1, .3, 1, 3, 10, 30];
currentMin = 10^6; %some arbitrarily large number
counter = 0
%for i = 1:length(C_List)
%	for j = 1:length(Sigma_List)
%		counter+=1
%		current_C = C_List(i);
%		current_sigma = Sigma_List(j);
%		model= svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
%		predictions = svmPredict(model, Xval);
%		newError = mean(double(predictions ~= yval));
%		if newError < currentMin;
%			C = current_C
%			sigma = current_sigma
%			newError
%			currentMin = newError;
%		end
%	end
%end


C = 1
sigma = .1

%save best_parameters.m
% =========================================================================

end
