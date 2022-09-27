function accuracy = checkAccuracy(y,y_hat)
% Function calculates various statistics that quantify the pricing error of the
% neural network, namely: MSE, RMSE, MAE, MPE

% Syntax: 
%   -output = checkAccuracy(y,y_hat)

% Input:
%   -y: Real values
%   -y_hat: Predicted values [by neural network]

% Output:
%   -accuracy: Structure that entails MSE, RMSE, MAE, MPE

% Difference between y and y_hat
accuracy.Diff = y-y_hat;

% Mean-Squared-Error(MSE):
accuracy.MSE = mean(accuracy.Diff.^2);

% Root-Mean-Squared-Error(RMSE):
accuracy.RMSE = sqrt(accuracy.MSE);

% Mean-Absolute-Error(MAE):
accuracy.MAE = mean(abs(accuracy.Diff));

% Mean-Percent-Error(MPE):
accuracy.MPE = sqrt(accuracy.MSE)/mean(y);
end
