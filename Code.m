%% Bachelorthesis
%% Deep Learning and Option Pricing

%{
Replication of Culkin, R., & Das, S. R. (2017).
Machine learning in finance: The Case of Deep Learning for Option Pricing.
Journal of Investment Management, 15(4), 92-100.
%}

% Python:
pyenv("Version",'3.9')

% Black-Scholes formula
% function: black_scholes()
% -> see function documentation

% Generate 300,000 call option prices
Moneyness = [0.7 1.2]; % Moneyness
T = [1/252 3]; % Maturity
q = [0 0.03]; % Dividend rate
r_f = [0.01,0.03]; % Risk-free rate
sigma = [0.05,0.9]; % Implied vola

num_options = 300000;

% Sampling the parameter combinations:
% rng('default');
Quasi = sobolset(5,'Skip',1024);
inputs = Quasi(1:num_options,:);

% Column numbers for the parameters:
iMoneyness = 1; iT = 2; iq = 3; ir_f = 4; isigma = 5;

inputs(:,iMoneyness)= inputs(:,iMoneyness)*(Moneyness(2)-Moneyness(1))+Moneyness(1); 
%inputs(:,iK)= inputs(:,iK)*(K(2)-K(1))+K(1); 
inputs(:,iT)= inputs(:,iT)*(T(2)-T(1))+T(1); 
inputs(:,iq)= inputs(:,iq)*(q(2)-q(1))+q(1); 
inputs(:,ir_f)= inputs(:,ir_f)*(r_f(2)-r_f(1))+r_f(1); 
inputs(:,isigma)= inputs(:,isigma)*(sigma(2)-sigma(1))+sigma(1); 

column_names = {'Moneyness';'Maturity';'Div_Yield';'Risk_Free';'Sigma'; 'Call'};

black_scholes_data = array2table([inputs,zeros(length(inputs),1)],"VariableNames",column_names);

% Open question: Does that generate to many quotes of the same "specification" ?!!!!!

i = 0;
while i < num_options
    
    i = i+1
    black_scholes_data.Call(i) = black_scholes(black_scholes_data.Moneyness(i), ...
        1, black_scholes_data.Risk_Free(i), ...
        black_scholes_data.Div_Yield(i), black_scholes_data.Maturity(i), ...
        black_scholes_data.Sigma(i), 'C');

end

% Overview of the data:
head(black_scholes_data)

% Plot of generated call options:
scatter3(black_scholes_data.Moneyness,black_scholes_data.Maturity, ...
    black_scholes_data.Call,MarkerEdgeColor="red", MarkerFaceColor="blue")

% Histogram of prices:
histogram(black_scholes_data.Call)

% Number of observations left:
height(black_scholes_data); 

%% Neural Network

% Split in training (~0.8) and test (~0.2) data // Validation

% Creating a partition object:
rng('default');
% 20% of the data being hold out for testing
Partition = cvpartition(height(black_scholes_data),'HoldOut', 0.2);
TestIndex = test(Partition);
TrainIndex = training(Partition);

% Splitting into train and test:
Train = black_scholes_data(TrainIndex,:);
Test = black_scholes_data(TestIndex,:);

% Dividing X and Y of Train and Test Data:  
Train_X = [Train.Moneyness,Train.Risk_Free,Train.Div_Yield,Train.Maturity,Train.Sigma];
Train_Y = Train.Call;

Test_X = [Test.Moneyness,Test.Risk_Free,Test.Div_Yield,Test.Maturity,Test.Sigma];
Test_Y = Test.Call;

% Validation:
NumTrain = length(Train_X);
% 10% (of training data) is validation data:
% rng('default')
ValidationIndex = randperm(NumTrain,floor(NumTrain*0.1));
Validation_X = Train_X(ValidationIndex,:);
Train_X(ValidationIndex,:) = [];
Validation_Y = Train_Y(ValidationIndex,:);
Train_Y(ValidationIndex,:) = [];

% Setting up neural network:

% Number of inputs:
inputFeatures = 5;

% Layers:
layers = [
    featureInputLayer(inputFeatures,Normalization='zscore')
    fullyConnectedLayer(100)
    leakyReluLayer
    dropoutLayer(0.25)
    fullyConnectedLayer(100)
    eluLayer
    dropoutLayer(0.25)
    fullyConnectedLayer(100)
    reluLayer
    dropoutLayer(0.25)
    fullyConnectedLayer(100)
    eluLayer
    dropoutLayer(0.25)
    fullyConnectedLayer(1)
    regressionLayer];

plot(layerGraph(layers));

% Training options
epochs = 30;
batchSize = 265; % 256

options = trainingOptions("adam", ...
    MaxEpochs=epochs, ...
    Shuffle='every-epoch', ...
    MiniBatchSize=batchSize, ...
    Plots="training-progress",Verbose=1, ...
    L2Regularization=1.9e-7, ...
    InitialLearnRate=8.8e-3, ...
    LearnRateSchedule='piecewise', ...
    LearnRateDropPeriod=4, ...
    LearnRateDropFactor=0.128,...
    ValidationData={Validation_X,Validation_Y},...
    ValidationFrequency=50);

% Training:
[net, info] = trainNetwork(Train_X,Train_Y,layers,options)

% Prediction:
Pred_Y = predict(net,Test_X);

% Checking accuracy of predictions:
accuracy = checkAccuracy(Test_Y,Pred_Y); % -> see function file for further information/description

% R-Squared:
model1 = fitlm(Test_Y,Pred_Y);
model1.Rsquared.Ordinary
model1.Rsquared.Adjusted

% Histogram of errors:
figure
histogram(accuracy.Diff)
xlabel('Error Distribution')
ylabel('Counts')

% Plot of predicted vs actual option prices:
figure
plot(Test_Y,Pred_Y,'.',[min(Test_Y),max(Test_Y)],[min(Test_Y),max(Test_Y)],'r')
xlabel('Actual scaled price')
ylabel('Predicted Price')
title('Predictions on Test Data')

%% Monte Carlo Simulation

% Number of simulations:
N = 1000;

% Initialization of price storage vector:
Price = nan(num_options,1);

% rng('default')

Start = tic;
for j = 1:num_options
    % Parameters of option j:
    s = black_scholes_data.Moneyness(j);
    k = 1;
    r = black_scholes_data.Risk_Free(j);
    vola = black_scholes_data.Sigma(j);

    % Parameters for GBM:
    log_s = log(s);
    dt = black_scholes_data.Maturity(j);
    udt = (r-0.5*vola^2)*dt;
    voladt = vola*sqrt(dt);

    % Vectorized Monte Carlo simulation for option j:
    log_s_t = log_s+udt+voladt*randn(N,1);
    s_t = exp(log_s_t);
    C_t = max(0,s_t-k);

    % Value of option j:
    Price(j) = mean(C_t)*exp(-r*dt);
    j
end

End = toc(Start);

% Visualization
figure
plot(black_scholes_data.Call,Price, '.',[min(black_scholes_data.Call) ...
    max(black_scholes_data.Call)],[min(black_scholes_data.Call),...
    max(black_scholes_data.Call)],'r');
xlabel('Black-Scholes Prices');
ylabel('Monte Carlo Prices');
title('Comparison of Black-Scholes vs. Monte Carlo');
axis([min(black_scholes_data.Call) max(black_scholes_data.Call) min(Price) max(Price)]);

% R-Squared:
model2 = fitlm(black_scholes_data.Call,Price);
model2.Rsquared.Ordinary
model2.Rsquared.Adjusted

%% 2. Monte Carlo Simulation

% Price vector:
Price2 = zeros(num_options,1);

% Number of paths:
trials = 1000;

Start = tic;

% Simulation
for i = 1:num_options
    S0 = black_scholes_data.Moneyness(i); % Standardized price of underlying asset
    Sigma = black_scholes_data.Sigma(i); % Volatility of underlying asset
    Strike = 1; % Standardized Strike price (= 1)
    OptSpec = 'call'; % Call option
    Settle = '1-Jan-2022'; % Fixed Settlement date of option as reference point
    Maturity = datestr(daysadd(Settle, ceil(black_scholes_data.Maturity(i)*252))); % Maturity date of option
    r = black_scholes_data.Risk_Free(i); % Risk-free rate (annual, continuous compounding)
    Compounding = -1; % Continuous compounding
    Basis = 0; % day count convention
    T = yearfrac(Settle, Maturity, Basis); % Time to expiration in years

    NTRIALS = trials; 
    NPERIODS = 1; 
    dt = T/NPERIODS;
    OptionGBM = gbm(r, Sigma, 'StartState', S0);
    [Paths, Times, Z] = simBySolution(OptionGBM, NPERIODS, ...
    'NTRIALS',NTRIALS, 'DeltaTime',dt,'Antithetic',true);
    
    RateSpec = intenvset('ValuationDate', Settle, 'StartDates', Settle, ...
               'EndDates', Maturity, 'Rate', r, 'Compounding', Compounding, ...
               'Basis', Basis);
    
    SimulatedPrices = squeeze(Paths);
    OptPrice = optpricebysim(RateSpec, SimulatedPrices, Times, OptSpec, ...
               Strike, T, 'AmericanOpt', 0); % European Option
    Price2(i) = OptPrice;
    i

end

End = toc(Start);

% Visualization
figure
plot(black_scholes_data.Call,Price2, '.',[min(black_scholes_data.Call) ...
    max(black_scholes_data.Call)],[min(black_scholes_data.Call),...
    max(black_scholes_data.Call)],'r');
xlabel('Black-Scholes Prices');
ylabel('Monte Carlo Prices');
title('Comparison of Black-Scholes vs. Monte Carlo');
axis([min(black_scholes_data.Call) max(black_scholes_data.Call) min(Price2) max(Price2)]);


% R-Squared:
model3 = fitlm(black_scholes_data.Call,Price2);
model3.Rsquared.Ordinary
model3.Rsquared.Adjusted


