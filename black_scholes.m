function Price = black_scholes(S,K,r_f,q,T,sigma,oType)
% Function calculates the price of an option according to Black-Scholes

% Syntax: 
%   -output = black_scholes(S,K,r_f,q,T,sigma,oType)

% Input:
%   -S: stock price
%   -K: strike price
%   -r_f: risk-free rate
%   -q: dividend yield 
%   -T: remaining maturity
%   -sigma: implied volatility
%   -oType: Call-/Put option

% Output:
%   -Price: Price according to Black-Scholes formula

    d1 = (log(S/K)+(r_f-q+0.5*sigma^2)*T)/(sigma*sqrt(T));

    d2 = d1-sigma*sqrt(T);

    if oType == 'C'

        Price = S*exp(-q*T)*normcdf(d1)-K*exp(-r_f*T)*normcdf(d2);

    elseif  oType == 'P'

        Price = K*exp(-r_f*T)*normcdf(-d2)-S*exp(-q*T)*normcdf(-d1);

    else 

        disp('Please specify the option type as "C" or "P"')
    end
end