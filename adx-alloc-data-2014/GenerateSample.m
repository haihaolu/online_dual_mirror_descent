function [Q, t] = GenerateSample (A, T, rho, type_ad, type_prob, tau, type, samples)
% GenerateSample: generates a sample of impressions using the generative model given by the parameters. 
%
%   INPUT
%       A:int = number of advertisers
%       T:int = number of types
%       rho:double(1,A) = capacity to impression ratio of the advertisers
%	type_ad:bool(T,A) = type-ad graph incidence matrix. type_ad(t,a) = 1 if advertiser a is in type t
%       type_prob:double(1,T) = occurrence probability of each type
%       tau:double(1,A) = goodwill penalty paid by publisher if impression is assigned to an advertiser not
%                         interested in the type
%       type:struct(T) = structure specifying the mean and covariance of the lognormal distribution of the 
%                        advertisers within each type. the structure has two fields: 
%                           - type(t).mu:double(1,A(t)) = mean vector of the normal distribution.
%                           - type(t).SIGMA:double(A(t),A(t)) = covariance matrix of the normal distribution.    
%                        where A(t)=sum(type_ad(t,:)) is the number of ads in the type
%       samples:int = Number of samples to take.
%
%   OUTPUT
%       Q:double(samples,A) = sample quality data. Each row is a different sampled impression. 
%       t:int(samples,1) = sampled impression types.
%
% Copyright 2010 Google Inc. All Rights Reserved.
% Author: Santiago Balseiro
    
    % precompute cholesky decomp. of the covariance matrices for the multi
    % variate normal number generator.
    cholesky = cell(1,T);
    for t=1:T
        cholesky{t} = cholcov(type(t).SIGMA);
    end
    % cdf of the types distribution 
    type_prob_cdf = cumsum(type_prob);
    % make sure that the last element is exactly one. Else, in the very
    % unlikely case (yet possible) that the random number is 1, the last
    % element may not be returned.
    type_prob_cdf(T) = 1;
   
    % samples
    Q = zeros(samples, A);
    t = zeros(samples, 1);
    
    for n=1:samples
        % sample the type
        r = rand();
        t(n) = find(r <= type_prob_cdf, 1, 'first');
        
        % sample the quality for the ads in the type
        Q_ads = exp(mvnrnd (type(t(n)).mu, type(t(n)).SIGMA, [], cholesky{t(n)}));
        % build the complete quality vector for ads in the type and those
        % who are not
        Q(n, :) = tau;
        Q(n, type_ad(t(n),:)==1) = Q_ads;        
    end   
end
