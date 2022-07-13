function [sample_Q, sample_t] = GenerateSampleErgodic (A, T, rho, type_ad, type_prob, tau, type, samples, type_corr, Q_corr, continue_corr)
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

    % cdf of the types distribution 
    type_prob_cdf = cumsum(type_prob);
    % make sure that the last element is exactly one. Else, in the very
    % unlikely case (yet possible) that the random number is 1, the last
    % element may not be returned.
    type_prob_cdf(T) = 1;
    
    
    % determine the types
    sample_t = zeros(samples, 1);
    % sample the initial type
    sample_t(1) = find( rand() <= type_prob_cdf, 1, 'first');    
    for n = 2:samples
        if rand() > type_corr
            sample_t(n) = find( rand() <= type_prob_cdf, 1, 'first');    
        else
            sample_t(n) = sample_t(n-1);
        end
    end
    % build a 0/1 matrix with the identities of the types
    sample_t_matrix = zeros(samples, T);
    sample_t_matrix( sub2ind( [samples,T], (1:samples), sample_t') ) = 1;
    cumsum_sample_t = cumsum(sample_t_matrix);
    
    
    % precompute the normal numbers.
    logQ = cell(1,T);
    for t=1:T
        logQs = zeros(samples, numel(type(t).mu));
        logQs(1,:) = mvnrnd (type(t).mu, type(t).SIGMA);        
        epsilons = mvnrnd (type(t).mu * (1-Q_corr), type(t).SIGMA * (1-Q_corr)^2, samples);
        for n = 2:samples
            logQs(n,:) = Q_corr * logQs(n-1, :) + epsilons(n,:);
        end
        logQ{t} = logQs;
    end
    

    % build up the samples
    sample_Q = ones(samples, A) * tau;   
    for n=1:samples
        % build the complete quality vector for ads in the type and those
        % who are not
        t = sample_t(n);
        if continue_corr
            sample_Q(n, type_ad(t,:)==1) = exp( logQ{t}(cumsum_sample_t(n,t),:) );        
        else
            sample_Q(n, type_ad(t,:)==1) = exp( logQ{t}(n,:) );        
        end
    end   
end
