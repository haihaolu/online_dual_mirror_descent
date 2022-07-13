function [ A, T, rho, type_prob, type_ad, type ] = LoadSynthFile( adfile, typefile )
% Loads synthetic data files into memory.
%
%   INPUT
%       adfile: name of file with ad data. A and rho are built from this file.
%               One row per type with format:
%               advertiser: %d rho: %f
%       typefile: name of file with type data. T, type_prob, type_ad, and type are
%                 are built from this file. One row per type with format:
%                 type: %d prob: %f advertisers: [%d+] mean [%f+] cov [%f+]
%
%   OUTPUT
%       A:int = number of advertisers
%       T:int = number of types
%       rho:double(1,A) = capacity to impression ratio of the advertisers
%       type_prob:double(1,T) = occurrence probability of each type
%	    type_ad:bool(T,A) = type-ad graph incidence matrix. type_ad(t,a) = 1 if advertiser a is in type t
%       type:struct(T) = structure specifying the mean and covariance of the lognormal distribution of the 
%                        advertisers within each type. the structure has two fields: 
%                           - type(t).mu:double(1,A(t)) = mean vector of the normal distribution.
%                           - type(t).SIGMA:double(A(t),A(t)) = covariance matrix of the normal distribution.    
%                        where A(t)=sum(type_ad(t,:)) is the number of ads in the type
%
% Copyright 2010 Google Inc. All Rights Reserved.
% Author: Santiago Balseiro

    tol = 1e-6;
%% Advertisers
    
    % attempt to open data file
    [fid, message] = fopen(adfile);
    if fid < 0 || isempty(fid)
        error('LoadSynthFile:fopen', 'The file ''%s'' could not be opened because: %s',adfile,message);
    end

    inputs = textscan(fid, 'advertiser: %d rho: %f' );
    
    % close data file
    fclose(fid);
    
    % process the result
    % number of advertisers
    A = length(inputs{1});
    ad_index =inputs{1};
    % capacities
    rho = inputs{2}';
    
    
%% Types
    
    % attempt to open data file
    [fid, message] = fopen(typefile);
    if fid < 0 || isempty(fid)
        error('LoadSynthFile:fopen', 'The file ''%s'' could not be opened because: %s',typefile,message);
    end

    inputs = textscan(fid, 'type: %d prob: %f advertisers: [%[^]]] mean: [%[^]]] cov: [%[^]]]' );
    
    % close data file
    fclose(fid);
    
    % process the result
    % number of types
    T = length(inputs{1});
    % probabilities
    type_prob = inputs{2}';
    
    % type_ad matrix
    type_ad = zeros(T,A);
    for t=1:T
        % scan the ads in the type
        row_ads = textscan(inputs{3}{t}, '%d', 'Delimiter', ',');
        % write results to the type_ad matrix
        for ad = row_ads{1}'
            type_ad(t, ad_index == ad) = 1;
        end
    end
    
    % mean and convariance
    % type data
    type = struct('mu', cell(1,T) , 'SIGMA', cell(1,T));    
    for t=1:T
        % number of ads expected
        ad_count = sum(type_ad(t,:));

        % scan the mean vector in the type
        mean_vector = textscan(inputs{4}{t}, '%f', 'Delimiter', ',');
        assert (length(mean_vector{1}) == ad_count, 'Mean vectorof type %d has wrong length', t);

        % store it in the type data struc
        type(t).mu = mean_vector{1}';
        
        % scan the covariance vector in the type
        % the vector is the upper triangular part of the cov matrix by
        % column (column major order)
        cov_vector = textscan(inputs{5}{t}, '%f', 'Delimiter', ',');
        assert (length(cov_vector{1}) == ad_count*(ad_count+1) / 2, 'Covariance vector of type %d has wrong length',t );
        
        % convert the vector to an upper triangular matrix
        type(t).SIGMA = zeros(ad_count, ad_count);
        type(t).SIGMA(triu(ones(ad_count),0)==1) = cov_vector{1}';
        
        % copy the upper triangular part to the lower triangular part
        transpose = type(t).SIGMA';
        type(t).SIGMA(tril(ones(ad_count),-1) == 1) = transpose(tril(ones(ad_count),-1) == 1);
        
        % check the covariance matrix
        assert (all(all(type(t).SIGMA==type(t).SIGMA')), 'Covariance matrix of type %d is not symmetric', t);
        assert(all(eig(type(t).SIGMA) > -tol), 'Covariance matrix of type %d is not positive definite', t);
    end
        
end
    


