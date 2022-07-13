function [ AdX ] = LoadAdXFile( adxfile, lambda, price_scale_factor)
% Loads an AdX file. The data is used to solve the maximum expected revenue
% problem under opportunity cost c, that is,
%   R(c) = max_{0<s<1} r(s) + (1-s)*c
%
%   INPUT
%       adxfile= name of file with adx data. The file has a header:
%                accept.prob price revenue
%                The file should have three columnns of numbers separated
%                by white spaces. accept.prob is the probability that the
%                impression is accepted by the AdX, i.e, there is a bid
%                larger than the price. revenue is the expected revenue
%                under that reserve price or accept.prob.
%       lambda:double = the dollar/quality tradeoff parameter of
%                    the publisher. Not loaded from the file. Defaults to
%                    1.
%       price_scale_factor:double = scale factor that price should be
%                                   multiplied by to account for the units.
%
%   OUTPUT
%       AdX:struct = AdX data. The structure has the following fields:
%                - reserve_price(s) = a function giving the reserve price
%                    needed to induce an acceptance probability of s.
%                - opt_revenue(c) = a function giving the maximum expected
%                    revenue under opportunity cost c (as given by R(c)).
%                - opt_survival_prob(c) = the survival probability that
%                    verifies the maximum of R(c).
%                - revenue(s) = a function giving the expected revenue from
%                    the AdX given an acceptance probability of s.
%                - lambda:double = the dollar/quality tradeoff parameter of
%                    the publisher. Not loaded from the file. Defaults to 1.
%
% Copyright 2010 Google Inc. All Rights Reserved.
% Author: Santiago Balseiro

%% OPEN THE FILE AND READ
    
    % attempt to open data file
    [fid, message] = fopen(adxfile);
    if fid < 0 || isempty(fid)
        error('LoadAdXFile:fopen', 'The file ''%s'' could not be opened because: %s', adfile, message);
    end

    % get the header
    line = fgetl(fid);
    
    % read the rest of the data
    inputs = textscan(fid, '%f %f %f');
    
    % close data file
    fclose(fid);
    
%% PROCESS THE RESULTS
    % check for lambda argument
    if nargin < 2
        lambda = 1;        
    end
    if nargin < 3
        price_scale_factor = 1;
    end
    
    % parse the header
    [header1, line] = strtok (line);
    [header2, line] = strtok (line);    
    [header3] = strtok (line);    
    if strcmpi(header1, 'accept.prob') == 0 || strcmpi(header2, 'price') == 0 || strcmpi(header3, 'revenue') == 0
        error('LoadAdXFile: file has wrong headers. Was expecting: accept.prob price revenue')
    end
    
    % get the arrays
    survival_prob = inputs{1};
    reserve_price = inputs{2} * price_scale_factor;
    revenue = inputs{3} * price_scale_factor;
    
    % sort according to survival_prob in increasing order
    [survival_prob, ix] = sort(survival_prob);
    reserve_price = reserve_price(ix);
    revenue = revenue(ix);
        
    % for the optimization we take the convex hull of the revenue function
    % with respect to the survival probabilities
    ch = convhull( survival_prob, revenue, 'simplify', true );
    ch = flipud(ch);
    ch = ch([ true; diff(survival_prob(ch) ) > 0 ] ); % remove extra points
    % convex hull
    revenue_ch = revenue(ch);
    survival_prob_ch = survival_prob(ch);
    marginal_revenue_ch = diff( revenue_ch ) ./ diff ( survival_prob_ch );
    % build the polynomials for the interpolation
    pp_survival_prob = mkpp( [-Inf; flipud( marginal_revenue_ch ); Inf], flipud( survival_prob_ch ) );
    pp_revenue = mkpp( [-Inf; flipud( marginal_revenue_ch ); Inf], flipud( revenue_ch ) );
    
    %% construct the AdX functions
    % optimal survival probability function
    function s = opt_survival_prob_fcn(c)
        s = ppval( pp_survival_prob, c);
        % always return a row vector
        s = reshape( s, 1, numel(s));
    end
    % optimal expected revenue function
    function R = opt_revenue_fcn(c)
        r = ppval( pp_revenue, c);
        s = ppval( pp_survival_prob, c);
        R = r + (1-s) .* c;
        % always return a row vector
        R = reshape( R, 1, numel(R));
    end

    AdX = struct('reserve_price', @(s) interp1(survival_prob, reserve_price, s), ...
        'opt_survival_prob', @opt_survival_prob_fcn, ...
        'opt_revenue', @opt_revenue_fcn, ...
        'revenue', @(s) interp1(survival_prob, revenue, s), ... 
        'lambda', lambda );
end
    


