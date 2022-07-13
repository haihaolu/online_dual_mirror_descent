
tau = 0;
lambda = 0.1;
values_i = [1,2,5];

for i = values_i
    adfile = sprintf('pub%d-ads.txt', i);
    typefile = sprintf('pub%d-types.txt', i);
    samplefile = sprintf('pub%d-sample.txt', i);
    optfile = sprintf('pub%d-opt.txt', i);
    [ A, T, rho, type_prob, type_ad, type ] = LoadSynthFile( adfile, typefile );

    %fprintf('Loading...\n');
    %[Q, t] = GenerateSample (A, T, rho, type_ad, type_prob, tau, type, samples);

    fprintf('Loading...\n');
    Q = csvread(samplefile);
    
    mat2str(max(Q(:)))
    %Q = Q ./ max(Q(:));
    %T = size(Q,1);
    %offline = SolveOfflineProblem( A, T, Q', rho', lambda)
    
    %dlmwrite(optfile, offline,'precision',10)
    
end