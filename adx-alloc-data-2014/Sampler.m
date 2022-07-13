
samples = 1e5;
samples = 5e5;
tau = 0;

values_i = [1,2,5];
%values_i = 2;
continue_corr = false;
doplots = true;
values_corr = [0, 0.5, 0.9, 0.99, 0.999];


for i = values_i %1:7
    adfile = sprintf('pub%d-ads.txt', i);
    typefile = sprintf('pub%d-types.txt', i);
    fprintf('Loading...\n');
    [ A, T, rho, type_prob, type_ad, type ] = LoadSynthFile( adfile, typefile );

    for corr = values_corr
        type_corr = corr;
        Q_corr = corr;        
        fprintf('Generating...\n');
        [Q, t] = GenerateSampleErgodic (A, T, rho, type_ad, type_prob, tau, type, samples, type_corr, Q_corr, continue_corr);
        
        if doplots
            plot( 1:samples, Q )
            xlabel('time (t)')
            ylabel('quality (Q)')
            filename = sprintf('pub%d-corr%g-sample.png', i, corr);
            print( filename, '-dpng')
            %f = gcf;
            %f.PaperOrientation = 'landscape';
            %print( filename, '-dpdf', '-fillpage')
        end

        fprintf('Saving...\n');
        samplefile = sprintf('pub%d-corr%g-sample.txt', i, corr)
        csvwrite(samplefile,Q);
    end
end