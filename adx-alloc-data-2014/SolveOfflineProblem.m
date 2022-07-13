function [offline] = SolveOfflineProblem( A, T, Q, rho, lambda)

    cvx_tic
    cvx_solver mosek_2
    cvx_begin
        variable x(A,T);
        maximize sum( sum( Q .* x ) ) + lambda * sum( sum( entr( x ) ) + entr( 1 - sum(x,1) ) ) ;
        subject to
        sum(x,2) <= rho * T;
        sum(x,1) <= 1;
        x >= 0;
    cvx_end
    cvx_toc

    offline = cvx_optval;

end