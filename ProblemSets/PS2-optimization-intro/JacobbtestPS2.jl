
# problem set 2 solutions

using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff, Test # for bonus at the very end

function allwrap()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    startval = rand(1)   # random starting value
    result = optimize(minusf, startval, BFGS())
    println("argmin (minimizer) is ",Optim.minimizer(result)[1])
    println("min is ",Optim.minimum(result))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y
    println(bols)
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println(bols_lm)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function logit(alpha, X, y)

        P = exp.(X*alpha)./(1 .+ exp.(X*alpha))

        loglike = -sum( (y.==1).*log.(P) .+ (y.==0).*log.(1 .- P) )

        return loglike
    end
    alpha_hat_optim = optimize(a -> logit(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(alpha_hat_optim.minimizer)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println(alpha_hat_glm)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, y)
        
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    alpha_zero = zeros(6*size(X,2))
    alpha_rand = rand(6*size(X,2))
    alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    alpha_start = alpha_true.*rand(size(alpha_true))
    println(size(alpha_true))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)
    
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # bonus: how to get standard errors?
    # need to obtain the hessian of the obj fun
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # first, we need to slightly modify our objective function
    function mlogit_for_h(alpha, X, y)
        
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(alpha)) # this line is new
        num   = zeros(T,N,J)                      # this line is new
        dem   = zeros(T,N)                        # this line is new
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    # declare that the objective function is twice differentiable
    td = TwiceDifferentiable(b -> mlogit_for_h(b, X, y), alpha_start; autodiff = :forward)
    # run the optimizer
    alpha_hat_optim_ad = optimize(td, alpha_zero, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle_ad = alpha_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, alpha_hat_mle_ad)
    # standard errors = sqrt(diag(inv(H))) [usually it's -H but we've already multiplied the obj fun by -1]
    # if ols: vcov = σ^2*(X'X)^-1
        # se: sqrt(diag(vcov))
    alpha_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println([alpha_hat_mle_ad alpha_hat_mle_ad_se]) # these standard errors match Stata

    return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
#allwrap()


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 7
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Problem Set 2 Tests" begin
    # Test for question 1
    @testset "Question 1: Optimization" begin
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        minusf(x) = -f(x)
        result = optimize(minusf, [0.0], BFGS())
        #@test Optim.minimizer(result)[1] == -7.37824
        @test isapprox(Optim.minimizer(result)[1], -7.37824, atol=1e-5)
        @test isapprox(Optim.minimum(result), -964.313384, atol=1e-5)
    end

    # Test for question 2
    @testset "Question 2: OLS" begin
        # Create a small dataset for testing
        X_test = rand(500,5)
        β = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_test = X_test*β + randn(500)
        
        function ols(beta, X, y)
            ssr = (y .- X*beta)'*(y .- X*beta)
            return ssr
        end
        
        beta_hat_ols = optimize(b -> ols(b, X_test, y_test), rand(size(X_test,2)), LBFGS()).minimizer
        bols = inv(X_test'*X_test)*X_test'*y_test
        
        @test isapprox(beta_hat_ols, bols, atol=1e-10)
        @test isapprox(beta_hat_ols, β, atol=1e0)
        @test isapprox(bols, β, atol=1e0)
    end

    # Test for question 3
    @testset "Question 3: Logit" begin
        function logit(alpha, X, y)
            P = exp.(X*alpha) ./ (1 .+ exp.(X*alpha))
            loglike = -sum((y .== 1) .* log.(P) .+ (y .== 0) .* log.(1 .- P))
            return loglike
        end
        
        X_test = randn(1000,5)
        β = [1.0, 0.5, -0.5, 2.0, -2.0]
        P_test = 1 ./ (1 .+ exp.(-X_test*β))
        y_test = rand.(Bernoulli.(P_test))
        
        alpha_hat = optimize(a -> logit(a, X_test, y_test), rand(size(X_test,2)), LBFGS()).minimizer
        
        # Test that the log-likelihood is negative
        @test isapprox(alpha_hat, β, atol=1e0)
        @test sign.(alpha_hat) == sign.(β)
    end

    # Test for question 5
    @testset "Question 5: Multinomial Logit" begin
        function mlogit(alpha, X, y)
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            for j=1:J
                bigY[:,j] = y.==j
            end
            bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
            
            num = zeros(N,J)
            dem = zeros(N)
            for j=1:J
                num[:,j] = exp.(X*bigAlpha[:,j])
                dem .+= num[:,j]
            end
            
            P = num./repeat(dem,1,J)
            
            loglike = -sum(bigY.*log.(P))
            
            return loglike
        end
        
        function generate_multinomial_logit_data(N::Int, K::Int)
            # Generate random normal X matrix
            X = randn(N, K)
        
            # Add a column of ones for the intercept
            X = hcat(ones(N), X)
        
            # Define beta parameter matrix (K+1 by 3)
            # Using nice round numbers as requested
            β = [
                1.0  0.1 0.0;  # Intercept
                2.0  1.1 0.0;
               -1.5  1.0 0.0;
                0.5 -0.5 0.0;
                1.0 -1.0 0.0
            ]
        
            # Ensure β has the correct dimensions
            @assert size(β, 1) == K + 1 "β should have K+1 rows"
            @assert size(β, 2) == 3 "β should have 3 columns for 3 choice alternatives"
        
            # Calculate utilities
            U = X * β
        
            # Calculate probabilities
            P = exp.(U) ./ sum(exp.(U), dims=2)
        
            # Generate y vector
            y = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]
        
            return X, y, β
        end
        
        N, K = 1000, 4
        X_test, y_test, β = generate_multinomial_logit_data(N, K)
        
        alpha_start = rand(2*size(X_test,2))
        alpha_hat = optimize(a -> mlogit(a, X_test, y_test), alpha_start, LBFGS()).minimizer
        
        # Test that the log-likelihood is negative
        @test size(alpha_hat) == size(β[:,1:2][:])
        @test isapprox(alpha_hat, β[:,1:2][:], atol=1e0)
        @test sign.(alpha_hat) == sign.(β[:,1:2][:])
        @show alpha_hat
        @show β[:,1:2][:]
    end
end
