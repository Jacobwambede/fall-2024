using Optim, HTTP, LinearAlgebra, Random, Statistics, DataFrames, CSV, Distributions, Test

# Question 1: Estimate linear regression model by GMM
function question1()
    println("Question 1: Linear Regression by GMM")
    
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1
    
    # GMM objective function
    function gmm_obj(β, X, y)
        ε = y - X * β
        g = X' * ε
        return g' * g
    end
    
    # Estimate using GMM
    β_gmm = optimize(β -> gmm_obj(β, X, y), rand(size(X,2)), LBFGS()).minimizer
    
    # Compare with OLS
    β_ols = inv(X'X) * X'y
    
    println("GMM estimates: ", β_gmm)
    println("OLS estimates: ", β_ols)
end

# Question 2: Estimate multinomial logit model by GMM
function question2()
    println("\nQuestion 2: Multinomial Logit by GMM")
    
    # Load and prepare data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref(8:13), :occupation] .= 7
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation
    
    # Multinomial logit GMM objective function
    function mlogit_gmm(β, X, y)
        K, N = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, K, J-1)
        
        P = zeros(N, J)
        for j in 1:J-1
            P[:, j] = exp.(X' * β_matrix[:, j])
        end
        P[:, J] .= 1
        P ./= sum(P, dims=2)
        
        g = zeros(N * J)
        for i in 1:N
            for j in 1:J
                g[(i-1)*J + j] = (y[i] == j ? 1 : 0) - P[i, j]
            end
        end
        
        return g' * g
    end
    
    # Estimate using GMM
    K = size(X, 2)
    J = length(unique(y))
    β_init = rand(K * (J-1))  # Initialize with random values
    β_gmm = optimize(β -> mlogit_gmm(β, X', y), β_init, LBFGS(), Optim.Options(g_tol=1e-5)).minimizer
    
    println("GMM estimates:")
    println(reshape(β_gmm, K, J-1))
    
    # Manual implementation of multinomial logit MLE for comparison
    function mlogit_mle(β, X, y)
        K, N = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, K, J-1)
        
        P = zeros(N, J)
        for j in 1:J-1
            P[:, j] = exp.(X' * β_matrix[:, j])
        end
        P[:, J] .= 1
        P ./= sum(P, dims=2)
        
        loglik = 0.0
        for i in 1:N
            loglik += log(P[i, y[i]])
        end
        
        return -loglik  # Return negative log-likelihood for minimization
    end
    
    β_mle = optimize(β -> mlogit_mle(β, X', y), β_init, LBFGS(), Optim.Options(g_tol=1e-5)).minimizer
    
    println("\nMLE estimates:")
    println(reshape(β_mle, K, J-1))
end

# Question 3: Simulate multinomial logit data and estimate
function question3()
    println("\nQuestion 3: Simulate and Estimate Multinomial Logit")
    
    function simulate_mlogit(N, K, J, true_β)
        X = [ones(N) randn(N, K-1)]
        U = zeros(N, J)
        for j in 1:J-1
            U[:, j] = X * true_β[:, j] + rand(Gumbel(), N)
        end
        U[:, J] = rand(Gumbel(), N)
        y = [argmax(U[i, :]) for i in 1:N]
        return X, y
    end
    
    N, K, J = 10000, 4, 5
    true_β = [1.0 0.5 -0.5 1.0; 0.5 1.0 0.5 -0.5; 0.0 -0.5 1.0 0.5; -0.5 0.0 -1.0 1.0]'
    X, y = simulate_mlogit(N, K, J, true_β)
    
    function mlogit_smm(β, X, y, D)
        K, N = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, K, J-1)
        
        obs_moments = zeros(N, J)
        for i in 1:N
            obs_moments[i, y[i]] = 1
        end
        
        sim_moments = zeros(N, J, D)
        for d in 1:D
            U = zeros(N, J)
            for j in 1:J-1
                U[:, j] = X' * β_matrix[:, j] + rand(Gumbel(), N)
            end
            U[:, J] = rand(Gumbel(), N)
            sim_y = [argmax(U[i, :]) for i in 1:N]
            for i in 1:N
                sim_moments[i, sim_y[i], d] = 1
            end
        end
        
        avg_sim_moments = mean(sim_moments, dims=3)[:,:,1]
        g = vec(obs_moments - avg_sim_moments)
        
        return g' * g
    end
    
    D = 50  # Number of simulations
    β_init = rand(K * (J-1))
    β_smm = optimize(β -> mlogit_smm(β, X', y, D), β_init, LBFGS(), Optim.Options(g_tol=1e-5)).minimizer
    
    println("True β:")
    println(true_β)
    println("\nEstimated β:")
    println(reshape(β_smm, K, J-1))
end

# Question 5: Estimate multinomial logit model by SMM using real data
function question5()
    println("\nQuestion 5: Multinomial Logit by SMM using Real Data")
    
    # Load and prepare data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref(8:13), :occupation] .= 7
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation
    
    function mlogit_smm(β, X, y, D)
        K, N = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, K, J-1)
        
        obs_moments = zeros(N, J)
        for i in 1:N
            obs_moments[i, y[i]] = 1
        end
        
        sim_moments = zeros(N, J, D)
        for d in 1:D
            U = zeros(N, J)
            for j in 1:J-1
                U[:, j] = X' * β_matrix[:, j] + rand(Gumbel(), N)
            end
            U[:, J] = rand(Gumbel(), N)
            sim_y = [argmax(U[i, :]) for i in 1:N]
            for i in 1:N
                sim_moments[i, sim_y[i], d] = 1
            end
        end
        
        avg_sim_moments = mean(sim_moments, dims=3)[:,:,1]
        g = vec(obs_moments - avg_sim_moments)
        
        return g' * g
    end
    
    K = size(X, 2)
    J = length(unique(y))
    D = 50  # Number of simulations
    β_init = rand(K * (J-1))
    β_smm = optimize(β -> mlogit_smm(β, X', y, D), β_init, LBFGS(), Optim.Options(g_tol=1e-5)).minimizer
    
    println("SMM estimates:")
    println(reshape(β_smm, K, J-1))
    
    # Manual implementation of multinomial logit MLE for comparison
    function mlogit_mle(β, X, y)
        K, N = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, K, J-1)
        
        P = zeros(N, J)
        for j in 1:J-1
            P[:, j] = exp.(X' * β_matrix[:, j])
        end
        P[:, J] .= 1
        P ./= sum(P, dims=2)
        
        loglik = 0.0
        for i in 1:N
            loglik += log(P[i, y[i]])
        end
        
        return -loglik  # Return negative log-likelihood for minimization
    end
    
    β_mle = optimize(β -> mlogit_mle(β, X', y), β_init, LBFGS(), Optim.Options(g_tol=1e-5)).minimizer
    
    println("\nMLE estimates:")
    println(reshape(β_mle, K, J-1))
end

# Main function to run all questions
function main()
    question1()
    question2()
    question3()
    question5()
end

# Run the main function
main()

# Unit tests
function run_tests()
    @testset "Problem Set 7 Tests" begin
        @testset "Question 1: Linear Regression GMM" begin
            β_gmm, β_ols = question1()
            @test length(β_gmm) == 4
            @test length(β_ols) == 4
            @test isapprox(β_gmm, β_ols, rtol=0.1)
        end

        @testset "Question 2: Multinomial Logit GMM" begin
            β_gmm, β_mle = question2()
            @test size(reshape(β_gmm, 4, 6)) == (4, 6)
            @test size(reshape(β_mle, 4, 6)) == (4, 6)
            @test isapprox(β_gmm, β_mle, rtol=0.2)
        end

        @testset "Question 3: Simulate and Estimate Multinomial Logit" begin
            true_β, estimated_β = question3()
            @test size(true_β) == size(estimated_β)
            @test isapprox(true_β, estimated_β, rtol=0.2)
        end

        @testset "Question 5: Multinomial Logit SMM with Real Data" begin
            β_smm = question5()
            @test size(β_smm) == (4, 6)
            @test all(isfinite, β_smm)
        end
    end
end
# Run the main function
main()