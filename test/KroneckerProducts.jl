module TestKroneckerProducts
using Test
using LinearAlgebra
using KroneckerProducts
using KroneckerProducts: kronecker, ⊗, KroneckerProduct, issquare, allsquare, isequiv, ispower, order, samesize
using Random
using LazyInverses

element_types = (Float64, ComplexF64)
for elty in element_types
    @testset "eltype = $elty" begin
        # IDEA: loop over kronecker matrices
        A = randn(elty, 3, 3)
        B = randn(elty, 2, 4)
        C = randn(elty, 4, 4)
        D = randn(elty, 5, 5)
        AA = A'A + I # to test positive definite matrices
        CC = C'C + I
        DD = D'D + I
        tuples = [(A, A), (A, A, copy(A)), (A, A, A), # square kronecker power
                (A, B), (A, C, D), (A, B, C, D), # square kronecker products
                (A, B, C), (B, B), (B, B, B), # rectangular kronecker products
                (AA, AA), (AA, CC), (AA, CC, DD)] # positive definite products
        @testset "smart constructor" begin
            @test kronecker(A) === A
            a = randn(elty)
            @test kronecker(a) == a
            @test Matrix(KroneckerProduct((A,))) ≈ A
        end
        for (i, tuple) in enumerate(tuples)
            @testset "kronecker product $i" begin
                K = kronecker(tuple)
                isposdef_K = isposdef(K) # compute once since we need it multiple times
                M = kron(tuple...)
                @testset "basic properties" begin
                    @test K isa KroneckerProduct
                    @test size(K) == size(M)
                    @test order(K) == length(tuple)
                    @test IndexStyle(typeof(K)) == IndexCartesian()
                    @test Matrix(K) ≈ M

                    @test issquare(K) == (size(K, 1) == size(K, 2))
                    @test samesize(K) == all(==(size(tuple[1])), tuple)
                    @test allsquare(K) == all(issquare, tuple)
                    @test ispower(K) == all(==(tuple[1]), tuple)
                    @test isequiv(K) == all(A->A===tuple[1], tuple)
                    @test isposdef_K == isposdef(M)

                    # testing adjoint, transpose, conj
                    @test Matrix(adjoint(K)) ≈ adjoint(M)
                    @test Matrix(transpose(K)) ≈ transpose(M)
                    @test Matrix(conj(K)) ≈ conj(M)
                end

                @testset "determinants" begin
                    if issquare(K)
                        @test det(K) ≈ det(M)
                        @test logabsdet(K)[1] ≈ logabsdet(M)[1]
                        @test logabsdet(K)[2] ≈ logabsdet(M)[2]
                        @test tr(K) ≈ tr(M)
                    end
                    if isposdef_K
                        @test logdet(K) ≈ logdet(M)
                    end
                end

                @testset "factorization" begin
                    # factorizing generic kronecker product
                    @test factorize(K) isa FactorizedKroneckerProduct
                    qr_K = qr(K)
                    # display(typeof(qr_K.factors[1]))
                    @test all(F -> F isa LinearAlgebra.QRCompactWY, qr_K.factors)
                    if isposdef_K
                        cholesky_K = cholesky(K)
                        @test all(F -> F isa Cholesky, cholesky_K.factors)
                        pivoted_cholesky_K = cholesky(K, Val(true))
                        @test all(F -> F isa CholeskyPivoted, pivoted_cholesky_K.factors)
                    end
                    if issquare(K)
                        inv_M = inv(M)
                        @test Matrix(inv(K)) ≈ inv_M
                        @test Matrix(inverse(K)) ≈ inv_M
                    end
                    pinv_M = pinv(M)
                    @test Matrix(pinv(K)) ≈ pinv_M
                    @test Matrix(pseudoinverse(K)) ≈ pinv_M
                end

                @testset "indexing" begin
                    for i in 1:size(K, 1), j in 1:size(K, 2)
                        @test K[i, j] == M[i, j]
                    end
                end

                @testset "mul!" begin # IDEA: test for more complex type combinations
                    x = randn(elty, size(K, 2))
                    y = randn(elty, size(K, 1))
                    mul!(y, K, x)
                    @test y ≈ M*x
                    α, β = randn(2)
                    r = α * M * x + β * y
                    mul!(y, K, x, α, β)
                    @test y ≈ r

                    # lmul!(α, K)
                    # rmul!(K, α)
                end

                @testset "algebra" begin
                    # right multiplication
                    x = randn(elty, size(K, 2))
                    @test K * x ≈ M * x

                    # left multiplication
                    x = randn(elty, size(K, 1))
                    @test x'K ≈ x'M

                    # matmat
                    k = 3
                    X = randn(elty, size(K, 2), k)
                    @test K*X ≈ M*X
                    X = randn(elty, size(K, 1), k)
                    @test X'*K ≈ X'*M
                    X = randn(elty, k, size(K, 1))
                    @test X*K ≈ X*M
                    X = randn(elty, k, size(K, 2))
                    @test K*X' ≈ M*X'

                    # right solve
                    if allsquare(K)
                        b = randn(elty, size(K, 1))
                        x = K \ b
                        @test K*x ≈ b
                    end
                    if isposdef_K
                        Chol = ⊗(cholesky, K)
                        @test isposdef(Chol)
                        b = randn(elty, size(Chol, 1))
                        @test K * (Chol \ b) ≈ b
                    end
                    # mixed product property
                    @test Matrix(K'K) ≈ M'M
                    a = randn(elty)
                    @test Matrix(a*K) ≈ a*M
                    @test Matrix(K*a) ≈ M*a
                end
            end # testset
        end # loop over kronecker_products
    end # testset
end # loop over elty

end # TestKroneckerProducts
