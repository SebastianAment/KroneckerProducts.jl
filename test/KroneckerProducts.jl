module TestKroneckerProducts
using Test
using LinearAlgebra
using KroneckerProducts
using KroneckerProducts: kronecker, ⊗, KroneckerProduct, issquare, allsquare, isequiv, ispower, order
                            # isposdef, issuccess, issymmetric
using Random

element_types = (Float64, ComplexF64)
for elty in element_types
    @testset "eltype = $elty" begin
        # IDEA: loop over kronecker matrices
        A = randn(elty, 3, 3)
        B = randn(elty, 2, 4)
        C = randn(elty, 4, 4)
        # D = randn(elty, 5, 5)
        # kronecker_products = [kronecker((A,)),
        #                     kronecker(A, B),
        #                     kronecker(A, B, C),
        #                     kronecker(A, C, D)]
        @testset "basic properties" begin

                    @test kronecker(A) isa Matrix
                    K = kronecker(A, B)
                    @test K isa KroneckerProduct
                    @test size(K) == size(A) .* size(B)
                    @test order(K) == 2
                    @test IndexStyle(typeof(K)) == IndexCartesian()

                    # test indexing
                    @test K[1, 1] == A[1, 1] * B[1, 1]

                    @test Matrix(K) ≈ kron(A, B)

                    kronAB = kron(A, B)

                    # matvec
                    x = randn(elty, size(K, 2))
                    @test K * x ≈ kronAB * x

                    # matmat
                    x = randn(elty, size(K, 2), 2)
                    @test K * x ≈ kronAB * x

                    # right solve
                    K = A ⊗ A
                    b = randn(elty, size(K, 1))
                    x = K \ b
                    @test K*x ≈ b

                    ############### ternary product
                    K = kronecker(A, B, C)
                    @test Matrix(K) ≈ kron(A, B, C)
                    @test order(K) == 3
                    @test size(K) == size(A) .* size(B) .* size(C)
                    kronABC = kron(A, B, C)

                    @test issquare(A ⊗ C)
                    @test !issquare(A ⊗ B)
                    @test allsquare(A ⊗ C)
                    @test !allsquare(A ⊗ B)
                    @test ispower(A ⊗ A ⊗ copy(A))
                    @test !ispower(A ⊗ B ⊗ A)
                    @test isequiv(A ⊗ A ⊗ A)
                    @test !isequiv(A ⊗ A ⊗ copy(A))
                    @test !isequiv(A ⊗ B ⊗ A)

                    # logabsdet test
                    @test all(logabsdet(KroneckerProduct((A,))) .≈ logabsdet(A))
                    @test logabsdet(A ⊗ A)[1] ≈ logabsdet(kron(A, A))[1]
                    @test logabsdet(A ⊗ A)[2] ≈ logabsdet(kron(A, A))[2]
                    @test logabsdet(A ⊗ C)[1] ≈ logabsdet(kron(A, C))[1]
                    @test logabsdet(A ⊗ C)[2] ≈ logabsdet(kron(A, C))[2]

                    # logdet test on positive definite matrix
                    AA = A'A + I
                    CC = C'C + I

                    @test logabsdet(AA ⊗ AA)[1] ≈ logabsdet(kron(AA, AA))[1]
                    @test logabsdet(AA ⊗ AA)[2] ≈ logabsdet(kron(AA, AA))[2]
                    @test logabsdet(AA ⊗ CC)[1] ≈ logabsdet(kron(AA, CC))[1]
                    @test logabsdet(AA ⊗ CC)[2] ≈ logabsdet(kron(AA, CC))[2]

                    @test logdet(KroneckerProduct((AA,))) ≈ logdet(AA)
                    @test logdet(AA ⊗ CC) ≈ logdet(kron(AA, CC))
                    @test logdet(AA ⊗ AA) ≈ logdet(kron(AA, AA)) # testing isequiv branch in logdet

                    @test det(KroneckerProduct((AA,))) ≈ det(AA)
                    @test det(AA ⊗ CC) ≈ det(kron(AA, CC))
                    @test det(AA ⊗ AA) ≈ det(kron(AA, AA))

                    @test tr(KroneckerProduct((AA,))) ≈ tr(AA)
                    @test tr(AA ⊗ CC) ≈ tr(kron(AA, CC))
                    @test tr(AA ⊗ AA) ≈ tr(kron(AA, AA))

                    @test Matrix(inv(AA ⊗ CC)) ≈ inv(kron(AA, CC))

                    @test Matrix(adjoint(AA ⊗ CC)) ≈ adjoint(kron(AA, CC))
                    @test Matrix(transpose(AA ⊗ CC)) ≈ transpose(kron(AA, CC))
                    @test Matrix(conj(AA ⊗ CC)) ≈ conj(kron(AA, CC))
                    @test factorize(AA ⊗ CC) isa FactorizedKroneckerProduct
                    @test all(F -> F isa LinearAlgebra.QRCompactWY, qr(AA ⊗ CC).factors)
                    @test all(F -> F isa Cholesky, cholesky(AA ⊗ CC).factors)
                    @test all(F -> F isa CholeskyPivoted, cholesky(AA ⊗ CC, Val(true)).factors)

                    # right multiplication
                    x = randn(elty, size(K, 2))
                    @test K * x ≈ kronABC * x

                    # left multiplication
                    x = randn(elty, size(K, 1))
                    @test x'K ≈ x'kronABC

                    # matmat
                    k = 3
                    X = randn(elty, size(K, 2), k)
                    @test K*X ≈ kronABC*X
                    X = randn(elty, size(K, 1), k)
                    @test X'*K ≈ X'*kronABC
                    X = randn(elty, k, size(K, 1))
                    @test X*K ≈ X*kronABC
                    X = randn(elty, k, size(K, 2))
                    @test K*X' ≈ kronABC*X'

                    # right solve
                    K = A ⊗ A ⊗ C
                    b = randn(elty, size(K, 1))
                    x = K \ b
                    @test K*x ≈ b

                    K = AA ⊗ CC
                    Chol = ⊗(cholesky, K)
                    @test isposdef(Chol)
                    b = randn(elty, size(Chol, 1))
                    @test K * (Chol \ b) ≈ b
        end

        @testset "indexing" begin
            # get index
            K = kronecker(A, B)
            kronAB = Matrix(K)
            for i in 1:size(K, 1), j in 1:size(K, 2)
                @test K[i,j] == kronAB[i,j]
            end
            K = kronecker(A, B, C)
            kronABC = Matrix(K)
            for i in 1:size(K, 1), j in 1:size(K, 2)
                @test K[i,j] == kronABC[i,j]
            end
        end

        @testset "mul!" begin
            n = 16
            A = randn(elty, n, n)
            B = randn(elty, n, n)
            K = kronecker(A, B)
            k = size(K, 1)
            x = randn(elty, k)
            y = randn(elty, k)
            mul!(y, K, x)
            MK = Matrix(K)
            @test y ≈ MK*x

            α, β = randn(2)
            r = α * MK * x + β * y
            mul!(y, K, x, α, β)
            @test y ≈ r
        end
    end # testset
end # loop over elty

end # TestKroneckerProducts
