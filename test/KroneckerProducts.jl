module TestKroneckerProducts
using Test
using KroneckerProducts: kronecker, ⊗, KroneckerProduct, issquare, allsquare,
                            isposdef, issuccess, issymmetric, ispower, order
using LinearAlgebra
using Random

# n, m, k = 3,
A = randn(3, 3)
B = randn(2, 4)
C = randn(2, 2)

@testset "basic properties" begin

    @test kronecker(A) isa Matrix
    K = kronecker(A, B)
    @test K isa KroneckerProduct
    @test size(K) == size(A) .* size(B)
    @test order(K) == 2

    # test indexing
    @test K[1,1] == A[1,1] * B[1,1]

    @test Matrix(K) ≈ kron(A, B)

    kronAB = kron(A, B)

    # matvec
    x = randn(size(K, 2))
    @test K * x ≈ kronAB * x

    # matmat
    x = randn(size(K, 2), 2)
    @test K * x ≈ kronAB * x

    # right solve
    K = A ⊗ A
    b = randn(size(K, 1))
    x = K \ b
    @test K*x ≈ b

    ############### ternary product
    C = randn(4, 4)
    K = kronecker(A, B, C)
    @test Matrix(K) ≈ kron(A, B, C)
    @test order(K) == 3
    @test size(K) == size(A) .* size(B) .* size(C)
    kronABC = kron(A, B, C)

    @test !issquare(A ⊗ B)
    @test issquare(A ⊗ C)
    @test allsquare(A ⊗ C)
    @test ispower(A ⊗ A ⊗ A)
    @test !ispower(A ⊗ B ⊗ A)
    # right multiplication
    x = randn(size(K, 2))
    @test K * x ≈ kronABC * x

    # left multiplication
    x = randn(size(K, 1))
    @test x'K ≈ x'kronABC

    # matmat
    x = randn(size(K, 2), 2)
    @test K*x ≈ kronABC*x
    x = randn(2, size(K, 1))
    @test x*K ≈ x*kronABC

    # right solve
    K = A ⊗ A ⊗ C
    b = randn(size(K, 1))
    x = K \ b
    @test K*x ≈ b

    K = A ⊗ B ⊗ C
    # println(KroneckerProducts.temporary_length(K))
    K = A'A ⊗ C'C
    Chol = ⊗(cholesky, K)
    @test isposdef(Chol)
    b = randn(size(Chol, 1))
    # b = K*x
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
    A = randn(n, n)
    B = randn(n, n)
    K = kronecker(A, B)
    k = size(K, 1)
    x = randn(k)
    y = randn(k)
    mul!(y, K, x)
    MK = Matrix(K)
    @test y ≈ MK*x

    α, β = randn(2)
    r = α * MK * x + β * y
    mul!(y, K, x, α, β)
    @test y ≈ r
end

end # TestKroneckerProducts
