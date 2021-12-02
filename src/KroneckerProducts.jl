module KroneckerProducts
using LinearAlgebra
using LinearAlgebra: checksquare
import LinearAlgebra: det, tr, logdet, issymmetric, ishermitian, isposdef,
adjoint, transpose, conj
import LinearAlgebra: issuccess, Matrix, size
import LinearAlgebra: cholesky, cholesky!, qr, qr!#, svd, eigen, bunchkaufman, lu,
import LinearAlgebra: \, *, /, rdiv!, ldiv!
using LazyInverses

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
LinearAlgebra.factorize(F::Factorization) = F

export kronecker, ⊗

# abstract type AbstractKroneckerProduct{T} end
# struct FactorizedKroneckerProduct{T} <: Factorization{T} end

# has field for temporary storage needed for vec-trick and solves
struct KroneckerProduct{T, A<:Tuple{Vararg{AbstractMatOrFac}},
                            V} <: Factorization{T}
    factors::A
    temporary::V
end

Base.eltype(K::KroneckerProduct{T}) where {T} = T
Base.IndexStyle(::Type{<:KroneckerProduct}) = IndexCartesian()

function KroneckerProduct(A::Tuple{Vararg{AbstractMatOrFac}})
    T = promote_type(eltype.(A)...)
    KroneckerProduct{T, typeof(A), Nothing}(A, nothing)
end

# TODO: initialize_temporary()
kronecker(A::AbstractMatOrFac) = A
kronecker(A::AbstractMatOrFac...) = KroneckerProduct(A)
kronecker(K::KroneckerProduct) = K
kronecker(A::Tuple{Vararg{AbstractMatOrFac}}) = KroneckerProduct(A)

# TODO: making sure we have a flat Kronecker hierarchy?
# only makes sense to break if we have a subset, which is a power
kronecker(A::AbstractMatOrFac, K::KroneckerProduct) = kronecker(A, K.factors...)
kronecker(K::KroneckerProduct, A::AbstractMatOrFac) = kronecker(A, K)
kronecker(K::KroneckerProduct, L::KroneckerProduct) = kronecker(K.factors..., L.factors...)

# create new product by applying function to each component matrix
kronecker(f, K::KroneckerProduct) = kronecker((f(A) for A in K.factors)...) #kronecker(f.(K.factors)...)
const ⊗ = kronecker

# not sure if this is necessary ...
# kronecker(A::AbstractMatrix, pow::Integer) = kronecker((A for _ in 1:pow)...)
# ⊗(A::AbstractMatrix, pow::Integer) = kronecker(A, pow)

Base.size(K::KroneckerProduct) = (size(K, 1), size(K, 2))
Base.size(K::KroneckerProduct, dim::Integer) = prod(A->size(A, dim), K.factors)

# TODO: meta
issquare(A::AbstractMatOrFac) = size(A, 1) == size(A, 2)
allsquare(K::KroneckerProduct) = all(issquare, K.factors)
issymmetric(K::KroneckerProduct) = all(issymmetric, K.factors)

isposdef(K::KroneckerProduct) = all(isposdef, K.factors)
# KroneckerProduct with factorizations
issuccess(K::KroneckerProduct) = all(issuccess, K.factors)

# to have this as a trait, need to return type
# factorizationtrait
isfactorized(A::AbstractMatrix) = false
isfactorized(A::Factorization) = true
isfactorized(K::KroneckerProduct) = all(isfactorized, K.factors)
# isfactorized(W::Woodbury) = all(isfactorized, (W.A, W.C))
# checks if all component matrices of a KroneckerProduct have the same size
samesize(K::KroneckerProduct) = (all(==(size(K.factors[1])), K.factors), size(K.factors[1]))

# checks if all component matrices are the same
ispower(K::KroneckerProduct) = all(==(K.factors[1]), K.factors)
order(M::AbstractMatrix) = 1
order(K::KroneckerProduct) = length(K.factors)

function logdet(K::KroneckerProduct)
    n = checksquare(K)
    f(A) = logdet(A) * (n ÷ size(A, 1))
    allsquare(K) ? sum(f, K.factors) : real(eltype(K))(-Inf)
    # power optimization integrated:
    # g(A) = (pow * n^(pow-1)) * logdet(A)
    # allsquare(K) ? (ispower(K) ? g(K.A) : sum(f, K.factors)) : real(eltype(K))(-Inf)
end

# specialization for kronecker power
function logdet(K::KroneckerProduct, ispow::Val{true})
    n = checksquare(K)
    pow = order(K)
    (pow * n^(pow-1)) * logdet(K.A)
end
det(K::KroneckerProduct) = exp(logdet(K))

# TODO: meta?
function tr(K::KroneckerProduct)
    n = checksquare(K)
    allsquare(K) ? prod(tr, K.factors) : sum(K[i,i] for i in 1:n)
end

# TODO: delete this?
function inv(K::KroneckerProduct)
    checksquare(K)
    allsquare(K) ? ⊗(inv, K) : throw(SingularException(1))
end

# TODO: checking for square here disallows pseudo-inverse ...
function LazyInverses.inverse(K::KroneckerProduct)
    checksquare(K)
    allsquare(K) ? ⊗(inverse, K) : throw(SingularException(1))
end
LazyInverses.pseudoinverse(K::KroneckerProduct) = ⊗(pseudoinverse, K)

# IDEA: meta programming?
LinearAlgebra.factorize(K::KroneckerProduct) = ⊗(factorize, K)
LinearAlgebra.adjoint(K::KroneckerProduct) = ⊗(adjoint, K)
LinearAlgebra.transpose(K::KroneckerProduct) = ⊗(transpose, K)
LinearAlgebra.conj(K::KroneckerProduct) = ⊗(conj, K)

# useful for least-squares problems
function LinearAlgebra.qr(K::KroneckerProduct, ::Val{true})
    kronecker(A->qr(A, Val(true)), K)
end
function LinearAlgebra.cholesky(K::KroneckerProduct, ::Val{true}; tol = 1e-12, check = true)
    kronecker(A->cholesky(A, Val(true), tol = tol, check = check), K)
end
function LinearAlgebra.cholesky(K::KroneckerProduct, ::Val{false} = Val(false))
    kronecker(A->cholesky(A), K)
end

collect(K::KroneckerProduct) = kron(Matrix.(K.factors)...)
LinearAlgebra.Matrix(K::KroneckerProduct) = collect(K)

# mixed product property
*(K1::KroneckerProduct, K2::KroneckerProduct) = ⊗(K1.factors .* K2.factors)

*(a::Number, K::KroneckerProduct) = ⊗(a * K.factors[1], K.factors[2:end]...)
*(K::KroneckerProduct, a::Number) = ⊗(K.factors[1:end-1], K.factors[end] * a)

function LinearAlgebra.lmul!(a::Number, K::KroneckerProduct)
    lmul!(a, K.factors[1])
end

function LinearAlgebra.rmul!(K::KroneckerProduct, a::Number)
    rmul!(K.factors[end], a)
end

################################ vec trick #####################################
function *(K::KroneckerProduct, x::AbstractVecOrMat)
    size(K, 2) ≠ size(x, 1) && throw(DimensionMismatch("$(size(K, 2)) ≠ $(size(x, 1))"))
    X = x
    for (i, A) in enumerate(reverse(K.factors))
        X = reshape(X, (size(A, 2), :))
        X = adjoint(A * X)
    end
    (x isa AbstractVector) ? vec(X) : reshape(X, (size(x, 2), :))'
end

*(x::Adjoint{<:Number, <:AbstractVector}, K::KroneckerProduct) = adjoint(K'*x')
*(x::AbstractMatrix, K::KroneckerProduct) = adjoint(K'*x')
\(K::KroneckerProduct, x::AbstractVecOrMat) = pseudoinverse(factorize(K)) * x
/(x::AbstractVecOrMat, K::KroneckerProduct) = x * inverse(factorize(K)) # pseudoinverse(K, Val(:R)) * x

function dot(x::AbstractVecOrMat, K::KroneckerProduct, y::AbstractVecOrMat)
    dot(x, K*y) # fallback for now
end

function *(x::AbstractMatrix, K::KroneckerProduct, y::AbstractVecOrMat)
    x*(K*y)
end

function LinearAlgebra.mul!(y::AbstractVector, K::KroneckerProduct, x::AbstractVector,
                            α::Real = 1, β::Real = 0)
    Kx = K*x # could make this more efficient
    @. y = α * Kx + β * y
end

############################### indexing #######################################
# test if components support getindex ...
function Base.getindex(K::KroneckerProduct, i::Integer, j::Integer)
    @boundscheck checkbounds(K, i, j)
    val = one(eltype(K))
    n, m = size(K)
    @inbounds for A in K.factors
        n ÷= size(A, 1); m ÷= size(A, 2);
        val *= A[cld(i, n), cld(j, m)]  # can this be replaced with fld1, mod1?
        i, j = mod1(i, n), mod1(j, m) # or some kind of shifted remainder?
    end
    return val
end

# we can delete this if we subtype AbstractMatrix
function Base.checkbounds(K::KroneckerProduct, i::Integer, j::Integer)
    (1 ≤ i ≤ size(K, 1) && 1 ≤ j ≤ size(K, 2)) || throw(BoundsError(K, [i,j]))
end

end # module
