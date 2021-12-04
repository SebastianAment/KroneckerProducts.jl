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

export kronecker, ⊗, KroneckerProduct, FactorizedKroneckerProduct

# has field for temporary storage needed for non-allocating multiplies and solves
struct KroneckerProduct{T, A<:Tuple, V} <: Factorization{T}
    factors::A
    temporaries::V
end
const FactorizedKroneckerProduct{T} = KroneckerProduct{T, <:Tuple{Vararg{Factorization}}}

Base.eltype(K::KroneckerProduct{T}) where {T} = T
Base.IndexStyle(::Type{<:KroneckerProduct}) = IndexCartesian()

function KroneckerProduct(A::Tuple{Vararg{Union{Number, AbstractMatOrFac}}})
    T = promote_type(eltype.(A)...)
    temporaries = initialize_temporaries(A, T)
    KroneckerProduct{T, typeof(A), typeof(temporaries)}(A, temporaries)
end

function kronecker end # smart constructor
const ⊗ = kronecker
kronecker(A::Union{Number, AbstractMatOrFac}) = A
kronecker(A::Union{Number, AbstractMatOrFac}...) = KroneckerProduct(A)
kronecker(A::Tuple) = KroneckerProduct(A)
kronecker(K::KroneckerProduct) = K

# IDEA: this makes sure we have a flat Kronecker hierarchy? could have separate "flatten" function
# only makes sense to break if we have a subset, which is a power
kronecker(A::Union{Number, AbstractMatOrFac}, K::KroneckerProduct) = kronecker(A, K.factors...)
kronecker(K::KroneckerProduct, A::Union{Number, AbstractMatOrFac}) = kronecker(A, K)
kronecker(K::KroneckerProduct, L::KroneckerProduct) = kronecker(K.factors..., L.factors...)

# create new product by applying function to each component matrix
kronecker(f, K::KroneckerProduct) = kronecker((f(A) for A in K.factors)...) #kronecker(f.(K.factors)...)

kronecker_power(A, pow::Integer) = kronecker((A for _ in 1:pow)...)

Base.size(K::KroneckerProduct) = (size(K, 1), size(K, 2))
Base.size(K::KroneckerProduct, dim::Integer) = prod(A->size(A, dim), K.factors)

function collect(K::KroneckerProduct)
    f(A) = A isa Factorization ? AbstractMatrix(A) : A
    kron(f.(K.factors)...)
end
LinearAlgebra.Matrix(K::KroneckerProduct) = collect(K)

issquare(A::AbstractMatOrFac) = size(A, 1) == size(A, 2)
# IDEA: meta
allsquare(K::KroneckerProduct) = all(issquare, K.factors)
issymmetric(K::KroneckerProduct) = all(issymmetric, K.factors)
isposdef(K::KroneckerProduct) = all(isposdef, K.factors)
# KroneckerProduct with factorizations
issuccess(K::KroneckerProduct) = all(issuccess, K.factors)

# to have this as a trait, need to return type
# factorizationtrait
isfactorized(A::AbstractMatrix) = false
isfactorized(A::Factorization) = true
isfactorized(K::FactorizedKroneckerProduct) = true
isfactorized(K::KroneckerProduct) = false

# isfactorized(W::Woodbury) = all(isfactorized, (W.A, W.C))
# checks if all component matrices of a KroneckerProduct have the same size
samesize(K::KroneckerProduct) = samesize(K.factors)
samesize(A...) = samesize(A)
samesize(A::Tuple) = all(==(size(A[1])), A)

# checks if all component matrices are the same
ispower(K::KroneckerProduct) = isequiv(K) || all(==(K.factors[1]), K.factors)
# same as ispower but only does pointer check (constant for each factor)
isequiv(K::KroneckerProduct) = all(F -> F === K.factors[1], K.factors)
order(M::AbstractMatrix) = 1
order(K::KroneckerProduct) = length(K.factors)

function LinearAlgebra.logdet(K::KroneckerProduct)
    n = checksquare(K)
    if !allsquare(K) # implies that there is a rank deficient factor
        real(eltype(K))(-Inf)
    elseif isequiv(K) # if all factors are equivalent (have same pointer), use more efficient formula
        p = length(K.factors)
        A = K.factors[1]
        k = size(A, 1)
        p * (n ÷ k) * logdet(A)
    else
        f(A) = logdet(A) * (n ÷ size(A, 1))
        sum(f, K.factors)
    end
end

LinearAlgebra.det(K::KroneckerProduct) = exp(logdet(K))

# TODO: meta?
function LinearAlgebra.tr(K::KroneckerProduct)
    n = checksquare(K)
    allsquare(K) ? prod(tr, K.factors) : sum(K[i, i] for i in 1:n)
end

# TODO: delete this?
function Base.inv(K::KroneckerProduct)
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
function LinearAlgebra.qr(K::KroneckerProduct, ::NoPivot = NoPivot())
    kronecker(A->qr(A, NoPivot()), K)
end
function LinearAlgebra.qr(K::KroneckerProduct, ::Val{true})
    kronecker(A->qr(A, Val(true)), K)
end
function LinearAlgebra.cholesky(K::KroneckerProduct, ::Val{false} = Val(false); check = true)
    kronecker(A->cholesky(A, check = check), K)
end
function LinearAlgebra.cholesky(K::KroneckerProduct, ::Val{true}; tol = 1e-12, check = true)
    kronecker(A->cholesky(A, Val(true), tol = tol, check = check), K)
end

############################ multiplication ####################################
# mixed product property
function *(K1::KroneckerProduct, K2::KroneckerProduct)
    if all((A, B) -> size(A) == size(B), zip(K1.factors, K2.factors)) # if the components have the same sizes
        kronecker(K1.factors .* K2.factors)
    else # fallback
        Matrix(K1) * Matrix(K2)
    end
end

*(a::Number, K::KroneckerProduct) = ⊗(a * K.factors[1], K.factors[2:end]...)
*(K::KroneckerProduct, a::Number) = ⊗(K.factors[1:end-1], K.factors[end] * a)

function LinearAlgebra.lmul!(a::Number, K::KroneckerProduct)
    lmul!(a, K.factors[1])
end

function LinearAlgebra.rmul!(K::KroneckerProduct, a::Number)
    rmul!(K.factors[end], a)
end

################################ vec trick #####################################
function *(K::KroneckerProduct, x::AbstractVector)
    T = promote_type(eltype(K), eltype(x))
    y = zeros(T, size(K, 1))
    mul!(y, K, x)
end
function *(K::KroneckerProduct, X::AbstractMatrix)
    T = promote_type(eltype(K), eltype(X))
    Y = zeros(T, size(K, 1), size(X, 2))
    mul!(Y, K, X)
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

function LinearAlgebra.mul!(y::AbstractVecOrMat, K::KroneckerProduct, x::AbstractVecOrMat,
                            α::Real = 1, β::Real = 0)
    size(K, 2) ≠ size(x, 1) && throw(DimensionMismatch("$(size(K, 2)) ≠ $(size(x, 1))"))
    X = x
    temporaries = get_temporaries(K, y)
    for (A, z) in Iterators.reverse(zip(K.factors, temporaries))
        X = reshape(X, size(A, 2), :)
        Z = reshape(z, size(A, 1), :)
        mul!(Z, A, X)
        X = Z # pointer swap
        X = X' # taking adjoint very expensive!
    end
    Kx = (x isa AbstractVector) ? vec(X) : reshape(X, (size(x, 2), :))'
    @. y = α * Kx + β * y # IDEA: could fold this into last iteration of loop
end

# returns temporaries necessary for efficient multiplication y = K*x with KroneckerProduct
function get_temporaries(K::KroneckerProduct, y::AbstractVecOrMat)
    temporaries = K.temporaries
    T = eltype(temporaries[1])
    # need to allocate temporaries if size is not compatible, or element type differs
    if isnothing(temporaries) || !(length(y) == length(temporaries[1])) || T != eltype(y)
        temporaries = initialize_temporaries(K, y)
    end
    return temporaries
end

# intializes temporaries for multiplication and solve,
# if external vectors have different element types, need to either
# fall back to allocating multiplication, or throw error
# y is assumed to be pre-allocated result for mul!(y, K, x)
function initialize_temporaries(K::KroneckerProduct, y::AbstractVecOrMat)
    initialize_temporaries(K.factors, eltype(y), size(y, 2))
end
function initialize_temporaries(factors::Tuple, T = promote_type(eltype.(A)...), num_cols::Int = 1)
    n = prod(A->size(A, 2), factors) * num_cols
    temporaries = []
    if samesize(factors) # if all A have the same size only need two temporaries that we alternate
        t1, t2 = zeros(T, n), zeros(T, n)
        for _ in factors
            push!(temporaries, t1)
            t1, t2 = t1, t2
        end
    else
        for A in Iterators.reverse(factors)
            k = size(A, 1)
            m = n ÷ size(A, 2)
            push!(temporaries, zeros(T, k * m))
            n = k * m
        end
    end
    return tuple(Iterators.reverse(temporaries)...)
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
