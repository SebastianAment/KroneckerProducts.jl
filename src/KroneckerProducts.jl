module KroneckerProducts
using LinearAlgebra
using LinearAlgebra: checksquare
import LinearAlgebra: \, *, /, rdiv!, ldiv!
include("lu.jl")
using LazyInverses

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
LinearAlgebra.factorize(F::Factorization) = F

export kronecker, ⊗, KroneckerProduct, FactorizedKroneckerProduct, kronecker_power

# has field for temporary storage needed for non-allocating multiplies and solves
# currently made struct mutable and automatically store latest temporary in structure
# IDEA: could allocate two potentially large arrays for MatMatMul,
# and create temporary views for MatVecMul
mutable struct KroneckerProduct{T, A<:Tuple, V} <: Factorization{T}
    factors::A
    temporaries::V
end
const FactorizedKroneckerProduct{T} = KroneckerProduct{T, <:Tuple{Vararg{Factorization}}}

Base.eltype(K::KroneckerProduct{T}) where {T} = T
Base.IndexStyle(::Type{<:KroneckerProduct}) = IndexCartesian()

function KroneckerProduct(A::Tuple{Vararg{Union{Number, AbstractMatOrFac}}})
    T = promote_type(eltype.(A)...)
    temporaries = nothing
    T_temp = Union{Nothing, NTuple{length(A), Vector}}
    KroneckerProduct{T, typeof(A), T_temp}(A, temporaries)
end

# num_cols is number of columns to be pre-allocated for mul!, if only matrix-vector
# multiplies are required, num_cols = 1 is sufficient
# eltype_temp is the element type of the temporaries to be allocate
function KroneckerProduct(A::Tuple{Vararg{Union{Number, AbstractMatOrFac}}},
                          num_cols::Int, eltype_temp = promote_type(eltype.(A)...))
    temporaries = initialize_temporaries(A, eltype_temp, num_cols)
    T = promote_type(eltype.(A)...)
    KroneckerProduct{T, typeof(A), typeof(temporaries)}(A, temporaries)
end

function kronecker end # smart constructor
const ⊗ = kronecker
kronecker(A::Union{Number, AbstractMatOrFac}) = A
kronecker(A::Union{Number, AbstractMatOrFac}...) = KroneckerProduct(A)
kronecker(A::Tuple) = KroneckerProduct(A)
kronecker(K::KroneckerProduct) = K

function Base.copy(K::KroneckerProduct)
    factors = tuple(copy.(K.factors)...)
    temporaries = isnothing(K.temporaries) ? nothing : tuple(copy.(K.temporaries)...)
    typeof(K)(factors, temporaries)
end

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
    order(K) == 1 ? f(K.factors[1]) : kron(f.(K.factors)...)
end
LinearAlgebra.Matrix(K::KroneckerProduct) = collect(K)

issquare(A::AbstractMatOrFac) = size(A, 1) == size(A, 2)
# IDEA: meta programming
allsquare(K::KroneckerProduct) = all(issquare, K.factors)
LinearAlgebra.issymmetric(K::KroneckerProduct) = all(issymmetric, K.factors)
LinearAlgebra.isposdef(K::KroneckerProduct) = all(isposdef, K.factors)
# KroneckerProduct with factorizations
LinearAlgebra.issuccess(K::KroneckerProduct) = all(issuccess, K.factors)

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

function LinearAlgebra.det(K::KroneckerProduct)
    logabsdet_K, sign_K = logabsdet(K)
    exp(logabsdet_K) * sign_K
end
function LinearAlgebra.logdet(K::KroneckerProduct)
    logabsdet_K, sign_K = logabsdet(K)
    if sign_K ≈ 1
        logabsdet_K
    else
        throw(DomaingError("determinant non-positive: sign = $sign_K"))
    end
end
function LinearAlgebra.logabsdet(K::KroneckerProduct)
    n = checksquare(K)
    if !allsquare(K) # implies that there is a rank deficient factor
        real(eltype(K))(-Inf), zero(eltype(K))
    elseif isequiv(K) # if all factors are equivalent (have same pointer), use more efficient formula
        A = K.factors[1]
        logabsdet_A, sign_A = logabsdet(A)
        p = length(K.factors)
        k = p * (n ÷ size(A, 1))
        k * logabsdet_A, sign_A^k
    else
        logabsdet_K = zero(eltype(K))
        sign_K = one(eltype(K))
        for A in K.factors
            logabsdet_A, sign_A = logabsdet(A)
            k = n ÷ size(A, 1)
            logabsdet_K += k * logabsdet_A
            sign_K *= sign_A^k
        end
        return logabsdet_K, sign_K
    end
end

function LinearAlgebra.tr(K::KroneckerProduct)
    n = checksquare(K)
    allsquare(K) ? prod(tr, K.factors) : sum(K[i] for i in diagind(K))
end

# TODO: delete this?
function Base.inv(K::KroneckerProduct)
    checksquare(K)
    allsquare(K) ? ⊗(inv, K) : throw(SingularException(1))
end

function LazyInverses.inverse(K::KroneckerProduct)
    checksquare(K)
    allsquare(K) ? ⊗(inverse, K) : throw(SingularException(1))
end
LinearAlgebra.pinv(K::KroneckerProduct) = ⊗(pinv, K)
LazyInverses.pseudoinverse(K::KroneckerProduct) = ⊗(pseudoinverse, K)

LinearAlgebra.factorize(K::KroneckerProduct) = ⊗(factorize, K)
LinearAlgebra.adjoint(K::KroneckerProduct) = ⊗(adjoint, K)
LinearAlgebra.transpose(K::KroneckerProduct) = ⊗(transpose, K)
LinearAlgebra.conj(K::KroneckerProduct) = ⊗(conj, K)

# useful for least-squares problems
function LinearAlgebra.qr(K::KroneckerProduct, piv = NoPivot())
    kronecker(A->qr(A, piv), K)
end
function LinearAlgebra.cholesky(K::KroneckerProduct, piv = Val(false); check = true)
    kronecker(A->cholesky(A, piv, check = check), K)
end

############################ multiplication ####################################
# mixed product property
function *(K1::KroneckerProduct, K2::KroneckerProduct)
    size(K1, 2) == size(K2, 1) || throw(DimensionMismatch("size(K1, 2) = $(size(K1, 2)) ≠ $(size(K2, 1)) = size(K2, 1)"))
    if all((AB) -> size(AB[1]) == size(AB[2]), zip(K1.factors, K2.factors)) # if the components have the same sizes
        kronecker(K1.factors .* K2.factors)
    else # fallback
        Matrix(K1) * Matrix(K2)
    end
end

*(a::Number, K::KroneckerProduct) = ⊗(a * K.factors[1], K.factors[2:end]...)
*(K::KroneckerProduct, a::Number) = ⊗(K.factors[1:end-1]..., K.factors[end] * a)

function LinearAlgebra.lmul!(a::Number, K::KroneckerProduct)
    lmul!(a, K.factors[1])
    return K
end

function LinearAlgebra.rmul!(K::KroneckerProduct, a::Number)
    rmul!(K.factors[end], a)
    return K
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
/(x::AbstractVecOrMat, K::KroneckerProduct) = x * pseudoinverse(factorize(K))

function LinearAlgebra.dot(x::AbstractVecOrMat, K::KroneckerProduct, y::AbstractVecOrMat)
    dot(x, K*y) # fallback for now
end

function *(x::AbstractMatrix, K::KroneckerProduct, y::AbstractVecOrMat)
    x*(K*y)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, K::KroneckerProduct, x::AbstractVecOrMat,
                            α::Real = 1, β::Real = 0)
    _mul_check_args(y, K, x)
    X = x
    temporaries = set_temporaries!(K, x, y)
    for (A, z) in Iterators.reverse(zip(K.factors, temporaries)) # this loop takes up the majority of the runtime
        X = reshape(X, size(A, 2), :)
        Z = reshape(z, :, size(A, 1))
        mul!(Z, transpose(X), transpose(A)) # same as Z = transpose(A*X)
        X = Z # pointer change
    end
    Kx = (x isa AbstractVector) ? vec(X) : transpose(reshape(X, (size(x, 2), :)))
    @. y = α * Kx + β * y
end

function _mul_check_args(y::AbstractVecOrMat, K::KroneckerProduct, x::AbstractVecOrMat)
    size(K, 2) ≠ size(x, 1) && throw(DimensionMismatch("size(K, 2) = $(size(K, 2)) ≠ $(size(x, 1)) = size(x, 1)"))
    size(K, 1) ≠ size(y, 1) && throw(DimensionMismatch("size(K, 1) = $(size(K, 1)) ≠ $(size(y, 1)) = size(y, 1)"))
    size(x, 2) ≠ size(y, 2) && throw(DimensionMismatch("size(x, 2) = $(size(x, 2)) ≠ $(size(y, 2)) = size(y, 2)"))
    return true
end

# returns temporaries necessary for efficient multiplication y = K*x with KroneckerProduct
function set_temporaries!(K::KroneckerProduct, x::AbstractVecOrMat, y::AbstractVecOrMat)
    temporaries = K.temporaries
    allocated = !isnothing(temporaries)
    if allocated
        correct_type = eltype(temporaries[1]) == eltype(y) # is a little more restrictive than necessary, i.e. real could be cast into complex
        correct_length = length(y) == length(temporaries[1])
        allocated = correct_type && correct_length
    end
    # need to allocate temporaries if size is not compatible, or element type differs
    if !allocated
        temporaries = initialize_temporaries(K, y)
    end
    return K.temporaries = temporaries
end

# allocates temporaries for multiplication and solve,
# if external vectors have different element types or different sizes,
# need to allocate new temporaries
# does not allocate new temporaries if existing ones satisfy the type and size
# requirement for the current multiplication
function initialize_temporaries(K::KroneckerProduct, y::AbstractVecOrMat)
    initialize_temporaries(K.factors, eltype(y), size(y, 2))
end
function initialize_temporaries(factors::Tuple, T = promote_type(eltype.(A)...), num_cols::Int = 1)
    # first, calculate maximum length of temporary array
    n = prod(A->size(A, 2), factors) * num_cols
    max_n = n
    for A in Iterators.reverse(factors)
        n = size(A, 1) * (n ÷ size(A, 2))
        max_n = max(n, max_n)
    end
    # second, allocate two arrays of this size and create views into them alternatingly
    t1, t2 = zeros(T, max_n), zeros(T, max_n) # only need to allocate two sufficiently large arrays
    temporaries = []
    n = prod(A->size(A, 2), factors) * num_cols
    for (i, A) in enumerate(Iterators.reverse(factors))
        n = size(A, 1) * (n ÷ size(A, 2))
        t = @view t1[1:n]
        # @time temporaries[i] = t
        push!(temporaries, t)
        t1, t2 = t2, t1
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
