module KroneckerProducts
using LinearAlgebra

# TODO: Make own project?
# TODO: kronecker sum: ⊕(A, B) = LazySum(⊗(A, I(checksquare(B))), ⊗(I(checksquare(A)), B))
# struct KroneckerSum{T, A<:Tuple{Vararg{AbstractMatOrFac}}} <: Factorization{T}
#     factors::A
#     # temporary::V
# end
# # this should probably be an inner constructor
# function KroneckerSum(A::Tuple{Vararg{AbstractMatOrFac}})
#     T = promote_type(eltype.(A)...)
#     KroneckerProduct{T, typeof(A), Nothing}(A, nothing)
# end
# ⊕(A::AbstractMatrix...) = KroneckerSum(A)
# TODO:
# Base.getindex(K::KroneckerSum, i, j) = LazySum(⊗(K.A, I(checksquare(K.B))), ⊗(I(checksquare(K.A)), K.B))

# TODO: figure out temporary pre-allocation
# TODO: explore relationship to tensor product and matricification
# For partinioned matrices:
# Katri-Rao product KhatriRaoProduct() khatrirao A * B = (A[i,j] ⊗ B[i,j])[i,j]
# Tracy-Singh product A ∘ B = (A[i,j] .* B)[i,j] = ((A[i,j] ⊗ B[k,l])[k,l])[i,j]
# TODO: multi-ary matrix multiplication in optimal order via dynamic programming
# (also for vec-trick)
# TODO:
# 3) need to test for complex valued kronecker matrices, might need to replace
# some adjoints with transposes in the code.

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}

using LinearAlgebra: checksquare
import LinearAlgebra: det, tr, logdet, issymmetric, ishermitian, isposdef,
adjoint, transpose, conj
import LinearAlgebra: Factorization, factorize, issuccess, Matrix, size
import LinearAlgebra: cholesky, cholesky!, qr, qr!#, svd, eigen, bunchkaufman, lu,
import LinearAlgebra: \, *, /, rdiv!, ldiv!

# depends on Inverse
import LazyInverse: Inverse, inverse, pseudoinverse, PseudoInverse

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
kronecker(f::Function, K::KroneckerProduct) = kronecker((f(A) for A in K.factors)...) #kronecker(f.(K.factors)...)
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

# checks if all component matrices point to the same place in memory
# this yields a "kronecker power"
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
function inverse(K::KroneckerProduct)
    checksquare(K)
    allsquare(K) ? ⊗(inverse, K) : throw(SingularException(1))
end
pseudoinverse(K::KroneckerProduct) = ⊗(pseudoinverse, K)

# TODO: meta programming
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

*(a::Number, K::KroneckerProduct) = kronecker(a * K.factors[1], K.factors[2:end]...)
*(K::KroneckerProduct, a::Number) = kronecker(K.factors[1:end-1], K.factors[end] * a)

function LinearAlgebra.lmul!(a::Number, K::KroneckerProduct)
    lmul!(a, K.factors[1])
end

function LinearAlgebra.rmul!(K::KroneckerProduct, a::Number)
    rmul!(K.factors[end], a)
end

############################ vec trick #########################################
# TODO: pre-allocate intermediaries
# TODO: define mul!
function *(K::KroneckerProduct, x::AbstractVecOrMat)
    size(K, 2) ≠ size(x, 1) && throw(DimensionMismatch("$(size(K, 2)) ≠ $(size(x, 1))"))
    X = x
    for (i, A) in enumerate(reverse(K.factors))
        X = reshape(X, (size(A, 2), :))
        X = (A * X)'
    end
    if x isa AbstractVector
        return vec(X)
    elseif x isa AbstractMatrix
        return reshape(X, (size(x, 2), :))'
    end
end

*(x::Adjoint{<:Number, <:AbstractVector}, K::KroneckerProduct) = (K'*x')'
*(x::AbstractMatrix, K::KroneckerProduct) = (K'*x')'
\(K::KroneckerProduct, x::AbstractVecOrMat) = pseudoinverse(K) * x
/(x::AbstractVecOrMat, K::KroneckerProduct) = x * inverse(K) # pseudoinverse(K, Val(:R)) * x

function dot(x::AbstractVecOrMat, K::KroneckerProduct, y::AbstractVecOrMat)
    dot(x, K*y)
    # would have to define a lazy mul myself, in the flavor of
    # Ky = ApplyVector(applied(*, K, y))
    # dot(x, Ky)
end

function *(x::AbstractVecOrMat, K::KroneckerProduct, y::AbstractVecOrMat)
    x*(K*y)
end

# gets number of elements necessary to pre-allocate all temporary storage
# same structure as vec trick below, but only computes intermediate matrix sizes
# the easiest solution is to create to arrays of this size,
# then pointer swap
function temporary_length(K::KroneckerProduct)
     X = (size(K, 2), 1) # start out as column vector
     M = 1
     for A in reverse(K.factors)
         # X = reshape(X, (size(A, 2), :))
         X = (size(A, 2), prod(X) ÷ size(A, 2))
         # X = (A * X)
         X = (size(A, 1), X[2])
         # X = X'
         X = (X[2], X[1])
         M = max(M, prod(X))
     end
     return M
end

# TODO: since we seem to need two temporaries
# function temporary_lengths(K::KroneckerProduct)
#      X = (size(K, 2), 1) # start out as column vector
#      Y = X
#      N1, N2 = 1, 1
#      for A in reverse(K.factors)
#          # X = reshape(X, (size(A, 2), :))
#          X = (size(A, 2), prod(X) ÷ size(A, 2))
#          # Y = (A * X)
#          Y = (size(A, 1), X[2])
#          # Y = Y'
#          Y = (Y[1], Y[2])
#          N1 = max(N1, prod(X))
#          N2 = max(N2, prod(Y))
#          X, Y = Y, X
#      end
#      return M
# end

# TODO:  preallocate intermediaries, error checking,
# function mul!(y::AbstractVector, K::KroneckerProduct, x::AbstractVector,
#             t1 = zeros())
#     X = x
#     for A in reverse(K.factors)
#         X = reshape(X, (size(A, 2), :))
#         X = (A * X)'
#     end
#     y .=
# end

# ideally, implement
# function mul!(y, K, x, α, β)
#     return -1
# end

# mul!(y::AbstractVector, K::KroneckerProduct, x::AbstractVector)
# mul!(y::AbstractMatrix, K::KroneckerProduct, x::AbstractMatrix)

# function Base.:*(K::KroneckerProduct, v::AbstractVector)
#     return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K, v)
# end

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

############################# factorization ####################################
# TODO: need to define Cholesky constructor for KroneckerProduct ..., does this make sense?
# function cholesky(K::KroneckerProduct; check = true)
#     C = kronecker(A->cholesky(A, check = check), K)
#     uplo = :U
#     info = issuccess(C) ? 0 : -1
#     U = kronecker((A.U for A in C.factors)...) # this is essential for fast ternary multiplication
#     Cholesky(U, uplo, info)
# end

# function *(x::AbstractVecOrMat, K::KroneckerProduct{<:Number, NTuple{<:Cholesky}},
#                                                             y::AbstractVecOrMat)
#     return
# end

# sym = :(cholesky(M::MyMat, ::Val{true} = Val(true)))
# for f in (:factorize, :lu, :qr, :cholesky)
#     @eval LinearAlgebra.$f(x::KroneckerProduct) = KroneckerFactorization($f(x.a), $f(x.b))
# end
#
# _facts = [:cholesky, :cholesky!] # :qr, :qr:]
# # _pivs = [:true, :false]
# _call = :(KroneckerProduct(f(K.A), f(K.B)))
# for fac in _facts
#     for piv in [:true, :false]
#         function $fac(K::KroneckerProduct, ::Val{$piv}; check::Bool = true)
#             f(A) = $fac(A, Val($piv), check = check)
#             $_call
#         end
#     end
# end

# cholesky
# function cholesky(K::KroneckerProduct, ::Val{false}=Val(false); check::Bool = true)
#     f(A) = cholesky(A, Val(false), check = check)
#     KroneckerProduct(f(K.A), f(K.B))
# end
#
# # in place cholesky
# function cholesky!(K::KroneckerProduct, ::Val{false}=Val(false); check::Bool = true)
#     f(A) = cholesky!(A, Val(false), check = check)
#     KroneckerProduct(f(K.A), f(K.B))
# end
#
# # pivoted cholesky
# function cholesky(K::KroneckerProduct, ::Val{true}; tol = 0.0, check::Bool = true)
#     f(A) = cholesky(A, Val(true), tol = tol, check = check)
#     KroneckerProduct(f(K.A), f(K.B))
# end
#
# # in-place pivoted cholesky
# function cholesky!(K::KroneckerProduct, ::Val{true}; tol = 0.0, check::Bool = true)
#     f(A) = cholesky!(A, Val(true), tol = tol, check = check)
#     KroneckerProduct(f(K.A), f(K.B))
# end
#
# # qr
# function qr(K::KroneckerProduct, v::V = Val(false)) where {V<:Union{Val{true}, Val{false}}}
#     f(A) = qr(A, v)
#     KroneckerProduct(f(K.A), f(K.B))
# end
#
# # in-place qr
# function qr!(K::KroneckerProduct, v::V = Val(false)) where {V<:Union{Val{true}, Val{false}}}
#     f(A) = qr!(A, v)
#     KroneckerProduct(f(K.A), f(K.B))
# end

end

# TODO: getindex when we don't subtype abstract matrix
# function Base.getindex(K::KroneckerProduct, ::Colon, j::Integer)
#     @boundscheck checkbounds(K, 1, j)
#     val = ones(eltype(K), size(K, 1))
#     n, m = size(K)
#     @inbounds for A in K.factors
#         n, m = m ÷ size(A, 2) #(n, m) ./ size(A)
#         val .*= A[:, cld(j, m)]  # can this be replaced with fld1, mod1?
#         j = mod1(j, m)
#     end
#     return val
# end
# Base.getindex(K::KroneckerProduct, i::Integer, ::Colon) = (K'[:, i])'
# Base.getindex(K::KroneckerProduct, ::Colon, Colon) = Matrix(K)

# """
#     tr(K::KroneckerPower)
#
# Compute the trace of a Kronecker power.
# """
# function LinearAlgebra.tr(K::KroneckerPower)
#     checksquare(K.A)
#     return tr(K.A)^K.pow
# end
#
# # Matrix operations
#
# """
#     inv(K::KroneckerPower)
#
# Compute the inverse of a Kronecker power.
# """
# function Base.inv(K::KroneckerPower)
#     checksquare(K.A)
#     return KroneckerPower(inv(K.A), K.pow)
# end
#
# """
#     adjoint(K::KroneckerPower)
#
# Compute the adjoint of a Kronecker power.
# """
# function Base.adjoint(K::KroneckerPower)
#     return KroneckerPower(K.A', K.pow)
# end
#
# """
#     transpose(K::KroneckerPower)
#
# Compute the transpose of a Kronecker power.
# """
# function Base.transpose(K::KroneckerPower)
#     return KroneckerPower(transpose(K.A), K.pow)
# end
#
# """
#     conj(K::KroneckerPower)
#
# Compute the conjugate of a Kronecker power.
# """
# function Base.conj(K::KroneckerPower)
#     return KroneckerPower(conj(K.A), K.pow)
# end
#
# # mixed-product property
# function Base.:*(K1::KroneckerPower{T,TA,N},
#                         K2::KroneckerPower{S,TB,N}) where {T,TA,S,TB,N}
#     if size(K1, 2) != size(K2, 1)
#         throw(DimensionMismatch("Mismatch between K1 and K2"))
#     end
#     return KroneckerPower(K1.A * K2.A, N)
# end

# function \(K::KroneckerProduct, x::AbstractVector)
#     X = x
#     for A in reverse(K.factors)
#         X = reshape(X, (size(A, 1), :))
#         X = (A \ X)'
#     end
#     return vec(X)
# end

# recursive
# function *(K::KroneckerProduct, c::AbstractVector)
#     A, B = getmatrices(K)
#     C = reshape(c, (size(B, 1), size(A, 1)))
#     x = vec((A \ (B \ C)')')
# end
