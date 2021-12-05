# TODO: these definitions should be in Base
function LinearAlgebra.ldiv!(A::LU{<:Any,<:StridedMatrix}, B::AbstractVecOrMat)
    _apply_ipiv_rows!(A, B)
    ldiv!(UpperTriangular(A.factors), ldiv!(UnitLowerTriangular(A.factors), B))
end

_apply_ipiv_rows!(A::LU, B::AbstractVecOrMat) = _ipiv_rows!(A, 1 : length(A.ipiv), B)
_apply_inverse_ipiv_rows!(A::LU, B::AbstractVecOrMat) = _ipiv_rows!(A, length(A.ipiv) : -1 : 1, B)

function _ipiv_rows!(A::LU, order::OrdinalRange, B::AbstractVecOrMat)
    for i = order
        if i != A.ipiv[i]
            _swap_rows!(B, i, A.ipiv[i])
        end
    end
    B
end
function _swap_rows!(B::AbstractVector, i::Integer, j::Integer)
    B[i], B[j] = B[j], B[i]
    B
end

function _swap_rows!(B::AbstractMatrix, i::Integer, j::Integer)
    for col = 1 : size(B, 2)
        B[i,col], B[j,col] = B[j,col], B[i,col]
    end
    B
end

LinearAlgebra.transpose(C::Cholesky) = Cholesky(conj(C.U))
LinearAlgebra.transpose(C::CholeskyPivoted) = CholeskyPivoted(conj(C.U), C.piv)
