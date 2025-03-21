#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    void transMat(Shape &shape, bool trans) {
        if(trans){
            int size = shape.size();
            int temp = shape[size - 1];
            shape[size - 1] = shape[size - 2];
            shape[size - 2] = temp;
        }
        
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Tensor A = inputs[0];
        Tensor B = inputs[1];
        Shape aShape = A->getDims();
        Shape bShape = B->getDims();
        infini::transMat(aShape, transA);
        infini::transMat(bShape, transB);
        aShape[aShape.size() - 1] = bShape[aShape.size() - 1];

        vector<Shape> ans = {aShape};

        return ans;
    }

} // namespace infini