#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    vector<int> getTransposePermute(Operator op) {
        std::cout<<"get premute"<<std::endl;
        auto transposeObj =  std::dynamic_pointer_cast<TransposeObj>(op);
        return transposeObj->getPermute();
    }

    bool isTrans(vector<int> permute) {
        std::cout<<"chekc permute"<<std::endl;
        // 检查是不是交换后两个维度。
        int size = permute.size();
        vector<int> expectedPermute(size, 0);
        for(int i = 0; i < size; i++) {
            expectedPermute[i] = i;
        }
        expectedPermute[size - 1] = size - 2;
        expectedPermute[size - 2] = size - 1;
        return expectedPermute == permute;
    }

    void GraphObj::removeOp(Operator &op) {
        for(auto &input: op->getInputs()) {
            if(input) {
                // 删除 input的target
                input->removeTarget(op);
                auto succOps = op->getSuccessors();
                for(auto sucOp: succOps) {
                    // input的target变成 op的后继
                    input->addTarget(sucOp);
                    // op后继的前驱不在是op
                    sucOp->removePredecessors(op);
                    // op后继的输入变成op的输入
                    for(auto &output: op->getOutputs()){
                        sucOp->replaceInput(output, input);
                    }
                    
                    // op的后继的前驱变成op的前驱
                    for(auto preOps: op->getPredecessors()) {
                        sucOp->addPredecessors(preOps);
                    }
                }
            }
        }

        for(auto &output: op->getOutputs()) {
            if(output) {
                removeTensor(output);
            }
        }
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        // 1. 如果相邻的transpose的perm相同则可以删除
        // 2. 如果矩阵乘的上一个算子是transpose且transpose交换的是后两个维度，则transpose可以删除。
        int opSize = ops.size();
        int cur = 1;
        vector<Operator> needRemoveOperators;
        while(cur < opSize) {
            int pre = cur - 1;
            OpType curOpType = ops[cur]->getOpType();
            OpType preOpType = ops[pre]->getOpType();
            if( curOpType == OpType::Transpose && preOpType == OpType::Transpose) {
                vector<int> curPermte = infini::getTransposePermute(ops[cur]);
                vector<int> prePermte = infini::getTransposePermute(ops[pre]);
                if(curPermte == prePermte) {
                    // 删除两个transpose
                    needRemoveOperators.emplace_back(ops[pre]);
                    needRemoveOperators.emplace_back(ops[cur]);
                    removeOp(ops[pre]);
                    removeOp(ops[cur]);
                    cur += 1;
                }

            }else if(curOpType == OpType::MatMul && preOpType == OpType::Transpose) {
                vector<int> prePermte = infini::getTransposePermute(ops[pre]);
                if(infini::isTrans(prePermte)){
                    if(ops[cur]->getInputs()[0]->getSource() == ops[pre]) {
                        needRemoveOperators.emplace_back(ops[pre]);
                        auto matmulObj =  std::dynamic_pointer_cast<MatmulObj>(ops[cur]);
                        matmulObj->setTransA(true);
                        removeOp(ops[pre]);


                    }else if(ops[cur]->getInputs()[1]->getSource() == ops[pre]) {
                        needRemoveOperators.emplace_back(ops[pre]);
                        auto matmulObj =  std::dynamic_pointer_cast<MatmulObj>(ops[cur]);
                        matmulObj->setTransB(true);
                        removeOp(ops[pre]);
                    }
                }

            }
            cur += 1;
        }
        // 删除op
        for(auto op: needRemoveOperators) {
            removeOperator(op);
        }


    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        allocator.info();
        // 分配内存，先只算偏移量
        vector<size_t> offsets;
        for(auto tensor: tensors) {
            size_t offset = allocator.alloc(tensor->getBytes());
            offsets.push_back(offset);
            std::cout<<">>>>>>>tensor: "<<tensor<<std::endl;
        }
        // 统一分配全部内存
        auto start = reinterpret_cast<char*>(allocator.getPtr());
        std::cout<<"start:"<<start<<std::endl;
        for(auto off: offsets){
            std::cout<<"off: "<<off<<std::endl;
        }
        // 绑定内存
        for(size_t i = 0; i < offsets.size(); i++){
            Blob blob = make_ref<BlobObj>(runtime, start + offsets[i]);
            tensors[i]->setDataBlob(blob);
        }
        allocator.info();

        

    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini