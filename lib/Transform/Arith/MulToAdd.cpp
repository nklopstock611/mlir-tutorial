#include "lib/Transform/Arith/MulToAdd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DEF_MULTOADD
#include "lib/Transform/Arith/Passes.h.inc"

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;
using arith::ShLIOp;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);

    // canonicalization patterns ensure the constant is on the right, if there
    // is a constant See
    // https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;

    if (!is_power_of_two) {
      return failure();
    }

    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value / 2));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};

/**
 * EXERCISE: When multiplying by a power of two, replace it with an
 * appropriate left-shift op instead. Browse the arith dialect docs to
 * find the right op.
*/

// Do y=C*x like x << log2(C), if C is a power of 2. Else do nothing.
struct PowerOfTwoLeftShift : public OpRewritePattern<MulIOp> {
  PowerOfTwoLeftShift(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);

    // canonicalization patterns ensure the constant is on the right, if there
    // is a constant See
    // https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;

    if (!is_power_of_two) {
      return failure();
    }

    // log base 2 of value to get the shift amount.
    int64_t shift_amount = 0;
    while (value >>= 1) { // shift right until value is 0 -> basically a log2 with bit ops
      shift_amount++;
    }

    // creates a new constant with the shift amount.
    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), shift_amount));

    // creates a new left shift operation with the left operand and the new constant.
    ShLIOp newSHLI = rewriter.create<ShLIOp>(op.getLoc(), lhs, newConstant);

    // replaces the original multiplication with the new left shift.
    rewriter.replaceOp(op, newSHLI);
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};

// Replace y = 9*x with y = 8*x + x
struct PeelFromMul : public OpRewritePattern<MulIOp> {
  PeelFromMul(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}
    
  /**
   * Because it inherits from OpRewritePattern<MulIOp>, it means it's
   * designed to apply a transformation to integer multiplication operations.
   * 
   * The contructor initializes the pattern with the context and the benefit = 1.
   * The benefit is a heuristic that the pattern matcher uses to determine which
   * pattern to apply when multiple patterns match the same operation.It gives a
   * sense of priority.
  */

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    /**
     * This is the most important method! It is invoked automatically by the
     * rewrite engine when it finds an operation that matches the pattern.
    */

		/**
		 * Extraction of the two main operands of the multiplication.
     * Canonicalization patterns ensure the constant is on the right, if there
     * is a constant :: (addi 9, x) -> (addi x, 9)
     * See: https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
		*/
		
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // check if the right operand is a constant.
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();

    /**
     * We are guaranteed `value` is not a power of two, because the greedy
     * rewrite engine ensures the PowerOfTwoExpand pattern is run first, since
     * it has higher benefit.
    */

    // creates a new constant with value - 1.
    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value - 1));

    // creates a new multiplication with the left operand and the new constant.
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);

    // creates a new addition with the new multiplication and the left operand.
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, lhs);

    // replaces the original multiplication with the new addition.
    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};

struct MulToAdd : impl::MulToAddBase<MulToAdd> {
  using MulToAddBase::MulToAddBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowerOfTwoExpand>(&getContext());
    patterns.add<PeelFromMul>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace tutorial
} // namespace mlir
