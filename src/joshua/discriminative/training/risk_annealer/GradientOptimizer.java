package joshua.discriminative.training.risk_annealer;

import joshua.discriminative.training.lbfgs.LBFGSWrapper;

/**
 * @author Zhifei Li, <zhifei.work@gmail.com>
 */
public class GradientOptimizer extends LBFGSWrapper {
  GradientComputer gradientComputer;

  public GradientOptimizer(int numPara, double[] initWeights, boolean isMinimizer,
      GradientComputer gradientComputer, boolean useL2Regula, double varianceForL2,
      boolean useModelDivergenceRegula, double lambda, int printFirstN) {
    super(numPara, initWeights, isMinimizer, useL2Regula, varianceForL2, useModelDivergenceRegula,
        lambda, printFirstN);
    this.gradientComputer = gradientComputer;
  }

  // IMPLEMENTING THE ABSTRACT SUPER CLASS FUNCTION
  public double[] computeFuncValAndGradient(double[] curWeights, double[] resFuncVal) {

    // COMPUTE THE GRADIENT AND FUNCTION VALUE
    gradientComputer.reComputeFunctionValueAndGradient(curWeights);

    resFuncVal[0] = gradientComputer.getLatestFunctionValue();
    return gradientComputer.getLatestGradient();
  }
}
