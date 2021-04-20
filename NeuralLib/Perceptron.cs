using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib
{
    public class Perceptron : INeuron
    {
        public enum ActivationFunctionType { Sigmoid, BipolarSigmoid, HyperbolicTangens, Softmax};
        /* ******************************************************************************************************** */
        /* Блок свойств */
        /* ******************************************************************************************************** */
        public double[] X { get; set; } // массив входных значений нейрона
        public double[] W { get; set; } // двумерный массив коэффициентов нейрона. 
        public double Bias { get; set; }  // массив смещений (bias)
        public double Error { get; set; }
        public double Output { get; set; }
        public double LocalGrad { get; set; }
        public ActivationFunctionType FunctionType { get; set; }
        /* ******************************************************************************************************* */
        /* Блок конструкторов */
        /* ******************************************************************************************************* */
        /* Конструктор принимающий в качестве параметров countIn - число входов, countNeuron - количество нейронов */

        Random rand = new Random();
        public Perceptron()
        { }
        public Perceptron(int numIn, ActivationFunctionType type)
        {
            X = new double[numIn];
            W = new double[numIn];
            Bias = rand.NextDouble();
            for (int i = 0; i < W.Length; i++)
            {
                W[i] = rand.NextDouble();
            }
            FunctionType = type;
        }
        /* Блок методов */
        public double GetLinearOutput() // функция вычисляет выход линейного нейрона
        {
            double linearOut = 0.0;
            for (int i = 0; i < X.Length; i++)
            {
                linearOut = linearOut + X[i] * W[i];
            }
            linearOut += Bias;
            return linearOut;
        }

        public double GetSigmoidOut()
        {
            Output = 1 / (1 + Math.Exp(-GetLinearOutput()));
            return Output;
        }
        public double GetTandOut()
        {
            Output = 1 / (1 + Math.Exp(-GetLinearOutput()));
            return Output;
        }
        public double GetLinearError(double d) // вычислить ошибку сети
        {
            double Error = d - GetLinearOutput();
            return Error;
        }
        public void CalcWeights(double learningRate, double d) // вычислить новые коэффициенты с учетом ошибки
        {
            LocalGrad = 0.0;
            double grad;
           // grad = 1 * (d - Output) * Output * (1 - Output); // local grad for hidden layer
            grad = Output * (1 - Output)*d; // local grad for output layer
            for (int i = 0; i < X.Length; i++)
            {
                LocalGrad += grad * W[i];
            }
            Bias = Bias + learningRate * grad;
            for (int i = 0; i < X.Length; i++)
            {
                W[i] = W[i] + learningRate * grad * X[i];
            }
        }
        public double GetOutput()
        {
            switch (FunctionType)
            {
                case ActivationFunctionType.Sigmoid:
                    {
                        double temp = 0.0;
                        for (int i = 0; i < X.Length; i++)
                        {
                            temp = temp + X[i] * W[i];
                        }
                        temp += Bias;
                        Output = 1 / (1 + Math.Exp(-temp));
                        break;
                    }
                case ActivationFunctionType.BipolarSigmoid:
                    {
                        double temp = 0.0;
                        for (int i = 0; i < X.Length; i++)
                        {
                            temp = temp + X[i] * W[i];
                        }
                        temp += Bias;
                        Output =  (2 / (1 + Math.Exp(-temp))) - 1; // двухполюсный сигмоид
                        break;
                    }
                case ActivationFunctionType.HyperbolicTangens:
                    break;
           }
            return Output;
        }
    }
}