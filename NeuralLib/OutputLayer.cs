using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib
{
    public class OutputLayer : ILayer
    {
        public enum ActivationFunctionType { Linear, Sigmoid, BipolarSigmoid, HyperbolicTangens, Softmax };
        /* ******************************************************************************************************** */
        /* Блок свойств */
        /* ******************************************************************************************************** */
        public int NumNeurons { get; set; } // число нейронов
        public double[] LocalGrad { get; set; } // локальный градиент
        public double[,] W { get; set; } // двумерный массив весовых коэффициентов
        public double[] X { get; set; } // массив входов
        public double[] Output { get; set; } // выход слоя
        public double[] Bias { get; set; } // массив смещений
        public double[] Error { get; set; }
        ActivationFunctionType actFTyp;
        public OutputLayer()
        {
        }
        public OutputLayer(int numIn, int numNeurons, ActivationFunctionType actFunc)
        {
            Random rand = new Random();
            actFTyp = actFunc;
            NumNeurons = numNeurons;
           
            X = new double[numIn];
            Error = new double[NumNeurons];
            Bias = new double[NumNeurons];
            W = new double[NumNeurons, numIn];
            Output = new double[NumNeurons];
            LocalGrad = new double[X.Length];
            //for (int i = 0; i < NumNeurons; i++)
            //{
            //    Bias[i] = rand.NextDouble();
            //    for (int j = 0; j < numIn; j++)
            //    {
            //        W[i, j] = rand.NextDouble();
            //    }
            //}
            Parallel.For(0, NumNeurons, (i) =>
            {
                Bias[i] = rand.NextDouble();
                Parallel.For(0, numIn, (j) =>
                {
                    W[i, j] = rand.NextDouble();
                });
            });
        }
        public double[] GetOutput(double[] input)
        {
            X = input;
          
            double y;
            //for (int i = 0; i < NumNeurons; i++)
            //{
            //    y = 0.0;
            //    for (int j = 0; j < X.Length; j++)
            //    {
            //        y += X[j] * W[i, j];
            //    }
            //    y += Bias[i];
            //    switch (actFTyp)
            //    {
            //        case ActivationFunctionType.Linear:
            //            Output[i] = y;
            //            break;
            //        case ActivationFunctionType.Sigmoid:
            //            Output[i] = 1 / (1 + Math.Exp(-y));
            //            break;
            //        case ActivationFunctionType.HyperbolicTangens:
            //            Output[i] = Math.Tanh(y);
            //            break;
            //        case ActivationFunctionType.BipolarSigmoid:
            //            Output[i] = (2 / (1 + Math.Exp(-y))) - 1;
            //            break;
            //    }
            //}
            for (int i = 0; i < NumNeurons; i++)
            {
                y = 0.0;
                Parallel.For(0, X.Length, (j) =>
                {
                    y += X[j] * W[i, j];
                });

                y += Bias[i];
                switch (actFTyp)
                {
                    case ActivationFunctionType.Linear:
                        Output[i] = y;
                        break;
                    case ActivationFunctionType.Sigmoid:
                        Output[i] = 1 / (1 + Math.Exp(-y));
                        break;
                    case ActivationFunctionType.HyperbolicTangens:
                        Output[i] = Math.Tanh(y);
                        break;
                    case ActivationFunctionType.BipolarSigmoid:
                        Output[i] = (2 / (1 + Math.Exp(-y))) - 1;
                        break;
                }
            }
            return Output;
        }
        public void CalcWeights(double learningRate, double[] d)
        {

            //for (int i = 0; i < NumNeurons; i++)
            //{
            //    switch (actFTyp)
            //    {
            //        case ActivationFunctionType.Linear: // ошибка выходного слоя (линейная функция активации)
            //            Error[i] = d[i] - Output[i];
            //            break;
            //        case ActivationFunctionType.Sigmoid: // ошибка выходного слоя (сигмоидальная функция активации)
            //            Error[i] = (d[i] - Output[i]) * Output[i] * (1 - Output[i]);
            //            break;
            //        case ActivationFunctionType.HyperbolicTangens: // ошибка выходного слоя (функция активации гиперболического тангенса)
            //            Error[i] = (d[i] - Output[i]) * (1 - Output[i]) * (1 + Output[i]);
            //            break;
            //        case ActivationFunctionType.BipolarSigmoid: // ошибка выходного слоя (функция активации биполярного сигмоида)
            //            //Error[i] = d[i] - Output[i];
            //            break;
            //    }
            //    Bias[i] = Bias[i] + learningRate * Error[i];
            //    for (int j = 0; j < X.Length; j++)
            //    {
            //        LocalGrad[j] += Error[i] * W[i, j];
            //        W[i, j] += learningRate * Error[i] * X[j];
            //    }
            //}
            for (int i = 0; i < NumNeurons; i++)
            {
                switch (actFTyp)
                {
                    case ActivationFunctionType.Linear: // ошибка выходного слоя (линейная функция активации)
                        Error[i] = d[i] - Output[i];
                        break;
                    case ActivationFunctionType.Sigmoid: // ошибка выходного слоя (сигмоидальная функция активации)
                        Error[i] = (d[i] - Output[i]) * Output[i] * (1 - Output[i]);
                        break;
                    case ActivationFunctionType.HyperbolicTangens: // ошибка выходного слоя (функция активации гиперболического тангенса)
                        Error[i] = (d[i] - Output[i]) * (1 - Output[i]) * (1 + Output[i]);
                        break;
                    case ActivationFunctionType.BipolarSigmoid: // ошибка выходного слоя (функция активации биполярного сигмоида)
                        //Error[i] = d[i] - Output[i];
                        break;
                }
                Bias[i] = Bias[i] + learningRate * Error[i];
                Parallel.For(0, X.Length, (j) =>
                {
                    LocalGrad[j] += Error[i] * W[i, j];
                    W[i, j] += learningRate * Error[i] * X[j];
                });
            }
        }
    }
}
