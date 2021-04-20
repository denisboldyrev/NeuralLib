using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib
{
    public class LinearNeuron : INeuron
    {
        /* ******************************************************************************************************** */
        /* Блок свойств */
        /* ******************************************************************************************************** */
        public double[] X { get; set; } // массив входных значений нейрона
        public double[] W { get; set; } // двумерный массив коэффициентов нейрона. 
        public double Output { get; set; } //  линейный выход сети 
        public double Error { get; set; }
        public double LocalGrad { get; set; }
        public double Bias { get; set; } // массив смещений

        /* ******************************************************************************************************** */
        /* Блок конструкторов */
        /* ******************************************************************************************************** */

        /* Конструктор принимающий в качестве параметров countIn - число входов, countNeuron - количество нейронов */
        public LinearNeuron(int numIn)
        {
            X = new double[numIn];
            W = new double[numIn];
        }
        /* Блок методов */
        public double GetOutput() // функция вычисляет выход линейного нейрона
        {
            Output = 0;
            for (int i = 0; i < X.Length; i++)
            {
                Output = Output + X[i] * W[i];
            }
            return Output;
        }
        public void CalcWeights(double a, double d) // вычислить новые коэффициенты с учетом ошибки
        {
            Error = d - Output;
            double[] temp = new double[W.Length];
            for (int i = 0; i < W.Length; i++)
                temp[i] = W[i] + a * X[i] * Error;
            this.W = temp;
        }
    }
}