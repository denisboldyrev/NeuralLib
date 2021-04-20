using System;
using System.Threading.Tasks;

namespace NeuralLib
{
    public class Layer : Object
    {
        /* ******************************************************************************************************** */
        /* Блок свойств */
        /* ******************************************************************************************************** */
        public double[] LocalGrad { get; set; } // локальный градиент
       // public double LocalGrad { get; set; } // локальный градиент
        public double [,] W { get; set; } // двумерный массив весовых коэффициентов
        public double [] X { get; set; } // массив входов
        public double [] Output { get; set; } // выход слоя
        public int NumNeurons { get; set; } // число нейронов
        public double [] OutError { get; set; } // ошибка выходного слоя
        public double [] OutHError { get; set; } // ошибка скрытого слоя
        public double [] Bias { get; set; } // массив смещений
        public double HiddenGrad { get; set; } // градиент скрытого слоя

        Random rand = new Random();
        /* ******************************************************************************************************** */
        /* Блок конструкторов */
        /* ******************************************************************************************************** */
        public Layer(int numIn, int numNeurons)
        {   
            NumNeurons = numNeurons;
            //   X = input;
            Output = new double[numNeurons];
            // Каждый столбец соответствует массиву коэффициентов одного нейрона слоя. 
            W = new double[NumNeurons, numIn];
            Bias = new double[NumNeurons]; // смещение каждого нейрона сети

            for (int i = 0; i < NumNeurons; i++)
            {
                Bias[i] = rand.NextDouble();
                for (int j = 0; j < numIn; j++)
                {
                    W[i,j] = rand.NextDouble();
                }
            }
        }
        public Layer()
        {
        }
        public double[] GetLayerOut(double a)
        {
            double y = 0;
            double sum = 0;

            Parallel.For(0, NumNeurons, (i) =>
            {
                y = 0;
                sum = 0;
                for (int j = 0; j < X.Length; j++)
                {
                    sum += X[j] * W[i, j];
                }
                y = Bias[i] + sum;
                Output[i] = 1 / (1 + Math.Exp(-a * y));
            });
            return Output;
        }
        public double[] GetLinearOut()
        {
            double sum = 0;
            for (int i = 0; i < NumNeurons; i++)
            {
                sum = 0;
                for (int j = 0; j < X.Length; j++)
                {
                    sum += X[j] * W[i, j];
                }
                Output[i] = Bias[i] + sum;
            }
            return Output;
        }
        public double[] GetTangLayerOut(double a, double b)
        {
            double y;
            double sum;
            for (int i = 0; i < NumNeurons; i++)
            {
                y = 0;
                sum = 0;
                for (int j = 0; j < X.Length; j++)
                {
                    sum += X[j] * W[i, j];
                }
                y = Bias[i] + sum;
                Output[i] = Math.Tanh(a * y);
                //Output[i] = (2 / (1 + Math.Exp(-y))) - 1; // двухполюсный сигмоид
            }
            return Output;
        }
        public void CalcWeightsOutLayer(double learningRate, double[] d, double a)
        {
            LocalGrad = new double[X.Length];
            double[] error = new double[NumNeurons];
            Parallel.For(0, error.Length, (i) =>
             {
                 error[i] = a * (d[i] - Output[i]) * Output[i] * (1 - Output[i]);
                 Bias[i] = Bias[i] + learningRate * error[i];
                 for (int j = 0; j < X.Length; j++)
                 {
                     LocalGrad[j] += error[i] * W[i, j];
                     W[i, j] = W[i, j] + learningRate * error[i] * X[j];
                 }
             });
        }
        public void CalcWeightsHiddenLayer(double learningRate, double[] localGrad, double a)
        {
            LocalGrad = new double[X.Length];
            double[] error = new double[NumNeurons];
            for (int i = 0; i < error.Length; i++)
            {
                error[i] = a * Output[i] * (1 - Output[i]) * localGrad[i];
                Bias[i] = Bias[i] + learningRate * error[i];
                for (int j = 0; j < X.Length; j++)
                {
                    LocalGrad[j] += error[i] * W[i, j];
                    W[i, j] = W[i, j] + learningRate * error[i] * X[j];
                }
            }
        }
        public void CalcWeightsTangOutLayer(double learningRate, double[] d, double a, double b)
        {
            LocalGrad = new double[X.Length];
            //LocalGrad = 0;
            double[] error = new double[NumNeurons];
            for (int i = 0; i < error.Length; i++)
            {
                error[i] = (b / a) * (d[i] - Output[i]) * (a - Output[i]) * (a + Output[i]);
                Bias[i] = Bias[i] + learningRate * error[i];
                //grad[i] = 0.5 * (d[i] - Output[i]) * (1 - Output[i]) * (1 + Output[i]); // двухполюсный сигмоид
                for (int j = 0; j < X.Length; j++)
                {
                    LocalGrad[j] += error[i] * W[i, j];
                    W[i, j] = W[i, j] + learningRate * error[i] * X[j];
                }
            }

        }
        public void CalcWeightsLinearOutLayer(double learningRate, double[] d)
        {
            LocalGrad = new double[X.Length];
            //LocalGrad = 0;
            double[] error = new double[NumNeurons];
            for (int i = 0; i < error.Length; i++)
            {
                error[i] = d[i] - Output[i];
                Bias[i] = Bias[i] + learningRate * error[i];
                for (int j = 0; j < X.Length; j++)
                {
                    LocalGrad[j] += error[i] * W[i, j];
                    W[i, j] = W[i, j] + learningRate * error[i] * X[j];
                }
            }
        }
        public void CalcWeightsHiddenTangLayer(double learningRate, double[] localGrad, double a, double b)
        {
            LocalGrad = new double[X.Length];
            double[] error = new double[NumNeurons];
            for (int i = 0; i < error.Length; i++)
            {
                error[i] = (b / a) * (a - Output[i]) * (a + Output[i]) * localGrad[i];
                //grad[i] = 0.5 * (a - Output[i]) * (a + Output[i]) * localGrad[i]; // двухполюсный сигмоид
                Bias[i] = Bias[i] + learningRate * error[i];
                for (int j = 0; j < X.Length; j++)
                {
                    LocalGrad[j] += error[i] * W[i, j];
                    W[i, j] = W[i, j] + learningRate * error[i] * X[j];
                }
            }
        }
        public double GetOutError(double d)
        {
            OutError[0] = d - Output[0];
            return OutError[0];
        }


        //public async Task<double[]> GetLayerOut(double a)
        //{
        //   return await Task.Run(()=> {
        //        double y = 0;
        //        double sum = 0;
        //        for (int i = 0; i < NumNeurons; i++)
        //        {
        //            y = 0;
        //            sum = 0;
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                sum += X[j] * W[i, j];
        //            }
        //            y = Bias[i] + sum;
        //            Output[i] = 1 / (1 + Math.Exp(-a * y));
        //        }
        //        return Output;
        //    });
        //}
        //public async Task<double[]> GetLinearOut()
        //{
        //    return await Task.Run(() => {
        //        double sum = 0;
        //        for (int i = 0; i < NumNeurons; i++)
        //        {
        //            sum = 0;
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                sum += X[j] * W[i, j];
        //            }
        //            Output[i] = Bias[i] + sum;
        //        }
        //        return Output;
        //    });
        //}
        //public async Task<double[]> GetTangLayerOut(double a, double b)
        //{
        //    return await Task.Run(() =>
        //    {
        //        double y;
        //        double sum;
        //        for (int i = 0; i < NumNeurons; i++)
        //        {
        //            y = 0;
        //            sum = 0;
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                sum += X[j] * W[i, j];
        //            }
        //            y = Bias[i] + sum;
        //            Output[i] = Math.Tanh(a * y);
        //            //Output[i] = (2 / (1 + Math.Exp(-y))) - 1; // двухполюсный сигмоид
        //        }
        //        return Output;
        //    });
        //}
        //public async Task CalcWeightsOutLayer(double learningRate, double[] d, double a)
        //{
        //    await Task.Run(() =>
        //    {
        //        LocalGrad = new double[X.Length];
        //        double[] error = new double[NumNeurons];
        //        for (int i = 0; i < error.Length; i++)
        //        {
        //            error[i] = a * (d[i] - Output[i]) * Output[i] * (1 - Output[i]);
        //            Bias[i] = Bias[i] + learningRate * error[i];
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                LocalGrad[j] += error[i] * W[i, j];
        //                W[i, j] = W[i, j] + learningRate * error[i] * X[j];
        //            }
        //        }
        //    });
        //}
        //public async Task CalcWeightsHiddenLayer(double learningRate, double[] localGrad, double a)
        //{
        //    await Task.Run(() =>
        //    {
        //        LocalGrad = new double[X.Length];
        //        double[] error = new double[NumNeurons];
        //        for (int i = 0; i < error.Length; i++)
        //        {
        //            error[i] = a * Output[i] * (1 - Output[i]) * localGrad[i];
        //            Bias[i] = Bias[i] + learningRate * error[i];
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                LocalGrad[j] += error[i] * W[i, j];
        //                W[i, j] = W[i, j] + learningRate * error[i] * X[j];
        //            }
        //        }
        //    });
        //}
        //public async Task CalcWeightsTangOutLayer(double learningRate, double[] d, double a, double b)
        //{
        //    await Task.Run(() =>
        //    {
        //        LocalGrad = new double[X.Length];
        //        //LocalGrad = 0;
        //        double[] error = new double[NumNeurons];
        //        for (int i = 0; i < error.Length; i++)
        //        {
        //            error[i] = (b / a) * (d[i] - Output[i]) * (a - Output[i]) * (a + Output[i]);
        //            Bias[i] = Bias[i] + learningRate * error[i];
        //            //grad[i] = 0.5 * (d[i] - Output[i]) * (1 - Output[i]) * (1 + Output[i]); // двухполюсный сигмоид
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                LocalGrad[j] += error[i] * W[i, j];
        //                W[i, j] = W[i, j] + learningRate * error[i] * X[j];
        //            }
        //        }
        //    });
        //}
        //public async Task CalcWeightsLinearOutLayer(double learningRate, double[] d)
        //{
        //    await Task.Run(() =>
        //    {
        //        LocalGrad = new double[X.Length];
        //        //LocalGrad = 0;
        //        double[] error = new double[NumNeurons];
        //        for (int i = 0; i < error.Length; i++)
        //        {
        //            error[i] = d[i] - Output[i];
        //            Bias[i] = Bias[i] + learningRate * error[i];
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                LocalGrad[j] += error[i] * W[i, j];
        //                W[i, j] = W[i, j] + learningRate * error[i] * X[j];
        //            }
        //        }
        //    });
        //}
        //public async Task CalcWeightsHiddenTangLayer(double learningRate, double[] localGrad, double a, double b)
        //{
        //    await Task.Run(() =>
        //    {
        //        LocalGrad = new double[X.Length];
        //        double[] error = new double[NumNeurons];
        //        for (int i = 0; i < error.Length; i++)
        //        {
        //            error[i] = (b / a) * (a - Output[i]) * (a + Output[i]) * localGrad[i];
        //            //grad[i] = 0.5 * (a - Output[i]) * (a + Output[i]) * localGrad[i]; // двухполюсный сигмоид
        //            Bias[i] = Bias[i] + learningRate * error[i];
        //            for (int j = 0; j < X.Length; j++)
        //            {
        //                LocalGrad[j] += error[i] * W[i, j];
        //                W[i, j] = W[i, j] + learningRate * error[i] * X[j];
        //            }
        //        }
        //    });
        //}
        //public async Task<double> GetOutError(double d)
        //{
        //    return await Task.Run(() =>
        //    {
        //        OutError[0] = d - Output[0];
        //        return OutError[0];
        //    });
        //}
    }
}
