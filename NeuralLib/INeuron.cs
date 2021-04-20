using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib
{
    public interface INeuron
    {

        double[] X { get; set; } // массив входных значений нейрона 
        double Output { get; set; }
        double[] W { get; set; } // двумерный массив коэффициентов нейрона. 
        double Error { get; set; }
        double Bias { get; set; } // массив смещений
        double GetOutput();
        void CalcWeights(double a, double d); // вычислить новые коэффициенты с учетом ошибки
        double LocalGrad { get; set; }
    }
}
