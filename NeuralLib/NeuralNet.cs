using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib
{
    public class NeuralNet
    {
        ILayer[] layers;
        public double[] Output { get; set; } // выход нейронной сети
        public double[] Error { get; set; } // ошибка выходного слоя
        public double MSE { get; set; } // среднеквадратическая ошибка
        public NeuralNet(params ILayer[] layers)
        {
            this.layers = layers;
        }

        public double[] GetOutput(double[] input) // получить выход нейронной сети
        {
            layers[0].GetOutput(input); // получить выход первого слоя
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].GetOutput(layers[i - 1].Output); // в цикле подать на вход следующего входа выход предыдущего
                // повторять до тех пор пока цикл не достигнет последнего слоя сети
            }
            Output = layers[layers.Length - 1].Output; // присвоить и возвратить значение выхода сети        
            return Output;
        }
        public double[] GetError()
        {
            return layers[layers.Length - 1].Error; // возвратить ошибку выходного слоя
        }
        public void Learn(double learningRate, double[] d)
        {
            layers[layers.Length - 1].CalcWeights(learningRate, d); // корректировать значения весовых коэффициентов выходного слоя
            for (int i = layers.Length-2; i >= 0; i--)
            {
                layers[i].CalcWeights(learningRate, layers[i + 1].LocalGrad);
                // в обратном порядке после выходного слоя выполнить корректировку весов. 
                // В качестве второго параметра функции корректировки весов выступает локальный градиент предыдущего слоя, 
                // двигаясь в обратном направлении(от выхода ко входу). 
                // цикл продолжается до тех пор пока не будет выполнена корректировка весовых коэффициентов первого слоя
            }
        }
        // not ready
        public double GetMSE()
        {
            return MSE; 
        }
    }
}
