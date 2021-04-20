using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib
{
    public class Node
    {
        private static readonly Random r = new Random();
        private static readonly Random r1 = new Random();
        public double[] W { get; set; }
        public Node(int numIns)
        {
            W = new double[numIns];
            for (int i = 0; i < W.Length; i++)
            {
                //W[i] = input[r.Next(input.Length)][r1.Next(input[i].Length)];
                W[i] = r.NextDouble();
                //  W[i] = r.Next(10);
            }
            //Parallel.For(0, W.Length, (i) =>
            //{
            //    //W[i] = input[r.Next(input.Length)][r1.Next(input[i].Length)];
            //    W[i] = r.NextDouble();
            //    //  W[i] = r.Next(10);
            //});
        }
        public double GetDistance(double[] input)
        {
            double distance = 0.0;
            for (int i = 0; i < W.Length; i++)
            {
                distance += (input[i] - W[i]) * (input[i] - W[i]);
            }
            return Math.Sqrt(distance);
        }
    }
    public class SOM2D
    {
        private static readonly Random r = new Random();
        public double[,] Output { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public Node[,] Nodes { get; set; }
        double learningRate = 0.0;
        double distance = 0.0;
        double sigma = 0.0;
        double gamma = 0.0;
        int indexWidth = 0;
        int indexHeight = 0;
        double etta = 0.0;
        public SOM2D(int width, int height, int numIn)
        {
            Width = width;
            Height = height;
            Nodes = new Node[Width, Height];
            Output = new double[Width, Height];
            for (int i = 0; i < Width; i++)
            {
                for (int j = 0; j < Height; j++)
                {
                    Nodes[i, j] = new Node(numIn);
                }
            }
        }
        public void Learn(double[] input, int currentIteration, double learningRate0, int sigma0)
        {
            gamma = 1000.0 / Math.Log10(sigma0); // default 1000
            learningRate = learningRate0 * Math.Exp(-((Convert.ToDouble(currentIteration) / 1000.0))); // default 1000
            sigma = sigma0 * Math.Exp(-((Convert.ToDouble(currentIteration) / gamma)));
            for (int i = 0; i < Width; i++)
            {
                for (int j = 0; j < Height; j++)
                {
                    Output[i, j] = Nodes[i, j].GetDistance(input);
                }
            }
            //Parallel.For(0, Width, (i) =>
            //{
            //    Parallel.For(0, Height, (j) =>
            //    {
            //        Output[i, j] = Nodes[i, j].GetDistance(input);
            //        //if (min >= Output[i,j])
            //        //{
            //        //    min = Output[i,j];
            //        //    indexWidth = i;
            //        //    indexHeight = j;
            //        //}
            //    });
            //});
            var result = from i in Enumerable.Range(0, Width)
                         from j in Enumerable.Range(0, Height)
                         orderby Output[i, j]
                         select new { i, j };
            var r = result.FirstOrDefault();// here are the indexes of min value

            indexWidth = r.i;
            indexHeight = r.j;

            for (int i = 0; i < Width; i++)
            {
                for (int j = 0; j < Height; j++)
                {
                    distance = (int)Math.Sqrt(Math.Pow(i - indexWidth, 2) + Math.Pow(j - indexHeight, 2));
                    etta = Math.Exp(-((distance * distance) / (2 * (sigma * sigma))));
                    for (int w = 0; w < input.Length; w++)
                    {
                        Nodes[i, j].W[w] += (learningRate >= 0.01 ? learningRate : 0.01) * (sigma >= etta ? etta : 1) * (input[w] - Nodes[i, j].W[w]);
                    }
                }
            }
        }
    }
    public class SOM
    {
        private static readonly Random r = new Random();
        public double[] Output { get; set; }
        public Node[] Nodes { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        double learningRate = 0.0;
        double sigma = 0.0;
        double gamma = 0.0;
        int index = 0;
        double distance = 0.0;
        double etta = 0.0;
        public SOM(int width, int height, int numIn)
        {
            Width = width;
            Height = height;
            Nodes = new Node[Width*Height];
            Output = new double[Width*Height];
            for (int i = 0; i < Width*Height; i++)
            {
                Nodes[i] = new Node(numIn);
            }
        }
        public void Learn(double[] input, int currentIteration, double learningRate0, int sigma0)
        {
            gamma = 1000.0 / Math.Log10(sigma0);
            learningRate = learningRate0 * Math.Exp(-((Convert.ToDouble(currentIteration) / 1000.0)));
            sigma = sigma0 * Math.Exp(-((Convert.ToDouble(currentIteration) / gamma)));

            for (int i = 0; i < Width*Height; i++)
            {
                    Output[i] = Nodes[i].GetDistance(input);
            }

            for (int i = 0; i < Width * Height; i++)
            {
                distance = Math.Pow(input[0] - Nodes[i].W[0], 2) + Math.Pow(input[1] - Nodes[i].W[1], 2);
                Output[i] = Math.Sqrt(distance);
            }
            // find index of min element
            index = Enumerable.Range(0, Output.Length).AsParallel().Aggregate((a, b) => (Output[a] < Output[b]) ? a : b); // returns 2
            for (int i = 0; i < Width * Height; i++)
            {
                distance = Math.Abs(i - index);
                etta = Math.Exp(-((distance * distance) / (2 * (sigma * sigma))));
                for (int w = 0; w < 2; w++)
                {
                    Nodes[i].W[w] += (learningRate >= 0.01 ? learningRate : 0.01) * (sigma>=etta?etta:1) * (input[w] - Nodes[i].W[w]);
                }
            }
        }
    }
}
