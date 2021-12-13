from ExDaAn import EDA
from Model import Liberty_Model
import argparse

if __name__=='__main__':

   parser = argparse.ArgumentParser(description='This is an Argument parser expecting path and model as imputs')
   parser.add_argument('-p','--path',  type=str, help='The Path of the data')
   parser.add_argument('-m','--model', type=str, help='Choose Between LR (balanced LogisticRegressin) or RF(balanced RandomForest)')

   args = parser.parse_args()

   eda = EDA(args.path)
   data = eda.execute()

   lm = Liberty_Model(data,args.model)
   lm.lmodel()


