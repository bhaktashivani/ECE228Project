import os

def do():
   for file in os.listdir("."): 
      if file.endswith(".csv"): 
          print(file)
if __name__== '__main__':
   input_dir = sys.argv[1]
   output_dir = sys.argv[2]
   do()
