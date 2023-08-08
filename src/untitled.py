import sys


inputs = []
for el in sys.argv[1:]:
    inputs.append(float(el))
    

    
if __name__ == '__main__':
    print(inputs)
    a,b,c = inputs
    
    print(f"results = {a * b - c}")
