from itertools import product
from tqdm import tqdm

def rep_simulations(fn, single_vars, list_vars, repetitions = 1, verbose = False):
    #input 
    #single_vars = {"b": b1, "c": c1}
    #list_vars = {"a": [a1,a2], "d": [d1,d2]}
    #fn = fn(a,b,c,d)
    #repetitions
    #
    #repeat fn for each possible vector (a1,b1,c1,d1), (a2,b1,c1,d1), (a1,b1,c1,d2), (a2,b1,c1,d2)
    
    
    outputs = []
    
    for rep in tqdm(range(repetitions), disable = not verbose):
        for lists_vars_element in [dict(zip(list_vars.keys(), items)) for items in product(*list(list_vars.values()))]:
            
            input_vector = single_vars|lists_vars_element
            outputs.append([lists_vars_element, fn(**input_vector)])
              
              
    return outputs
 
    
def nested_rep(input_fn1, output_fn1_names, input_fn2, verbose1 = True, verbose2 = False):
    outputs = []
    
    
    for out1 in tqdm(rep_simulations(*input_fn1), disable = not verbose1):
        
        fn2, single_vars2, list_vars2, repetitions2 = input_fn2
        temp_single_vars2 = {**single_vars2, **dict(zip(output_fn1_names, out1[1]))}
        for out2 in tqdm(rep_simulations(fn2, temp_single_vars2, list_vars2, repetitions2), disable = not verbose2):
            outputs.append({**input_fn1[1], **out1[0], **out2[0], **out2[1]})
        
    return outputs