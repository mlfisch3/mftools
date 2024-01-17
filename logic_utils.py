


def AND(x,y):
    return [y,False][x]

def NEGATE(x):
    return [True,False][x]

def NAND(x,y):
    return NEGATE(AND(x,y))

def OR(x,y):
    return [y,True][x]

def NOR(x,y):
    return NEGATE(OR(x,y))

def XOR(x,y):
    return [y,NEGATE(y)][x]
    

def AND(x,y):
    return [y,False][x]

def NAND(x,y):
    return not AND(x,y)

def OR(x,y):
    return [y,True][x]

def NOR(x,y):
    return not OR(x,y)

def XOR(x,y):
    return [y,not y][x]