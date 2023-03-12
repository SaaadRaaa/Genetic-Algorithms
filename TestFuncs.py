# %% [0] Import libs:
import myGenetic as GA
import numpy as np
# %% [1] Achley function:
# ----------------- Create poulation -----------------


nd = 3
x = GA.createPopReal(100, nd, [[-10, 10]])


# ----------------- Test function -------------------


def Ackley(x, a=20, b=0.2, c=2*np.pi):
    d = x.shape[0]
    f = -a*np.exp(-b*np.sqrt(np.sum(x**2)/d)) - \
        np.exp(np.sum(np.cos(c*x))/d)+a+np.exp(1)
    return 0-f


nPar = 20
for i in range(40):
    fitness = GA.fitCal(Ackley, x)
    print("Generation", i+1, "done.")
    parents = GA.parSelRWS(x, fitness, nPar, rechoose=True)
    children = np.array(list(map(
        GA.combConWHA, parents[0:nPar//2, :], parents[nPar//2:nPar, :]))).reshape((nPar, nd))
    np.apply_along_axis(GA.mutConUni, 1, children)
    x = GA.surSelRep(x, children, fitness, GA.fitCal(
        Ackley, children), dup=False)


print(x[np.argsort(list(map(abs, GA.fitCal(Ackley, x))))][:3])
print(np.sort(list(map(abs, GA.fitCal(Ackley, x))))[0:3])

# %% Levy function:
# ----------------- Create poulation -----------------


nd = 3
x = GA.createPopReal(200, nd, [[-10, 10]])


# ----------------- Test function -------------------


def Levy(x):
    w = 1+(x-1)/4
    f = [(w[i]-1)**2*(1+10*np.sin(np.pi*w[i]+1)**2)
         for i in range(1, w.shape[0]-1)]
    f.append(np.sin(np.pi*w[0]))
    f.append((w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])**2))
    return 0-sum(f)


nPar = 20
for i in range(300):
    fitness = GA.fitCal(Levy, x)
    print("Generation", i+1, "done.")
    parents = GA.parSelTNS(x, fitness, nPar, rechoose=False, indivs=3)
    children = np.array(list(map(
        GA.combConWHA, parents[0:nPar//2, :], parents[nPar//2:nPar, :]))).reshape((nPar, nd))
    np.apply_along_axis(GA.mutConGau, 1, children)
    x = GA.surSelRep(x, children, fitness, GA.fitCal(
        Levy, children), dup=False)


print(x[np.argsort(list(map(abs, GA.fitCal(Levy, x))))][:3])
print(np.sort(list(map(abs, GA.fitCal(Levy, x))))[0:3])

# %% Sphere function:
# ----------------- Create poulation -----------------


nd = 3
x = GA.createPopInt(10, nd, [[-5.12, 5.12]])


# ----------------- Test function -------------------

def Sphere(x):
    f = [x[i]**2 for i in range(0, nd)]
    return 0-sum(f)


nPar = 2
for i in range(5):
    fitness = GA.fitCal(Sphere, x)
    print("Generation", i+1, "done.")
    parents = GA.parSelTNS(x, fitness, nPar, rechoose=False)
    children = np.array(list(map(
        GA.combDisOPC, parents[0:nPar//2, :], parents[nPar//2:nPar, :]))).reshape((nPar, nd))
    np.apply_along_axis(GA.mutConUni, 1, children)
    x = GA.surSelAge(x, children)


print(x[np.argsort(list(map(abs, GA.fitCal(Sphere, x))))][:3])
print(np.sort(list(map(abs, GA.fitCal(Sphere, x))))[0:3])

# %% Booth function:
# ----------------- Create poulation -----------------


nd = 2
x = GA.createPopInt(100, nd, [[-10, 10]])


# ----------------- Test function -------------------


def Booth(x):
    f = ((x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2)
    return 0-f


nPar = 2
for i in range(40):
    fitness = GA.fitCal(Booth, x)
    print("Generation", i+1, "done.")
    parents = GA.parSelUNI(x, nPar, rechoose=True)
    children = np.array(list(map(
        GA.combDisNPC, parents[0:nPar//2, :], parents[nPar//2:nPar, :]))).reshape((nPar, nd))
    np.apply_along_axis(GA.mutConUni, 1, children)
    x = GA.surSelAge(x, children)


print(x[np.argsort(list(map(abs, GA.fitCal(Booth, x))))][:3])
print(np.sort(list(map(abs, GA.fitCal(Booth, x))))[0:3])
