import numpy as np
import pandas as pd
import yfinance as yf #pip install yfinance
tick = yf.Ticker('^GSPC')
hist = tick.history(period="max", rounding=True)
hist = hist[:'2019-10-07']
# hist = hist[-1000:]
h = hist.Close.tolist()

minimaIdxs = np.flatnonzero(
 hist.Close.rolling(window=3, min_periods=1, center=True).aggregate(
   lambda x: len(x) == 3 and x[0] > x[1] and x[2] > x[1])).tolist()
maximaIdxs = np.flatnonzero(
 hist.Close.rolling(window=3, min_periods=1, center=True).aggregate(
   lambda x: len(x) == 3 and x[0] < x[1] and x[2] < x[1])).tolist()


hs = hist.Close.loc[hist.Close.shift(-1) != hist.Close]
x = hs.rolling(window=3, center=True).aggregate(lambda x: x[0] > x[1] and x[2] > x[1])
minimaIdxs = [hist.index.get_loc(y) for y in x[x == 1].index]
x = hs.rolling(window=3, center=True).aggregate(lambda x: x[0] < x[1] and x[2] < x[1])
maximaIdxs = [hist.index.get_loc(y) for y in x[x == 1].index]

from findiff import FinDiff #pip install findiff
dx = 1 #1 day interval
d_dx = FinDiff(0, dx, 1)
d2_dx2 = FinDiff(0, dx, 2)
clarr = np.asarray(hist.Close)
mom = d_dx(clarr)
momacc = d2_dx2(clarr)

def get_extrema(isMin):
  return [x for x in range(len(mom))
    if (momacc[x] > 0 if isMin else momacc[x] < 0) and
      (mom[x] == 0 or #slope is 0
        (x != len(mom) - 1 and #check next day
          (mom[x] > 0 and mom[x+1] < 0 and
           h[x] >= h[x+1] or
           mom[x] < 0 and mom[x+1] > 0 and
           h[x] <= h[x+1]) or
         x != 0 and #check prior day
          (mom[x-1] > 0 and mom[x] < 0 and
           h[x-1] < h[x] or
           mom[x-1] < 0 and mom[x] > 0 and
           h[x-1] > h[x])))]
minimaIdxs, maximaIdxs = get_extrema(True), get_extrema(False)



import findiff
coeff = findiff.coefficients(deriv=1, acc=2)
print(coeff)


# hist = hist[:'2019–10–07']
day = 23043 # September 27, 2019 per example (-7 or 7th last point)
sum([coeff['center']['coefficients'][x] *
     hist.Close[day + coeff['center']['offsets'][x]]
     for x in range(len(coeff['center']['coefficients']))])


coeff=findiff.coefficients(deriv=2, acc=2)
print(coeff)

sum([coeff['center']['coefficients'][x] *
     hist.Close[day + coeff['center']['offsets'][x]]
     for x in range(len(coeff['center']['coefficients']))])


def get_bestfit3(x0, y0, x1, y1, x2, y2):
  xbar, ybar = (x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3
  xb0, yb0, xb1, yb1, xb2, yb2 = x0-xbar, y0-ybar, x1-xbar, y1-ybar, x2-xbar, y2-ybar
  xs = xb0*xb0+xb1*xb1+xb2*xb2
  m = (xb0*yb0+xb1*yb1+xb2*yb2) / xs
  b = ybar - m * xbar
  ys0, ys1, ys2 =(y0 - (m * x0 + b)),(y1 - (m * x1 + b)),(y2 - (m * x2 + b))
  ys = ys0*ys0+ys1*ys1+ys2*ys2
  ser = np.sqrt(ys / xs)
  return m, b, ys, ser, ser * np.sqrt((x0*x0+x1*x1+x2*x2)/3)


def get_bestfit(pts):
    xbar, ybar = [sum(x) / len (x) for x in zip(*pts)]
def subcalc(x, y):
    tx, ty = x - xbar, y - ybar
    return tx * ty, tx * tx, x * x
    (xy, xs, xx) =[sum(q) for q in zip(*[subcalc(x, y) for x, y in pts])]
    m = xy / xs
    b = ybar - m * xbar
    ys = sum([np.square(y - (m * x + b)) for x, y in pts])
    ser = np.sqrt(ys / ((len(pts) - 2) * xs))
    return m, b, ys, ser, ser * np.sqrt(xx / len(pts))



ymin, ymax = [h[x] for x in minimaIdxs], [h[x] for x in maximaIdxs]
zmin, zmne, _, _, _ = np.polyfit(minimaIdxs, ymin, 1, full=True)  #y=zmin[0]*x+zmin[1]
pmin = np.poly1d(zmin).c
zmax, zmxe, _, _, _ = np.polyfit(maximaIdxs, ymax, 1, full=True) #y=zmax[0]*x+zmax[1]
pmax = np.poly1d(zmax).c
print((pmin, pmax, zmne, zmxe))


p, r = np.polynomial.polynomial.Polynomial.fit(minimaIdxs, ymin, 1, full=True) #more numerically stable
pmin, zmne = list(reversed(p.convert().coef)), r[0]
p, r = np.polynomial.polynomial.Polynomial.fit(maximaIdxs, ymax, 1, full=True) #more numerically stable
pmax, zmxe = list(reversed(p.convert().coef)), r[0]




def get_trend(Idxs):
  trend = []
  for x in range(len(Idxs)):
    for y in range(x+1, len(Idxs)):
      for z in range(y+1, len(Idxs)):
        trend.append(([Idxs[x], Idxs[y], Idxs[z]],
          get_bestfit3(Idxs[x], h[Idxs[x]],
                       Idxs[y], h[Idxs[y]],
                       Idxs[z], h[Idxs[z]])))
  return list(filter(lambda val: val[1][3] <= fltpct, trend))
mintrend, maxtrend = get_trend(minimaIdxs), get_trend(maximaIdxs)