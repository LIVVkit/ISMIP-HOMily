#!/usr/bin/env python2

import numpy
import scipy

# defaults for cism grid
ewn = 40    
nsn = 40

sizes = {'a': [5, 10, 20, 40, 80, 160],
         'b': [5, 10, 20, 40, 80, 160],
         'c': [5, 10, 20, 40, 80, 160],
         'd': [5, 10, 20, 40, 80, 160],
         'e': [],
         'f': [100],
        }

#----------------------
# CISM ISMIP-HOM grids 
#----------------------
class cism_grids:
    """A class to create a CISM grid for comparison."""
    #NOTE: the *_extend variables appear to be on the x,y grids, but are actually on the staggered grid. 
    #      These fields contain the ice velocity computed at the upper right corner of each grid cell and the
    #      additional row/columninclude the first ghost value past ewn/nsn, which is valid at both 0.0 and 1.0
    #      on the non-dimensional coordinate system (*_hat). Note, CISM uses a cell-centered grid.
    #
    #      That is, CISM is setup, in the non-dimensional coordinate system, with the unstaggered grid going
    #      from [{-0.0125}, 0.0125, 0.0375, ..., 0.9875, {1.0125}] where {}-ed points are ghost points (a 
    #      cell centered grid), and the staggered points going from [{0}, 0.025, 0.05, ..., 0.975, {1.0}]. 
    #      The ghost points are NOT included in the netCDF file for either grid. 
    #
    #      So, on the extended grid, the length of the grids is the same as the regular x,y grid and reported
    #      to be on the x,y grid in the netCDF file. BUT, it's really on the staggered grid with the final 
    #      ghost point appended to the end of the array. Note, this is equivelent to reporting the upper-right 
    #      corner of the regular grid x,y cells. Because of the periodic BC, the final row/column {1.0} is 
    #      equal to the un-reported ghost points at {0.0}, and so, by making an extra-extended grid, with both
    #      ghost points, all interpolated points should fall within the convex hull of the extended grid.

    def __init__(self, size):
        self.size = size
        self.nx = ewn
        self.ny = nsn
        self.dx = float(self.size)*1000./float(self.nx)
        self.dy = float(self.size)*1000./float(self.ny)
        
        self.x = [(i+0.5)*self.dx for i in range(self.nx)] # unstaggered grid (x1,y1)
        self.y = [(j+0.5)*self.dy for j in range(self.ny)]
        self.y_grid, self.x_grid = scipy.meshgrid(self.y[:], self.x[:], indexing='ij')
        
        self.x_hat = [i/(self.size*1000) for i in self.x]
        self.y_hat = [i/(self.size*1000) for i in self.y]
        self.y_hat_grid, self.x_hat_grid = scipy.meshgrid(self.y_hat[:], self.x_hat[:], indexing='ij')
        
        self.x_stag = [(i+1)*self.dx for i in range(self.nx-1)] # staggered grid (x0,y0)
        self.y_stag = [(j+1)*self.dy for j in range(self.ny-1)]
        self.y_stag_grid, self.x_stag_grid = scipy.meshgrid(self.y_stag[:], self.x_stag[:], indexing='ij')

        self.x_hat_stag = [i/(self.size*1000) for i in self.x_stag]
        self.y_hat_stag = [i/(self.size*1000) for i in self.y_stag]
        self.y_hat_stag_grid, self.x_hat_stag_grid = scipy.meshgrid(self.y_hat_stag[:], self.x_hat_stag[:], indexing='ij')


#----------------
# The cism grids 
#----------------
grd = []
for sz in sizes['a']:
    grd.append(cism_grids(sz))

grids = {'a': grd,
         'b': grd,
         'c': grd,
         'd': grd,
         'e': [],
         'f': [cism_grids(sizes['f'][0])],
        }

