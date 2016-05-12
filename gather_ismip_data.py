#!/usr/bin/env python2

"""
This script gathers all the ISMIP-HOM experiments' data.
"""

import os
import math
import numpy
import scipy
import fnmatch

import matplotlib.pyplot as plt
import pprint as pp

# Location of data
#TODO: argparse this.
ismip_data = './ismip_all'

# defaults for cism grid
ewn = 40    
nsn = 40

#--------------------------
# ISMIP-HOM data constants 
#--------------------------
#NOTE: Not including 'aas1' as the files do not conform to the data standards.
full_stokes = ['aas2','cma1','fpa2','ghg1','jvj1','mmr1','oga1','rhi1',
               'rhi3','spr1','ssu1','yko1']

lmla = ['ahu1','ahu2','bds1','cma2','fpa1','fsa1','mbr1','rhi2','tpa1']
l1l2 = ['dpo1','rhi4']
l1l1 = ['lpe1','rhi5']
ltsml = ['mtk1']
higher_order = lmla + l1l2 + l1l1 + ltsml

sia = ['oso1']

header = {'a':['x_hat','y_hat','vx_surf','vy_surf','tau_xz','tau_yz','del_p'],
          'b':['x_hat','vx_surf','vz_surf','tau_xz','del_p'],
          'c':['x_hat','y_hat','vx_surf','vy_surf','vz_surf','vx_base',
               'vy_base','tau_xz','tau_yz','del_p'],
          'd':['x_hat','vx_surf','vz_surf','vx_base','tau_xz','del_p'],
          'e':['x_hat','vx_surf','vz_surf','tau_xz','del_p'],
          'f':['x_hat','y_hat','z_surf','vx','vy','vz']
         }
sizes = {'a': [5, 10, 20, 40, 80, 160],
         'b': [5, 10, 20, 40, 80, 160],
         'c': [5, 10, 20, 40, 80, 160],
         'd': [5, 10, 20, 40, 80, 160],
         'e': [],
         'f': [100],
        }


class ismip_datum:
    """A class to hold each model's data"""
    def __init__(self, data_file):
        self.df = data_file
        self.M, self.E, self.L = self.parse_file(data_file)
        
        # classify
        if self.M in full_stokes:
            self.order = 'full_stokes'
        elif self.M in higher_order:
            self.order = 'higher_order'
        elif self.M in sia:
            self.order = 'sia'
        else:
            self.order = 'unknown'
        
        # load the data
        if self.order != 'unknown':
            self.array = self.load_data()
        else:
            self.array = []


    def parse_file(self, data_file):
        """
        Parse the ismip_hom data filenames. They should look like: NNNMELLL.tex, where NNN = model name, 
             M = model number, E = experiment, LLL = three numbers denoting: 
                 experiments a-d: the length of L in km. 
                 experiment e: 000 for non-sliding and 001 for the experiment with 
                               the zone of zero basal raction
                 experiment f: the slip ratio; either 000 or 001.
        """
        code_name = os.path.basename(os.path.splitext(data_file)[0])
        return (code_name[0:4], code_name[4], code_name[5:]) # (model, experiment, length)

    def load_data(self):
        data = numpy.loadtxt(self.df)
        return data

    def make_grid(self):
        self.xy_array = 0

    def display(self):
        print("Data file: "+self.df)
        print("Order: "+self.order)
        print("Model: "+self.M)
        print("Experiment: "+self.E)
        print("Length: "+self.L)


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

#-----------
# The files 
#-----------
def recursive_glob(tree, pattern):
    matches = []
    for base, dirs, files in os.walk(tree):
        goodfiles = fnmatch.filter(files, pattern)
        matches.extend(os.path.join(base, f) for f in goodfiles)
    return matches

data_files = recursive_glob(ismip_data, '*.txt')

all_data = []
for i, df in enumerate(data_files):
    all_data.append(ismip_datum(df))
    #all_data[i].display()
    #print("------")

## see what experiments are in each order
#A = set([data.E for data in all_data if data.order == 'full_stokes' ])
#pp.pprint(A)

#for data in all_data:
#    if data.E == 'a' and data.order == 'full_stokes':
#        pp.pprint(data.array[:,0:2])
#        print('----------------------------')

#for data in all_data:
#    if data.E == 'a' and data.order == 'full_stokes':
#        plt.scatter(data.array[:,0],data.array[:,1],cmap ='RdYlGn_r')
#        plt.show()

## see if all are square grids or not...
#A = [len(data.array[:,1]) for data in all_data if data.order == 'full_stokes' ]
#pp.pprint( [math.sqrt(a) for a in A] )



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










