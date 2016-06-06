#!/usr/bin/env python2

# Copyright (c) 2015, UT-BATTELLE, LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
This script gathers all the ISMIP-HOM experiments' data.
"""


#FIXME: There are some problems with the recreation of the 5km subplot of Fig. 5.
#       In our plots there is an downward wiggle in the full stokes solutions
#       on the velocity peak at ~ 3/4 x_hat shown in the paper figure. Why?


import os
import numpy
import scipy
import errno
import fnmatch

import scipy.interpolate
import matplotlib.pyplot as plt

# Location of ISMIP-HOM data
#TODO: argparse this.
ismip_data = './ismip_all'

# Location to output files
#TODO: argparse this.
out_path = './output/'

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

# The ISMIP-HOM data file headers for each experiment.
#NOTE: These aren't used and are just here for informational purposes. 
header = {'a':['x_hat','y_hat','vx_surf','vy_surf','tau_xz','tau_yz','del_p'],
          'b':['x_hat','vx_surf','vz_surf','tau_xz','del_p'],
          'c':['x_hat','y_hat','vx_surf','vy_surf','vz_surf','vx_base', 'vy_base','tau_xz','tau_yz','del_p'],
          'd':['x_hat','vx_surf','vz_surf','vx_base','tau_xz','del_p'],
          'e':['x_hat','vx_surf','vz_surf','tau_xz','del_p'],
          'f':['x_hat','y_hat','z_surf','vx','vy','vz'],
         }


def mkdir_p(path):
    """
    Make parent directories as needed and no error if existing. Works like `mkdir -p`.
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


class ismip_datum:
    """A class to hold and process each model's data"""
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
            self.array = numpy.array([])
        
        # get interpolation grid
        self.make_grid(self.E)

        # interpolate the data
        self.interp_data(self.E)

    def parse_file(self, data_file):
        """
        Parse the ismip_hom data filenames. They should look like: NNNMELLL.tex, where NNN = model name, 
             M = model number, E = experiment, LLL = three numbers denoting: 
                 experiments a-d: the length of L in km. 
                 experiment e: 000 for non-sliding and 001 for the experiment with 
                               the zone of zero basal raction
                 experiment f: the slip ratio; either 000 or 001.
        """
        code_name = str.lower(os.path.basename(os.path.splitext(data_file)[0]))
        return (code_name[0:4], code_name[4], code_name[5:]) # (model, experiment, length)

    def load_data(self):
        data = numpy.loadtxt(self.df)
        return data

    def make_grid(self, exp):
        """
        Make the interpolation grids. 

        For experiment A and C, the plots are made at y = L/4 or 1/4 y_hat. For
        experiment F, the plots are made along 1/2 x_hat. The grids then will
        always include the y = 1/4 y_hat line and the x = 1/2 x_hat line.
        """

        if exp in  ['a','c','f']:
            #NOTE: linspace(start, stop, num) returns num points across the
            #      start->stop interval, including start and stop. So, to always hit
            #      1/4 and 1/2, you need X+1 points, where X%4 == 0. 
            self.points_p_quarter = 25
            self.x_hat = numpy.linspace(0.0, 1.0, self.points_p_quarter*4+1)
            self.y_hat = numpy.linspace(0.0, 1.0, self.points_p_quarter*4+1)
            self.x_hat_grid, self.y_hat_grid = scipy.meshgrid(self.x_hat, self.y_hat)
        else:
            self.x_hat = numpy.array([])
            self.y_hat = numpy.array([])
            self.x_hat_grid = numpy.array([])
            self.y_hat_grid = numpy.array([])

    def interp_data(self, exp):
        if self.x_hat_grid.size and exp in ['a','c']:
            self.vx_surf_i = scipy.interpolate.griddata(self.array[:,0:2], self.array[:,2], (self.x_hat_grid.ravel(), self.y_hat_grid.ravel()), method='linear')
            self.vy_surf_i = scipy.interpolate.griddata(self.array[:,0:2], self.array[:,3], (self.x_hat_grid.ravel(), self.y_hat_grid.ravel()), method='linear')
            if exp in ['c']:
                self.vz_surf_i = scipy.interpolate.griddata(self.array[:,0:2], self.array[:,4], (self.x_hat_grid.ravel(), self.y_hat_grid.ravel()), method='linear')
                self.vnorm_surf_i = numpy.sqrt( numpy.square(self.vx_surf_i) + numpy.square(self.vy_surf_i) + numpy.square(self.vz_surf_i) )
            else:
                self.vnorm_surf_i = numpy.sqrt( numpy.square(self.vx_surf_i) + numpy.square(self.vy_surf_i) )
            
            self.vx_surf_i = self.vx_surf_i.reshape(self.x_hat_grid.shape)
            self.vy_surf_i = self.vy_surf_i.reshape(self.x_hat_grid.shape)
            if exp in ['c']:
                self.vz_surf_i = self.vz_surf_i.reshape(self.x_hat_grid.shape)
            self.vnorm_surf_i = self.vnorm_surf_i.reshape(self.x_hat_grid.shape)

        elif self.x_hat_grid.size and exp in ['f']:
            #NOTE: Some of the Exp. F data is reported in the scaled coordinate system 
            #      (x_hat), and others is reported in the cartesian coordinate system (x).
            #      This normalizes the coordinate systems. 
            xy_f = self.array[:,0:2]
            if xy_f[0,0] < -1.0:
                xy_f = xy_f/100.0 + 0.5
            
            self.surf_i = scipy.interpolate.griddata(   xy_f, self.array[:,2], (self.x_hat_grid.ravel(), self.y_hat_grid.ravel()), method='linear')
            self.vx_surf_i = scipy.interpolate.griddata(xy_f, self.array[:,3], (self.x_hat_grid.ravel(), self.y_hat_grid.ravel()), method='linear')
            self.vy_surf_i = scipy.interpolate.griddata(xy_f, self.array[:,4], (self.x_hat_grid.ravel(), self.y_hat_grid.ravel()), method='linear')
            self.vz_surf_i = scipy.interpolate.griddata(xy_f, self.array[:,5], (self.x_hat_grid.ravel(), self.y_hat_grid.ravel()), method='linear')
            
            self.vnorm_surf_i = numpy.sqrt( numpy.square(self.vx_surf_i) + numpy.square(self.vy_surf_i) + numpy.square(self.vz_surf_i) )
            
            self.surf_i = self.surf_i.reshape(self.x_hat_grid.shape)
            self.vx_surf_i = self.vx_surf_i.reshape(self.x_hat_grid.shape)
            self.vy_surf_i = self.vy_surf_i.reshape(self.x_hat_grid.shape)
            self.vz_surf_i = self.vz_surf_i.reshape(self.x_hat_grid.shape)
            self.vnorm_surf_i = self.vnorm_surf_i.reshape(self.x_hat_grid.shape)

    def display(self):
        print("Data file: "+self.df)
        print("Order: "+self.order)
        print("Model: "+self.M)
        print("Experiment: "+self.E)
        print("Length: "+self.L)


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


#-----------------------
# Setup the ouput files 
#-----------------------
mkdir_p(out_path)


#------------------------------------------------------------------------
# Recreate all the analysis figures in:
# Pattyn, F., et al. (2008). Benchmark experiments for higher-order and 
# full-Stokes ice sheet models (ISMIP-HOM). The Cryosphere, 2, 95--108.
# doi:10.5194/tcd-2-111-200. 
# http://www.the-cryosphere.net/2/95/2008/tc-2-95-2008.html
#------------------------------------------------------------------------
#NOTE: Exp. A and C plot at y = L/4 or 1/4 y_hat, x = [0,..,1]x_hat
#NOTE: Exp. F plots at the central flowline in the ice-flow direction
#         y = [0,..,1]y_hat, x = 1/2 x_hat

fs_data = [data for data in all_data if data.order == 'full_stokes']
ho_data = [data for data in all_data if data.order in 'higher_order']

# figure 5: Results for Exp. A: norm of the surface velocity across the bump at
# y=L/4 for different length scales L. The mean value and standard deviation are
# shown for both types of models. 
#   There are 6 plot boxes for 5,10,20,40,80,160 km, and all have: 
#       x_title     = Normalized x
#       y_title     = Velocity (m a^{-1})
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range
fs_data_a = [data for data in fs_data if data.E == 'a']
ho_data_a = [data for data in ho_data if data.E == 'a']

plt.figure(5, figsize=(10,8), dpi=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plot_ls = ['005','010','020','040','080','160']
for i, l in enumerate(plot_ls):
    a_fs_lines = numpy.array([data.vnorm_surf_i[:,data.points_p_quarter] for data in fs_data_a if data.L == l])
  
    a_fs_mean = numpy.mean(a_fs_lines,0)
    a_fs_amin = numpy.amin(a_fs_lines,0)
    a_fs_amax = numpy.amax(a_fs_lines,0)

    a_ho_lines = numpy.array([data.vnorm_surf_i[:,data.points_p_quarter] for data in ho_data_a if data.L == l])
  
    a_ho_mean = numpy.mean(a_ho_lines,0)
    a_ho_amin = numpy.amin(a_ho_lines,0)
    a_ho_amax = numpy.amax(a_ho_lines,0)

    out_data = numpy.column_stack((fs_data_a[0].x_hat, a_fs_amin, a_fs_amax, a_fs_mean, a_ho_amin, a_ho_amax, a_ho_mean))
    out_header = ['x_hat',  'full-stokes min',  'full-stokes max',  'full-stokes mean', 
                           'higher-order min', 'higher-order max', 'higher-order mean' ]
    numpy.savetxt(out_path+'ExpA_Fig5_'+l+'.txt', out_data, delimiter=',', header=','.join(out_header))

    plt.subplot(2,3,i+1)
    
    plt.fill_between(fs_data_a[0].x_hat.T, a_ho_amin, a_ho_amax, facecolor='green', alpha=0.5)
    plt.fill_between(fs_data_a[0].x_hat.T, a_fs_amin, a_fs_amax, facecolor='blue', alpha=0.5)
    
    plt.plot(fs_data_a[0].x_hat.T, a_fs_mean, 'b-', linewidth=2)
    plt.plot(fs_data_a[0].x_hat.T, a_ho_mean, 'g-', linewidth=2)

    if i+1 > 3:
        plt.xlabel('Normalized x')
    if i+1 == 1 or i+1 == 4:
        plt.ylabel('Velocity (m a$^{-1}$)')
    
    plt.title(str(int(l))+'km')

plt.savefig(out_path+'ExpA_Fig5', bbox_inches='tight')
plt.show()


# figure 8: Results for Exp. C: norm of the surface velocity at y=L/4 for
# different length scales L. The mean value and standard deviation are shown for
# both types of models. 
#   There are 6 plot boxes for 5,10,20,40,80,160 km, and all have: 
#       x_title     = Normalized x
#       y_title     = Velocity (m a^{-1})
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range
fs_data_c = [data for data in fs_data if data.E == 'c']
ho_data_c = [data for data in ho_data if data.E == 'c']

plt.figure(8, figsize=(10,8), dpi=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plot_ls = ['005','010','020','040','080','160']
for i, l in enumerate(plot_ls):
    c_fs_lines = numpy.array([data.vnorm_surf_i[:,data.points_p_quarter] for data in fs_data_c if data.L == l])
  
    c_fs_mean = numpy.mean(c_fs_lines,0)
    c_fs_amin = numpy.amin(c_fs_lines,0)
    c_fs_amax = numpy.amax(c_fs_lines,0)

    c_ho_lines = numpy.array([data.vnorm_surf_i[:,data.points_p_quarter] for data in ho_data_c if data.L == l])
  
    c_ho_mean = numpy.mean(c_ho_lines,0)
    c_ho_amin = numpy.amin(c_ho_lines,0)
    c_ho_amax = numpy.amax(c_ho_lines,0)

    out_data = numpy.column_stack((fs_data_c[0].x_hat, c_fs_amin, c_fs_amax, c_fs_mean, c_ho_amin, c_ho_amax, c_ho_mean))
    out_header = ['x_hat',  'full-stokes min',  'full-stokes max',  'full-stokes mean', 
                           'higher-order min', 'higher-order max', 'higher-order mean' ]
    numpy.savetxt(out_path+'ExpC_Fig8_'+l+'.txt', out_data, delimiter=',', header=','.join(out_header))

    plt.subplot(2,3,i+1)
    
    plt.fill_between(fs_data_c[0].x_hat.T, c_ho_amin, c_ho_amax, facecolor='green', alpha=0.5)
    plt.fill_between(fs_data_c[0].x_hat.T, c_fs_amin, c_fs_amax, facecolor='blue', alpha=0.5)
    
    plt.plot(fs_data_c[0].x_hat.T, c_fs_mean, 'b-', linewidth=2)
    plt.plot(fs_data_c[0].x_hat.T, c_ho_mean, 'g-', linewidth=2)

    if i+1 > 3:
        plt.xlabel('Normalized x')
    if i+1 == 1 or i+1 == 4:
        plt.ylabel('Velocity (m a$^{-1}$)')
    
    plt.title(str(int(l))+'km')

plt.savefig(out_path+'ExpC_Fig8', bbox_inches='tight')
plt.show()


# figure 12: Stead state surface elevation along the central flowline for Exp. F
# for the no-sliding (top) and sliding (bottom) experiment. The black line
# indicates the analytical solution [Note: I don't actually see this in the
# figure].
#   The 2 plot boxes have: 
#       x_title     = Distance from center (km)
#       y_title     = Surface (m)
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range
fs_data_f = [data for data in fs_data if data.E == 'f']
ho_data_f = [data for data in ho_data if data.E == 'f']

plt.figure(12, figsize=(10,8), dpi=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plot_ls = ['000','001']
for i, l in enumerate(plot_ls):
    f_fs_lines = numpy.array([data.surf_i[data.points_p_quarter*2,:] for data in fs_data_f if data.L == l])
  
    f_fs_mean = numpy.mean(f_fs_lines,0)
    f_fs_amin = numpy.amin(f_fs_lines,0)
    f_fs_amax = numpy.amax(f_fs_lines,0)

    f_ho_lines = numpy.array([data.surf_i[data.points_p_quarter*2,:] for data in ho_data_f if data.L == l])
  
    f_ho_mean = numpy.mean(f_ho_lines,0)
    f_ho_amin = numpy.amin(f_ho_lines,0)
    f_ho_amax = numpy.amax(f_ho_lines,0)

    plt.subplot(2,1,i+1)
   
    out_data = numpy.column_stack((fs_data_c[0].x_hat, f_fs_amin, f_fs_amax, f_fs_mean, f_ho_amin, f_ho_amax, f_ho_mean))
    out_header = ['y_hat',  'full-stokes min',  'full-stokes max',  'full-stokes mean', 
                           'higher-order min', 'higher-order max', 'higher-order mean' ]
    numpy.savetxt(out_path+'ExpF_Fig12_'+l+'.txt', out_data, delimiter=',', header=','.join(out_header))

    plt.fill_between(ho_data_f[0].y_hat.T, f_ho_amin, f_ho_amax, facecolor='green', alpha=0.5)
    plt.fill_between(fs_data_f[0].y_hat.T, f_fs_amin, f_fs_amax, facecolor='blue', alpha=0.5)
    
    plt.plot(fs_data_f[0].y_hat.T, f_fs_mean, 'b-', linewidth=2)
    plt.plot(ho_data_f[0].y_hat.T, f_ho_mean, 'g-', linewidth=2)

    if i+1 > 1:
        plt.xlabel('Distance from center (km)')
    if l == '000':
        plt.title('No-Slip Bed')
    else:
        plt.title('Slip Bed')

    plt.ylabel('Surface (m)')

plt.savefig(out_path+'ExpF_Fig12', bbox_inches='tight')
plt.show()


# figure 13: Norm of the stead state surface velocit along the central flowline for Exp. F
# for the no-sliding (top) and sliding (bottom) experiment. The black line
# indicates the analytical solution [Note: I don't actually see this in the
# figure].
#   The 2 plot boxes have: 
#       x_title     = Distance from center (km)
#       y_title     = Surface (m)
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range
plt.figure(13, figsize=(10,8), dpi=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plot_ls = ['000','001']
for i, l in enumerate(plot_ls):
    f_fs_lines = numpy.array([data.vnorm_surf_i[data.points_p_quarter*2,:] for data in fs_data_f if data.L == l])
  
    f_fs_mean = numpy.mean(f_fs_lines,0)
    f_fs_amin = numpy.amin(f_fs_lines,0)
    f_fs_amax = numpy.amax(f_fs_lines,0)

    f_ho_lines = numpy.array([data.vnorm_surf_i[data.points_p_quarter*2,:] for data in ho_data_f if data.L == l])
  
    f_ho_mean = numpy.mean(f_ho_lines,0)
    f_ho_amin = numpy.amin(f_ho_lines,0)
    f_ho_amax = numpy.amax(f_ho_lines,0)

    out_data = numpy.column_stack((fs_data_c[0].x_hat, f_fs_amin, f_fs_amax, f_fs_mean, f_ho_amin, f_ho_amax, f_ho_mean))
    out_header = ['y_hat',  'full-stokes min',  'full-stokes max',  'full-stokes mean', 
                           'higher-order min', 'higher-order max', 'higher-order mean' ]
    numpy.savetxt(out_path+'ExpF_Fig13_'+l+'.txt', out_data, delimiter=',', header=','.join(out_header))

    plt.subplot(2,1,i+1)
   
    plt.fill_between(ho_data_f[0].y_hat.T, f_ho_amin, f_ho_amax, facecolor='green', alpha=0.5)
    plt.fill_between(fs_data_f[0].y_hat.T, f_fs_amin, f_fs_amax, facecolor='blue', alpha=0.5)
    
    plt.plot(fs_data_f[0].y_hat.T, f_fs_mean, 'b-', linewidth=2)
    plt.plot(ho_data_f[0].y_hat.T, f_ho_mean, 'g-', linewidth=2)

    if i+1 > 1:
        plt.xlabel('Distance from center (km)')
    if l == '000':
        plt.title('No-Slip Bed')
    else:
        plt.title('Slip Bed')

    plt.ylabel('Velocity (m a$^{-1}$)')

plt.savefig(out_path+'ExpF_Fig13', bbox_inches='tight')
plt.show()







#----------------------------------------------------------------------------
#NOTE: Skipping these figures as CISM is only running experiments A, C and F.
#----------------------------------------------------------------------------

# figure 6: Results for Exp. B: norm of the surface velocity for different
# length scales L. The mean value and standard deviation are # shown for both
# types of models.
#   There are 6 plot boxes for 5,10,20,40,80,160 km, and all have: 
#       x_title     = Normalized x
#       y_title     = Velocity (m a^{-1})
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range


# figure 9: Results for Exp. D: norm of the surface velocity for different
# length scales L. The mean value and standard deviation are shown for both
# types of models. 
#   There are 6 plot boxes for 5,10,20,40,80,160 km, and all have: 
#       x_title     = Normalized x
#       y_title     = Velocity (m a^{-1})
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range


# figure 10: Surface velocity in the direction of the ice flow for Exp. E for
# the no-sliding (top) and sliding (bottom) experiment.
#   The 2 plot boxes have: 
#       x_title     = Normalized x
#       y_title     = Velocity (m a^{-1})
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range

# figure 11: Basal shear stress in the direction of the ice flow for Exp. E for
# the no-sliding (top) and sliding (bottom) experiment.
#   The 2 plot boxes have: 
#       x_title     = Normalized x
#       y_title     = Velocity (m a^{-1})
#       Blue line   = FS Mean
#       Blue shade  = FS range
#       Green line  = NFS mean
#       Green shade = NFS range


