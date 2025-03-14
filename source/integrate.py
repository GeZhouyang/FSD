# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
from hoomd.PSEv3 import _PSEv3
from hoomd.PSEv3 import shear_function

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
import hoomd
from hoomd import _hoomd
from hoomd import compute
from hoomd.md import _md

import math

## One step overdamped integration with hydrodynamic interactions

class PSEv3(hoomd.md.integrate._integration_method):
    ## Specifies the integrator for Fast Stokesian Dynamics (FSD)
    #
    # group             Group of particles on which to apply this method.
    # T                 Temperature of the simulation (in energy units)
    # seed              Random seed to use for the run. Simulations that are identical, except for the seed, will follow different trajectories.
    # xi                Ewald splitting parameter
    # error 		Error threshold to use for calculations (Spectral Ewald parameters are determined on the fly using this bound)
    # function_form 	Time dependent shear rate
    # max_strain	Maximum strain of the box
    # fileprefix	Prefix for stresslet output
    # period		Frequency of stresslet output
    # nlist_type	Type of neighbor list to use
    # friction_type	Type of friction to add
    # h0		Maximum distance of frictional contact
    # alpha 		List of frictional magnitudes
    # ndsr              Non-dimensional shear rate (ratio of Stokes drag and max electrostatic repulsion)
    # kappa             inverse Debye length          
    # k_n               collision spring constant      
    #
    # T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # Internally, a compute.thermo is automatically specified and associated with a group.
    def __init__(self,
                 group,
                 T,
                 seed = 0,
                 xi = 0.5,
                 error = 0.001,
                 function_form = None,
                 max_strain = 0.5,
                 fileprefix="stresslet",
                 period = 0,
                 nlist_type = "cell",
                 friction_type = "none",
                 h0 = 0.0,
                 alpha = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                 ndsr = 1e-1, kappa = 1/0.05, k_n = 1e4, beta_AF = 0., epsq = 1e-5, sqm_B1=0., sqm_B2=0.,
                 N_mix=1, coef_B1_mask=1.0, coef_B2_mask=1.0, rot_diff=0., T_ext=0., omega_ext=0.):
        
        hoomd.util.print_status_line();
        
        # initialize base class
        hoomd.md.integrate._integration_method.__init__(self);
        
        # setup the variant inputs
        T = hoomd.variant._setup_variant_input(T);
        
        # Make sure the period is an integer
        period = int( period );
        
        # create the compute thermo
        compute._get_unique_thermo(group=group);
        
        # Cutoff distance for real space Ewald Sums
        self.rcut = math.sqrt( - math.log( error ) ) / xi;
        
        # Initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.contex.msg.error("Sorry, we have not written CPU code for Fast Stokesian Dynamics. \n");
            raise RuntimeError('Error creating Stokes');
        else:

            # Create a neighborlist exclusively for real space interactions. Use cell lists by 
            # default, but also allow the user to specify
            if ( nlist_type.upper() == "CELL" ):

                cl_stokes = _hoomd.CellListGPU(hoomd.context.current.system_definition);
                hoomd.context.current.system.addCompute(cl_stokes, "stokes_cl")
                self.nlist_ewald = _md.NeighborListGPUBinned(hoomd.context.current.system_definition, self.rcut, 0.4, cl_stokes);

            elif ( nlist_type.upper() == "TREE" ):

                self.nlist_ewald = _md.NeighborListGPUTree(hoomd.context.current.system_definition, self.rcut, 0.4)

            elif ( nlist_type.upper() == "STENCIL" ):

                cl_stokes  = _hoomd.CellListGPU(hoomd.context.current.system_definition)
                hoomd.context.current.system.addCompute(cl_stokes, "stokes_cl")
                cls_stokes = _hoomd.CellListStencil( hoomd.context.current.system_definition, cl_stokes )
                hoomd.context.current.system.addCompute( cls_stokes, "stokes_cls")
                self.nlist_ewald = _md.NeighborListGPUStencil(hoomd.context.current.system_definition, self.rcut, 0.4, cl_stokes, cls_stokes)

            else:
                hoomd.context.msg.error("Invalid neighborlist method specified. Valid options are: cell, tree, stencil. \n");
                raise RuntimeError('Error constructing neighborlist');
         
            # Set the neighbor list properties
            self.nlist_ewald.setEvery(1, True);
            hoomd.context.current.system.addCompute(self.nlist_ewald, "stokes_nlist_ewald")
            self.nlist_ewald.countExclusions();
            
            # Initialize Stokes Class
            self.cpp_method = _PSEv3.Stokes(hoomd.context.current.system_definition,
                                            group.cpp_group,
                                            T.cpp_variant,
                                            seed,
                                            self.nlist_ewald,
                                            xi,
                                            error,
                                            fileprefix,
                                            period,
                                            ndsr, kappa, k_n, beta_AF, epsq, sqm_B1, sqm_B2,
                                            N_mix, coef_B1_mask, coef_B2_mask, rot_diff, T_ext, omega_ext);  ##zhoge
        
        self.cpp_method.validateGroup()
        
        # Set shear conditions if necessary
        if function_form is not None:
            self.cpp_method.setShear(function_form.cpp_function, max_strain)
        else:
            no_shear_function = shear_function.steady(dt = 0)
            self.cpp_method.setShear(no_shear_function.cpp_function, max_strain)
        
        # Set up the parameters and resistance functions before running the simulation
        self.cpp_method.setParams()
        self.cpp_method.setResistanceTable()
        self.cpp_method.setFriction(friction_type.upper(), h0, alpha) # must be called AFTER setResistanceTable()
        self.cpp_method.setSparseMath()
        self.cpp_method.AllocateWorkSpaces()
    
    ## Changes parameters of an existing integrator
    def set_params(self, T=None, function_form = None, max_strain=0.5):
        hoomd.util.print_status_line();
        self.check_initialization();
    
        if T is not None:
            # setup the variant inputs
            T = hoomd.variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
    
        if function_form is not None:
            self.cpp_method.setShear(function_form.cpp_function, max_strain)
    
    ## Stop any shear
    def stop_shear(self, max_strain = 0.5):
        no_shear_function = shear_function.steady(dt = 0)
        self.cpp_method.setShear(no_shear_function.cpp_function, max_strain)
