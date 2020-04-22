# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:32:35 2016
Last modified, Thu Mar 28 17:51:17 CET 2019

@author: Didier M. Roche a.k.a. dmr
"""
# History:
# Changes from version 0.3 : uses d18Osw generated from the WOA salinity as input
# Changes from version 0.4 : corrected a bug on the depth axis (positive of negative)
# Changes from version 0.5 : updated FAME computations ..., lots of cleaning up
# Changes from version 0.6 : added the monthly temperature computations
# Changes from version 0.7 : added the limitation of growth to 10% maximum value growth
# Changes from version 0.8 : using version v0.5 of forams_prod_l09
# Changes from version 0.85: corrected a bug in the foram production limitation
# Changes from version 0.90: cleaned up and simplified coding
# Changes from version 0.92: using version v0.6 of forams_prod_l09
# Changes from version 0.93: using version v0.7 of forams_prod_l09 + isotopic desequilibrium for pachyderma
# Changes from version 0.94: using version v0.8 of forams_prod_l09
# Changes from version 0.95: big cleanup of the code to keep only what is stricly necessary
# Changes from version 0.96: separated module ...
# Changes from version 0.97: latest version including the change in encrustation term value for pachy_s
# Published version v1.0 --- cf. doi:10.5194/gmd-11-3587-2018
# Changes from version 1.00: included density
# Changes from version 1.10: cleaned up to account for absent density function
# Changes from version 1.11: created write_fame2 as a replacement for write_fame
# Changes from version 1.12: added the computation of the Marchitto equation for Cibicides
# Changes from version 1.13: cleaned up the cf_compliant part of the code, not needed anymore
# Changes from version 1.14: added Cibicides computation in the main code of Fame and in the output

__version__ = "1.15"

def delta_c_eq(Tc, delta_w):
    # Inputs: Tc, temperature in C, delta_w d18O water in per mil
    # The equation to be computed is the solution
    # to the Kim & O'Neil 1997 equation
    # Written as: T = 16.1 - 4.64*(delta_c-delta_w) + 0.09*(delta_c-delta_w)**2
    # Noting in the following ukn = delta_c-delta_w

    a = 0.09
    b = -4.64
    c = 16.1-Tc
    delta = b**2 - 4*a*c

    # The only likely solution is ukn_2, the other one is non-physical
    # ukn_1 = (b*-1.0 + delta**(0.5))/(2*a)

    ukn_2 = (b*-1.0 - delta**(0.5))/(2*a)

    return ukn_2+delta_w-0.27
#enddef delta_c_eq

def delta_c_mar(Tc, delta_w):
    # Inputs: Tc, temperature in C, delta_w d18O water in per mil
    # Calcite equilibrium as computed with the Marchitto equation
    # equation for Cibicides d18O (Marchitto et al., 2014)*
    # d_c = (d_w - 0.27) + 3.58 - 0.245*T + 0.0011 T^2
    # *ref: Marchitto et al. Improved oxygen isotope temperature calibrations for cosmopolitan benthic foraminifera. Geochimica et Cosmochimica Acta 130, 1-11 (2014).

    return (delta_w - 0.27) + 3.58 - 0.245 * Tc + 0.0011 * Tc**2
#enddef delta_c_mar

# from http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_closest(A,target) :
    import numpy as np
    return (np.abs(A-target)).argmin()
#enddef find_closest

def famed(woa_oxyg_dh,woa_oxyg_dh_m,temp_cl,temp_cl_m,depth,woa_rhom=None):

    import numpy as np
    from numpy import ma

# Main FAME computation ...
# =========================

#	INPUTS FOR THE FAME CALCULATIONS
#		woa_oxyg_dh   == d18sw       in seawater <time mean> shape is [102, 180, 360]    e.g. depth, lat, lon
#		woa_oxyg_dh_m == d18sw       in seawater monthly     shape is [12,57, 180, 360]  e.g. depth, lat, lon
#		temp_cl       == temperature in seawater <time mean> shape is [1, 102, 180, 360] e.g. time (degenerate), depth, lat, lon
#		temp_cl_m     == temperature in seawater monthly     shape is [12, 57, 180, 360] e.g. time             , depth, lat, lon
#		depth         == depth of the levels in meters assumed positive here

    # FAME is coded in Kelvins ...
    temp_kl = temp_cl + 273.15
    temp_kl_m = temp_cl_m + 273.15

    # Computation of equilibrium calcite from WOA fields ...
    delt_dh_init = delta_c_eq(temp_cl[...],woa_oxyg_dh)
    delt_dh_init_m = delta_c_eq(temp_cl_m,woa_oxyg_dh_m)


    # Added the computation of the equation of Marchitto everywhere, no weighting.

    #  Probably useless, only written for compatibility
    #~ d18Oca_mar   =  ma.mean(delta_c_mar(temp_cl  ,woa_oxyg_dh),axis=0)

    #  Actual calculation here
    d18Oca_mar_m =  ma.mean(delta_c_mar(temp_cl_m,woa_oxyg_dh_m),axis=0)

    if not ( woa_rhom is None ):
       # [DENSITY] -- addition of density from WOA
       rho_m = woa_rhom
    #endif

    # i.e. auld method, d18Oca averaged over 50 meters ... == "Lukas Jonkers" methodology
    depth_50m = find_closest(depth,50.0)
    depth_00m = find_closest(depth, 0.0)

    indx_less_50m = depth <= 50.0

    #~ if depth_50m == depth_00m : depth_50m += 1

    # NOTA: all *_lj variables have a dimension without time and depth
    #       e.g. [180,360], lat, lon

    d18Osw_lj = ma.mean(woa_oxyg_dh[indx_less_50m,...],axis=0)
    tempcl_lj = ma.mean(temp_cl[indx_less_50m,...],axis=0)
    d18Oca_lj = delta_c_eq(tempcl_lj,d18Osw_lj)

    d18Osw_ol = woa_oxyg_dh[depth_00m,...]
    tempcl_ol = temp_cl[depth_00m,...]
    d18Oca_ol = delta_c_eq(tempcl_ol,d18Osw_ol)

    if not ( woa_rhom is None ):
        # [DENSITY] -- addition of density from WOA
        rhom_ol = ma.mean(woa_rhom[:,depth_00m,...],axis=0)
    #endif

    import forams_prod_l09 as fpl

    # Maximum shape of result: nb_forams, lat, lon
    max_shape_final = (len(fpl.l09_cnsts_dic),)+d18Osw_ol.shape

    # Create a placeholder for the foram result
    delt_forams = ma.zeros(max_shape_final,np.float32)

    if not ( woa_rhom is None ):
       # [DENSITY] -- addition of density from WOA
       rhom_forams = ma.zeros(max_shape_final,np.float32)
    #endif

    for foram_specie in fpl.l09_cnsts_dic :

        # Rate of growth from the Lombard et al., 2009 methodology
        foram_growth = fpl.growth_rate_l09_array(foram_specie,temp_kl[...])
        foram_growth_m = fpl.growth_rate_l09_array(foram_specie,temp_kl_m)

        # Get the depth of the STD living foram in FAME
        f_dept = fpl.get_living_depth(foram_specie)

        # Find this depth as an index in the array water column
        indx_dfm = find_closest(depth,abs(float(f_dept[0])))
        #~ if indx_dfm == depth_00m : indx_dfm += 1

        indx_less_dfm = depth <= np.abs(float(f_dept[0]))

        # Shrink the FAME arrays to the foram living depth
        foram_growth = foram_growth[indx_less_dfm,...] # shape is depth, lat, lon
        foram_growth_m = foram_growth_m[:,indx_less_dfm,...] # shape is time, depth, lat, lon

        # Do the same for the equilibrium calcite from WOA
        delt_dh = delt_dh_init[indx_less_dfm,...] # idem
        delt_dh_m = delt_dh_init_m[:,indx_less_dfm,...] # idem

        # [DENSITY] -- addition of density from WOA
        if not ( woa_rhom is None ):
           rho_m_specie = rho_m[:,indx_less_dfm,...] # idem
        #endif

        # Get the location where there is SOME growth, based on a certain epsilon
        epsilon_growth = 0.1*fpl.l09_maxgrowth_dic[foram_specie][0] # or 0.032

        # Mask out the regions where the foram_growth is less than the epsilon
        masked_f_growth = ma.masked_less_equal(foram_growth,epsilon_growth)

        #~ nb_points_growth = (masked_f_growth * 0.0 + 1.0).filled(0.0)
        #~ if monthly is True:
           #~ nb_points_growth_m = (masked_f_growth_m * 0.0 + 1.0).filled(0.0)

        #~ nb_points_growth = ma.where(foram_growth > epsilon_growth,1,0) # 0.000001
        #~ if monthly is True:
           #~ nb_points_growth_m = ma.where(foram_growth_m > epsilon_growth,1,0) # 0.000001

        # Now sum the growth over the depth ...
        f_growth = ma.sum(masked_f_growth,axis=0)
        #~ n_growth = ma.where(ma.sum(nb_points_growth,axis=0)>0,1,0) # axis 0 = depth

        masked_f_growth_m = ma.masked_less_equal(foram_growth_m,epsilon_growth)
        f_growth_m = ma.sum(ma.sum(masked_f_growth_m,axis=1),axis=0)
        location_max_foramprod = ma.argmax(masked_f_growth_m,axis=1)
        location_max_foramprod = ma.masked_array(location_max_foramprod,mask=masked_f_growth_m[:,:,...].mask.all(axis=1))
        # location_max_foramprod = ma.masked_array(location_max_foramprod,mask=masked_f_growth_m[:,0,...].mask)

        # Computing the weighted sum for d18Ocalcite using growth over depth
        delt_fp = ma.sum(delt_dh*masked_f_growth,axis=0)
        delt_fp_m = ma.sum(ma.sum(delt_dh_m*masked_f_growth_m,axis=1),axis=0)

        # [DENSITY] -- addition of density from WOA
        if not ( woa_rhom is None ):
           rho_fp_m  = ma.sum(ma.sum(rho_m_specie*masked_f_growth_m,axis=1),axis=0)
        #endif

        # Mask out the points where no growth occur at all, in order to avoid NaNs ...
        delt_fp = delt_fp / ma.masked_less_equal(f_growth,0.0)
        delt_fp_m = delt_fp_m / ma.masked_less_equal(f_growth_m,0.0)
        if not ( woa_rhom is None ):
           # [DENSITY] -- addition of density from WOA
           rho_fp_m = rho_fp_m / ma.masked_less_equal(f_growth_m,0.0)
        #endif

        # Result of FAME
        Z_om_fm = delt_fp
        Z_om_fm_m = ma.masked_array(delt_fp_m,mask=ma.max(location_max_foramprod[:,...],axis=0).mask)

        # [DENSITY] -- addition of density from WOA
        if not ( woa_rhom is None ):
           Z_om_rho_m = ma.masked_array(rho_fp_m,mask=ma.max(location_max_foramprod[:,...],axis=0).mask)
        #endif

        if foram_specie == "pachy_s":
            Z_om_fm = Z_om_fm + 0.1 # in per mil
            Z_om_fm_m = Z_om_fm_m + 0.1 # in per mil

        index_for = list(fpl.l09_cnsts_dic.keys()).index(foram_specie)
        delt_forams[index_for,...] = Z_om_fm_m

        # [DENSITY] -- addition of density from WOA
        if not ( woa_rhom is None ):
           rhom_forams[index_for,...] = Z_om_rho_m
        #endif

    #endfor on foram_specie

    # For comparison with Lukas Jonkers: old method on first 50 meters
    Z_om_lj = d18Oca_lj

    # For comparison with previous figures: old method on first 00 meters
    Z_om_ol = d18Oca_ol

    # [DENSITY] -- addition of density from WOA
    if not ( woa_rhom is None ):
       print("Fame is used with density ...")
       return delt_forams, Z_om_ol, d18Oca_mar_m, rhom_forams, rhom_ol
    else:
       print("Fame is used without density ...")
       return delt_forams, Z_om_ol, d18Oca_mar_m
    #endif

#end def famed

def write_fame2(calc_benthos,equi_calc,resultats_fame, density=None,nc_out="out-test.nc"):

    import numpy as np
    import netCDF4
    import forams_prod_l09 as fpl

    # Write out the variables thus created

    dst = netCDF4.Dataset(nc_out,'r+',format='NETCDF4_CLASSIC')
    #~ dst.set_fill_on()

    import time
    dst.history = 'Created ' + time.ctime(time.time()) \
                      + ' by FAME (v'+__version__+') (dmr,jyp,cw)'

    dst.sync()
    dst.close()

    fill_value = netCDF4.default_fillvals["f4"]

    import nc_utils as ncu
    ncu.write_variable(equi_calc,"d18Oc_equ",nc_out)
    ncu.write_variable(calc_benthos,"d18Oc_cib",nc_out)

    for foram_species in fpl.l09_cnsts_dic :

        indx_f = list(fpl.l09_cnsts_dic.keys()).index(foram_species)
        ncu.write_variable(resultats_fame[indx_f,...],"d18Oc_"+foram_species,nc_out)

    return

#end def write_fame2

def write_fame(nc_in, equi_calc,resultats_fame, nc_out="out-test.nc", latvar="lat",lonvar="lon",depthvar="depth"):

    import numpy as np
    import netCDF4
    import forams_prod_l09 as fpl

    # Write out the variables thus created

    dst = netCDF4.Dataset(nc_out,'w',format='NETCDF4_CLASSIC')
    dst.set_fill_on()

    # dmr --- use a modified solution from:
    #         CITE: http://stackoverflow.com/questions/15141563/python-netcdf-making-a-copy-of-all-variables-and-attributes-but-one
    with netCDF4.Dataset(nc_in,'r') as src:
        for nm_dim, dimension in src.dimensions.items():

            if nm_dim in [latvar,lonvar]:
               dst.createDimension(nm_dim,
                   len(dimension) if not dimension.isunlimited() else None)
               try:
                   v_dim = src.variables[nm_dim]
               except:
                   print("No data associated with variable", nm_dim)

               t_var = dst.createVariable(nm_dim,v_dim.datatype,v_dim.dimensions)

               stdnm, stdunit, stdadd, stdmul = make_cf_compliant(nm_dim)

               t_var.units = stdunit
               t_var.standard_name = stdnm

               if ( stdmul != "1.0" ):
                   temp = v_dim[:]*float(stdmul)
               else:
                   temp = v_dim[:]
               #endif
               if ( stdadd != "1.0" ):
                   t_var[:] = temp[:]+float(stdadd)
               else:
                   t_var[:] = temp[:]
               #endif
            #endif on nm_dim
        #endfor on nm_dim
    #endwith

    import time
    dst.history = 'Created ' + time.ctime(time.time()) \
                      + ' by FAME (v'+__version__+') (dmr,jyp,cw)'

    var_dimout = ()

    for key in (latvar,lonvar):
        var_dimout += (dst.dimensions[key].name,)

    fill_value = netCDF4.default_fillvals["f4"]

    local_var = dst.createVariable(
                "d18Oc_std",np.float32().dtype,
                var_dimout, fill_value=fill_value
                                  )
    local_var.units = "per mil versus PDB"
    local_var.standard_name = "delta oxygen-18 of inorganic calcite"
    local_var[:] = equi_calc.filled(fill_value)


    for foram_species in fpl.l09_cnsts_dic :
    # for foram_species in ["dutertrei",] :
      local_var = dst.createVariable(
                  "d18Oc_"+foram_species,np.float32().dtype,
                  var_dimout, fill_value=fill_value
                                    )
      local_var.units = "per mil versus PDB"
      local_var.standard_name = "delta oxygen-18 of calcite in "+ foram_species

      indx_f = list(fpl.l09_cnsts_dic.keys()).index(foram_species)
      local_var[:] = resultats_fame[indx_f,...].filled(fill_value)

      #~ n_var = dst.createVariable(
                  #~ "nmg_"+foram_species,v_list[0].datatype,
                  #~ var_dimout, fill_value=fill_value
                                    #~ )
      #~ n_var.units = "month"
      #~ n_var.standard_name = "number months of non-zero growth "+ foram_species

      #~ n_var[:] = ngrowth_forams[indx_f,...].filled(fill_value)

      dst.sync()

    dst.close()

    return

#end def write_fame

# End of main FAME computation ...
# ================================

# The End of All Things (op. cit.)
