import json
import numpy as np
import os
from matplotlib import pyplot as plt

def measure_rho(data, max_sep, max_mag, tag=None, use_xy=False, cattype='hsm',
                alt_tt=False, opt=None, subtract_mean=False, do_rho0=False):
    """Compute the rho statistics
    """
    import treecorr

    if max_mag > 0:
        zeropt=30.
        m = zeropt- 2.5*np.log10(np.array(data['FLUX']))
        data = data[m < max_mag]

    if cattype == 'hsm':
        e1 = data['G1_DATA']
        e2 = data['G2_DATA']
        T = data['T_DATA']
        p_e1 = data['G1_MODEL']
        p_e2 = data['G2_MODEL']
        p_T = data['T_MODEL']
    elif cattype == 'ngmix':
        e1 = data['STAR_E1']
        e2 = data['STAR_E2']
        T = data['STAR_T']
        p_e1 = data['MODEL_E1']
        p_e2 = data['MODEL_E2']
        p_T = data['MODEL_T']
    else:
        print('Invalid cattype: ', cattype)
        exit

    q1 = e1-p_e1
    q2 = e2-p_e2
    dt = (T-p_T)/T
    w1 = e1 * dt
    w2 = e2 * dt
    print('mean e = ',np.mean(e1),np.mean(e2))
    print('std e = ',np.std(e1),np.std(e2))
    print('mean T = ',np.mean(T))
    print('std T = ',np.std(T))
    print('mean de = ',np.mean(q1),np.mean(q2))
    print('std de = ',np.std(q1),np.std(q2))
    print('mean dT = ',np.mean(T-p_T))
    print('std dT = ',np.std(T-p_T))
    print('mean dT/T = ',np.mean(dt))
    print('std dT/T = ',np.std(dt))
    if subtract_mean:
        e1 -= np.mean(e1)
        e2 -= np.mean(e2)
        q1 -= np.mean(q1)
        q2 -= np.mean(q2)
        w1 -= np.mean(w1)
        w2 -= np.mean(w2)
        dt -= np.mean(dt)

    if use_xy:
        x = data['X']
        y = data['Y']
        print('x = ',x)
        print('y = ',y)

        ecat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=e1, g2=e2)
        qcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=q1, g2=q2)
        wcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=w1, g2=w2, k=dt)
    else:
        ra = data['RA']
        dec = data['DEC']
        print('ra = ',ra)
        print('dec = ',dec)

        ecat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2)
        qcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=q1, g2=q2)
        wcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=w1, g2=w2, k=dt)

    ecat.name = 'ecat'
    qcat.name = 'qcat'
    wcat.name = 'wcat'
    if tag is not None:
        for cat in [ ecat, qcat, wcat ]:
            cat.name = tag + ":"  + cat.name

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = max_sep,
        bin_size = 0.2,
    )

    if opt == 'lucas':
        bin_config['min_sep'] = 2.5
        bin_config['max_sep'] = 250.
        bin_config['nbins'] = 20
        del bin_config['bin_size']

    if opt == 'fine_bin':
        bin_config['min_sep'] = 0.1
        bin_config['max_sep'] = 2000.
        bin_config['bin_size'] = 0.01


    pairs = [ (qcat, qcat),
              (ecat, qcat),
              (wcat, wcat),
              (qcat, wcat),
              (ecat, wcat) ]
    if do_rho0:
        pairs.append( (ecat, ecat) )
    results = []
    for (cat1, cat2) in pairs:
        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))

        rho = treecorr.GGCorrelation(bin_config, verbose=2)

        if cat1 is cat2:
            rho.process(cat1)
        else:
            rho.process(cat1, cat2)
        print('mean xi+ = ',rho.xip.mean())
        print('mean xi- = ',rho.xim.mean())
        results.append(rho)

    if alt_tt:
        print('Doing alt correlation of %s vs %s'%(dtcat.name, dtcat.name))

        rho = treecorr.KKCorrelation(bin_config, verbose=2)
        rho.process(dtcat)
        results.append(rho)

    return results


def write_stats(stat_file, rho1, rho2, rho3, rho4, rho5, rho0=None, tau0=None, tau2=None, tau5=None, corr_tt=None):

    stats = [
        rho1.meanlogr.tolist(),
        rho1.xip.tolist(),
        rho1.xip_im.tolist(),
        rho1.xim.tolist(),
        rho1.xim_im.tolist(),
        rho1.varxip.tolist(),
        rho1.varxim.tolist(),
        rho2.xip.tolist(),
        rho2.xip_im.tolist(),
        rho2.xim.tolist(),
        rho2.xim_im.tolist(),
        rho2.varxip.tolist(),
        rho2.varxim.tolist(),
        rho3.xip.tolist(),
        rho3.xip_im.tolist(),
        rho3.xim.tolist(),
        rho3.xim_im.tolist(),
        rho3.varxip.tolist(),
        rho3.varxim.tolist(),
        rho4.xip.tolist(),
        rho4.xip_im.tolist(),
        rho4.xim.tolist(),
        rho4.xim_im.tolist(),
        rho4.varxip.tolist(),
        rho4.varxim.tolist(),
        rho5.xip.tolist(),
        rho5.xip_im.tolist(),
        rho5.xim.tolist(),
        rho5.xim_im.tolist(),
        rho5.varxip.tolist(),
        rho5.varxim.tolist(),
    ]
    if rho0 is not None:
        stats.extend([
            rho0.xip.tolist(),
            rho0.xip_im.tolist(),
            rho0.xim.tolist(),
            rho0.xim_im.tolist(),
            rho0.varxip.tolist(),
            rho0.varxim.tolist(),
        ])
    if tau0 is not None:
        stats.extend([
            tau0.xip.tolist(),
            tau0.xip_im.tolist(),
            tau0.xim.tolist(),
            tau0.xim_im.tolist(),
            tau0.varxip.tolist(),
            tau0.varxim.tolist(),
        ])
    if tau2 is not None:
        stats.extend([
            tau2.xip.tolist(),
            tau2.xip_im.tolist(),
            tau2.xim.tolist(),
            tau2.xim_im.tolist(),
            tau2.varxip.tolist(),
            tau2.varxim.tolist(),
        ])
    if tau5 is not None:
        stats.extend([
            tau5.xip.tolist(),
            tau5.xip_im.tolist(),
            tau5.xim.tolist(),
            tau5.xim_im.tolist(),
            tau5.varxip.tolist(),
            tau5.varxim.tolist(),
        ])
    if corr_tt is not None:
        stats.extend([
            corr_tt.xi.tolist(),
            corr_tt.varxi.tolist()
        ])
    #print('stats = ',stats)
    print('stat_file = ',stat_file)
    with open(stat_file,'w') as fp:
        json.dump([stats], fp)
    print('Done writing ',stat_file)

def measure_tau_mpi(star_data, gal_data, patch_centers, min_sep=0.5, max_sep=250,
                    tau0=True, tau2=True, tau5=True, output_dir=None, version=None):
    """
    Compute the tau statistics using multiprocessing and low-memory usage options.
    """
    import treecorr
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Setup catalog and correlation configs
    econfig = dict(
        ra_col='RA',
        dec_col='DEC',
        ra_units='deg',
        dec_units='deg',
        g1_col='G1_DATA_MEANSUB',
        g2_col='G2_DATA_MEANSUB',
        patch_centers = patch_centers
    )
    
    qconfig = dict(
        ra_col='RA',
        dec_col='DEC',
        ra_units='deg',
        dec_units='deg',
        g1_col='DELTA_G1_MEANSUB',
        g2_col='DELTA_G2_MEANSUB',
        patch_centers = patch_centers
    )

    wconfig = dict(
        ra_col='RA',
        dec_col='DEC',
        ra_units='deg',
        dec_units='deg',
        g1_col='G1_X_DELTAT_MEANSUB',
        g2_col='G2_X_DELTAT_MEANSUB',
        patch_centers = patch_centers
    )

    galconfig = dict(
        ra_col='ra',
        dec_col='dec',
        ra_units='deg',
        dec_units='deg',
        g1_col='g1',
        g2_col='g2',
        patch_centers = patch_centers
    )

    ggconfig = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,
        min_sep = min_sep,
        max_sep = max_sep,
        bin_size = 0.2,
        var_method='bootstrap'
    )

    print('Loading catalogs as treecorr Catalogs.')
    ecat = treecorr.Catalog(star_data, econfig)
    qcat = treecorr.Catalog(star_data, qconfig)
    wcat = treecorr.Catalog(star_data, wconfig)
    gcat = treecorr.Catalog(gal_data, galconfig)

    ecat.name = 'ecat'
    qcat.name = 'qcat'
    wcat.name = 'wcat'
    gcat.name = 'gcat'

    pairs = [ (gcat, ecat),
              (gcat, qcat),
              (gcat, wcat) ]
    
    results = []

    if tau0:
        (cat1, cat2) = pairs[0]
        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))

        gg = treecorr.GGCorrelation(ggconfig, verbose=2)

        gg.process(cat1, cat2, low_mem=True, comm=comm)
        
        if rank == 0:
            print('mean xi+ = ',gg.xip.mean())
            print('mean xi- = ',gg.xim.mean())
            stats_fn = os.path.join(output_dir,
                'tau0_stats_%s.fits'%(version))
            print('Writing stats to: ', stats_fn)
            gg.write(stats_fn, write_patch_results=True)
            results.append(gg)

    if tau2:
        (cat1, cat2) = pairs[1]
        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))

        gg = treecorr.GGCorrelation(ggconfig, verbose=2)

        gg.process(cat1, cat2, low_mem=True, comm=comm)
        
        if rank == 0:
            print('mean xi+ = ',gg.xip.mean())
            print('mean xi- = ',gg.xim.mean())
            stats_fn = os.path.join(output_dir,
                'tau2_stats_%i.fits'%(version))
            print('Writing stats to: ', stats_fn)
            gg.write(stats_fn, write_patch_results=True)
            results.append(gg)

    if tau5:
        (cat1, cat2) = pairs[2]
        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))

        gg = treecorr.GGCorrelation(ggconfig, verbose=2)

        gg.process(cat1, cat2, low_mem=True, comm=comm)
        
        if rank == 0:
            print('mean xi+ = ',gg.xip.mean())
            print('mean xi- = ',gg.xim.mean())
            stats_fn = os.path.join(output_dir,
                'tau5_stats_%i.fits'%(version))
            print('Writing stats to: ', stats_fn)
            gg.write(stats_fn, write_patch_results=True)
            results.append(gg)

#    for (cat1, cat2) in pairs:
#
#        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))
#
#        gg = treecorr.GGCorrelation(ggconfig, verbose=2)
#
#        gg.process(cat1, cat2, low_mem=True, comm=comm)
#        
#        if rank == 0:
#           print('mean xi+ = ',gg.xip.mean())
#            print('mean xi- = ',gg.xim.mean())
#            gg.write(os.path.join(output_dir, '%s_%s_ggcorr_stats_1.fits'),
#                     write_patch_results=True)
#            results.append(gg)

    return results


def measure_tau(star_data, gal_data, max_sep, max_mag, tag=None, use_xy=False, cattype='hsm', prefix='piff',
                alt_tt=False, opt=None, subtract_mean=False):
    """Compute the rho statistics
    """
    import treecorr
    
    if max_mag > 0:
        zeropt=30.
        m = zeropt- 2.5*np.log10(np.array(star_data['FLUX']))
        star_data = star_data[m < max_mag]
    
    if cattype == 'hsm':
        e1 = star_data['G1_DATA']
        e2 = star_data['G2_DATA']
        T = star_data['T_DATA']
        p_e1 = star_data['G1_MODEL']
        p_e2 = star_data['G2_MODEL']
        p_T = star_data['T_MODEL']
    elif cattype == 'ngmix':
        e1 = star_data['STAR_E1']
        e2 = star_data['STAR_E2']
        T = star_data['STAR_T']
        p_e1 = star_data['MODEL_E1']
        p_e2 = star_data['MODEL_E2']
        p_T = star_data['MODEL_T']

    # e1 = star_data['obs_e1']
    # e2 = star_data['obs_e2']
    # T = star_data['obs_T']
    # p_e1 = star_data[prefix+'_e1']
    # p_e2 = star_data[prefix+'_e2']
    # p_T = star_data[prefix+'_T']
    # m = star_data['mag']

    g_e1 = gal_data['g1']
    g_e2 = gal_data['g2']
    # g_T = gal_data['gal_T']

    q1 = e1-p_e1
    q2 = e2-p_e2
    dt = (T-p_T)/T
    w1 = e1 * dt
    w2 = e2 * dt
    print('mean e = ',np.mean(e1),np.mean(e2))
    print('std e = ',np.std(e1),np.std(e2))
    print('mean T = ',np.mean(T))
    print('std T = ',np.std(T))
    print('mean de = ',np.mean(q1),np.mean(q2))
    print('std de = ',np.std(q1),np.std(q2))
    print('mean dT = ',np.mean(T-p_T))
    print('std dT = ',np.std(T-p_T))
    print('mean dT/T = ',np.mean(dt))
    print('std dT/T = ',np.std(dt))

    if subtract_mean:
        e1 -= np.mean(e1)
        e2 -= np.mean(e2)
        q1 -= np.mean(q1)
        q2 -= np.mean(q2)
        w1 -= np.mean(w1)
        w2 -= np.mean(w2)
        dt -= np.mean(dt)

    if use_xy:
        x = star_data['fov_x']
        y = star_data['fov_y']
        if max_mag > 0:
            x = x[m<max_mag]
            y = y[m<max_mag]
        print('x = ',x)
        print('y = ',y)

        ecat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=e1, g2=e2)
        qcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=q1, g2=q2)
        wcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=w1, g2=w2, k=dt)
    else:
        ra = star_data['RA']
        dec = star_data['DEC']
        ra_gal = gal_data['ra']
        dec_gal = gal_data['dec']
        if max_mag > 0:
            ra = ra[m<max_mag]
            dec = dec[m<max_mag]
        print('ra = ',ra)
        print('dec = ',dec)

        ecat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2)
        qcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=q1, g2=q2)
        wcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=w1, g2=w2, k=dt)
        gcat = treecorr.Catalog(ra=ra_gal, dec=dec_gal, ra_units='deg', dec_units='deg', g1=g_e1, g2=g_e2)

    ecat.name = 'ecat'
    qcat.name = 'qcat'
    wcat.name = 'wcat'
    gcat.name = 'gcat'
    if tag is not None:
        for cat in [ ecat, qcat, wcat, gcat ]:
            cat.name = tag + ":"  + cat.name

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = max_sep,
        bin_size = 0.2,
    )

    if opt == 'lucas':
        bin_config['min_sep'] = 2.5
        bin_config['max_sep'] = 250.
        bin_config['nbins'] = 20
        del bin_config['bin_size']

    if opt == 'fine_bin':
        bin_config['min_sep'] = 0.1
        bin_config['max_sep'] = 2000.
        bin_config['bin_size'] = 0.01


    pairs = [ (gcat, ecat),
              (gcat, qcat),
              (gcat, wcat) ]
    
    results = []
    for (cat1, cat2) in pairs:
        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))

        gg = treecorr.GGCorrelation(bin_config, verbose=2)
        gg.process(cat1, cat2)
        print('mean xi+ = ',gg.xip.mean())
        print('mean xi- = ',gg.xim.mean())
        results.append(gg)

#     if alt_tt:
#         print('Doing alt correlation of %s vs %s'%(dtcat.name, dtcat.name))

#         gg = treecorr.KKCorrelation(bin_config, verbose=2)
#         gg.process(dtcat)
#         results.append(gg)

    return results

def write_stats_tau(stat_file, tau0, tau2, tau5):

    stats = [
        tau0.meanlogr.tolist(),
        tau0.xip.tolist(),
        tau0.xip_im.tolist(),
        tau0.xim.tolist(),
        tau0.xim_im.tolist(),
        tau0.varxip.tolist(),
        tau0.varxim.tolist(),
        tau2.xip.tolist(),
        tau2.xip_im.tolist(),
        tau2.xim.tolist(),
        tau2.xim_im.tolist(),
        tau2.varxip.tolist(),
        tau2.varxim.tolist(),
        tau5.xip.tolist(),
        tau5.xip_im.tolist(),
        tau5.xim.tolist(),
        tau5.xim_im.tolist(),
        tau5.varxip.tolist(),
        tau5.varxim.tolist(),
    ]
    
    #print('stats = ',stats)
    print('stat_file = ',stat_file)
    with open(stat_file,'w') as fp:
        json.dump([stats], fp)
    print('Done writing ',stat_file)

def pretty_rho1(meanr, rho, sig, sqrtn, rho3=None, sig3=None, rho4=None, sig4=None, gband=False):
    import matplotlib.patches as mp
    if False:
        # This is all handwavy arguments about what the requirements are.  
        # I'm taking Cathering's CFHTLS xi+ values of 1.e-4 at 1 arcmin, 2e-6 at 40 arcmin.
        # Then I'm saying our requirements on rho need to be about 0.16 times this for SV (S/N=6),
        # but more like 0.03 times this for Y5.
        t1 = 0.5
        t2 = 300.
        xa = numpy.log(1.0)
        xb = numpy.log(40.0)
        ya = numpy.log(1.e-4)
        yb = numpy.log(2.e-6)
        x1 = numpy.log(t1)
        x2 = numpy.log(t2)
        y1 = (yb * (x1-xa) + ya * (xb-x1)) / (xb-xa)
        y2 = (yb * (x2-xa) + ya * (xb-x2)) / (xb-xa)
        xi1 = numpy.exp(y1)
        xi2 = numpy.exp(y2)

        sv_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.16, xi2 * 0.16, 0.], 
                           color = '#FFFF82')
        y5_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.03, xi2 * 0.03, 0.], 
                           color = '#BAFFA4')
    #else:
    elif False:
        # Use Alexandre's xi file as a requirement.
        theta = [0.5, 1.49530470404, 2.91321328166, 5.67019971545, 11.0321639144,
                 21.4548924111, 41.6931936543, 80.8508152859, 156.285886576,
                 297.92139021, 300.]

        dxi = [1.4e-5, 3.82239654447e-06, 2.36185315415e-06, 1.26849547074e-06, 6.3282672138e-07,
               3.25623661098e-07, 1.747852053e-07, 8.75326181278e-08, 3.60247306537e-08,
               1.13521735321e-08, 1.125e-8]
        req = [ x / 5.9 for x in dxi ]
        sv_req = plt.fill( [theta[0]] + theta + [theta[-1]], [0.] + req + [0.],
                           color = '#FFFF82')
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    rho1_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if rho3 is not None:
        plt.plot(meanr*1.03, rho3, color='green')
        plt.plot(meanr*1.03, -rho3, color='green', ls=':')
        plt.errorbar(meanr[rho3>0]*1.03, rho3[rho3>0], yerr=sig3[rho3>0]/sqrtn, color='green', ls='', marker='s')
        plt.errorbar(meanr[rho3<0]*1.03, -rho3[rho3<0], yerr=sig3[rho3<0]/sqrtn, color='green', ls='', marker='s')
        rho3_line = plt.errorbar(-meanr, rho3, yerr=sig3, color='green', marker='s')
    if rho4 is not None:
        plt.plot(meanr*1.06, rho4, color='red')
        plt.plot(meanr*1.06, -rho4, color='red', ls=':')
        plt.errorbar(meanr[rho4>0]*1.06, rho4[rho4>0], yerr=sig4[rho4>0]/sqrtn, color='red', ls='', marker='^')
        plt.errorbar(meanr[rho4<0]*1.06, -rho4[rho4<0], yerr=sig4[rho4<0]/sqrtn, color='red', ls='', marker='^')
        rho4_line = plt.errorbar(-meanr, rho4, yerr=sig4, color='red', marker='^')
    #sv_req = mp.Patch(color='#FFFF82')
    if rho3 is not None and rho4 is not None:
        if gband:
            loc = 'upper right'
            fontsize = 18
        else:
            loc = 'upper right'
            fontsize = 18
        plt.legend([rho1_line, rho3_line, rho4_line],
                   [r'$\rho_1(\theta)$', r'$\rho_3(\theta)$', r'$\rho_4(\theta)$'],
                   loc=loc, fontsize=fontsize)
        #plt.ylim( [1.e-9, 5.e-6] )
        #plt.ylim( [1.e-9, 2.e-5] )
        plt.ylim( [1.e-10, 1.e-5] )
    elif True:
        plt.legend([rho1_line, sv_req],
                   [r'$\rho_1(\theta)$', r'Requirement'],
                   loc='upper right')
        plt.ylim( [1.e-9, 5.e-6] )
    else: # For talk
        plt.legend([rho1_line, sv_req],
                   [r'$\rho_1(\theta)$',
                    r'Requirements for $d\sigma_8/\sigma_8 < 0.03$'],
                   loc='upper right')
        plt.ylim( [1.e-9, 3.e-6] )
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=fontsize+2)
    plt.ylabel(r'$\rho(\theta)$', fontsize=fontsize+2)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()

def pretty_rho2(meanr, rho, sig, sqrtn, rho5=None, sig5=None, gband=False):
    import matplotlib.patches as mp
    # The requirements on rho2 are less stringent.  They are larger by a factor 1/alpha.
    # Let's use alpha = 0.03.
    alpha = 0.03
    if False:
        t1 = 0.5
        t2 = 300.
        xa = numpy.log(1.0)
        xb = numpy.log(40.0)
        ya = numpy.log(1.e-4)
        yb = numpy.log(2.e-6)
        x1 = numpy.log(t1)
        x2 = numpy.log(t2)
        y1 = (yb * (x1-xa) + ya * (xb-x1)) / (xb-xa)
        y2 = (yb * (x2-xa) + ya * (xb-x2)) / (xb-xa)
        xi1 = numpy.exp(y1)
        xi2 = numpy.exp(y2)

        sv_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.16 / alpha, xi2 * 0.16 / alpha, 0.], 
                           color = '#FFFF82')
        y5_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.03 / alpha, xi2 * 0.03 / alpha, 0.], 
                           color = '#BAFFA4')

    #else:
    elif False:
        # Use Alexandre's xi file as a requirement.
        theta = [0.5, 1.49530470404, 2.91321328166, 5.67019971545, 11.0321639144,
                 21.4548924111, 41.6931936543, 80.8508152859, 156.285886576,
                 297.92139021, 300.]

        dxi = [1.4e-5, 3.82239654447e-06, 2.36185315415e-06, 1.26849547074e-06, 6.3282672138e-07,
               3.25623661098e-07, 1.747852053e-07, 8.75326181278e-08, 3.60247306537e-08,
               1.13521735321e-08, 1.125e-8]
        req = [ x / (2.4 * alpha) for x in dxi ]
        sv_req = plt.fill( [theta[0]] + theta + [theta[-1]], [0.] + req + [0.],
                           color = '#FFFF82')
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    rho2_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if rho5 is not None:
        plt.plot(meanr*1.03, rho5, color='green')
        plt.plot(meanr*1.03, -rho5, color='green', ls=':')
        plt.errorbar(meanr[rho5>0]*1.03, rho5[rho5>0], yerr=sig5[rho5>0]/sqrtn, color='green', ls='', marker='s')
        plt.errorbar(meanr[rho5<0]*1.03, -rho5[rho5<0], yerr=sig5[rho5<0]/sqrtn, color='green', ls='', marker='s')
        rho5_line = plt.errorbar(-meanr, rho5, yerr=sig5, color='green', marker='s')
    #sv_req = mp.Patch(color='#FFFF82')
    if rho5 is not None:
        if gband:
            loc = 'upper right'
            fontsize = 18
        else:
            loc = 'upper right'
            fontsize = 18
        plt.legend([rho2_line, rho5_line],
                   [r'$\rho_2(\theta)$', r'$\rho_5(\theta)$'],
                   loc=loc, fontsize=fontsize)
        #plt.ylim( [1.e-7, 5.e-4] )
        plt.ylim( [1.e-8, 1.e-5] )
    elif True: # For paper
        plt.legend([rho2_line, sv_req],
                   [r'$\rho_2(\theta)$', r'Requirement'],
                   loc='upper right')
        plt.ylim( [1.e-7, 5.e-4] )
    else: # For talk
        plt.legend([rho2_line, sv_req],
                   [r'$\rho_2(\theta)$',
                    r'Requirements for $d\sigma_8/\sigma_8 < 0.03$'],
                   loc='upper right')
        plt.ylim( [1.e-7, 3.e-4] )
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=fontsize+2)
    plt.ylabel(r'$\rho(\theta)$', fontsize=fontsize+2)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()


def pretty_rho0(meanr, rho, sig, sqrtn):
    import matplotlib.patches as mp
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    rho0_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if True:
        plt.legend([rho0_line],
                   [r'$\rho_0(\theta)$'],
                   loc='upper right', fontsize=18)
        plt.ylim( [1.e-6, 1.e-3] )
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=20)
    plt.ylabel(r'$\rho(\theta)$', fontsize=20)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()

def pretty_tau(meanr, rho, sig, sqrtn, tau2=None, sig2=None, tau5=None, sig5=None, gband=False):
    import matplotlib.patches as mp
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    tau0_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if tau2 is not None:
        plt.plot(meanr*1.03, tau2, color='green')
        plt.plot(meanr*1.03, -tau2, color='green', ls=':')
        plt.errorbar(meanr[tau2>0]*1.03, tau2[tau2>0], yerr=sig2[tau2>0]/sqrtn, color='green', ls='', marker='s')
        plt.errorbar(meanr[tau2<0]*1.03, -tau2[tau2<0], yerr=sig2[tau2<0]/sqrtn, color='green', ls='', marker='s')
        tau2_line = plt.errorbar(-meanr, tau2, yerr=sig2, color='green', marker='s')
    if tau5 is not None:
        plt.plot(meanr*1.06, tau5, color='red')
        plt.plot(meanr*1.06, -tau5, color='red', ls=':')
        plt.errorbar(meanr[tau5>0]*1.06, tau5[tau5>0], yerr=sig5[tau5>0]/sqrtn, color='red', ls='', marker='^')
        plt.errorbar(meanr[tau5<0]*1.06, -tau5[tau5<0], yerr=sig5[tau5<0]/sqrtn, color='red', ls='', marker='^')
        tau5_line = plt.errorbar(-meanr, tau5, yerr=sig5, color='red', marker='^')
    
    if tau2 is not None and tau5 is not None:
        if gband:
            loc = 'upper right'
            fontsize = 18
        else:
            loc = 'upper right'
            fontsize = 18
        plt.legend([tau0_line, tau2_line, tau5_line],
                   [r'$\tau_0(\theta)$', r'$\tau_2(\theta)$', r'$\tau_5(\theta)$'],
                   loc=loc, fontsize=fontsize)
        #plt.ylim( [1.e-9, 5.e-6] )
        #plt.ylim( [1.e-9, 2.e-5] )
        plt.ylim( [1.e-10, 1.e-5] )
    elif True:
        plt.legend([tau0_line],
                   [r'$\tau_0(\theta)$'],
                   loc='upper right', fontsize=18)
        plt.ylim( [1.e-6, 1.e-3] )
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=20)
    plt.ylabel(r'$\tau(\theta)$', fontsize=20)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()


def plot_corr_tt(meanr, corr, sig):
    plt.plot(meanr, corr, color='blue')
    plt.plot(meanr, -corr, color='blue', ls=':')
    plt.errorbar(meanr[corr>0], corr[corr>0], yerr=sig[corr>0], color='blue', ls='', marker='o')
    plt.errorbar(meanr[corr<0], -corr[corr<0], yerr=sig[corr<0], color='blue', ls='', marker='o')
    plt.ylim( [1.e-7, 5.e-5] )
    plt.xlim( [0.5,300.] )
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=24)
    plt.ylabel(r'$\langle (dT_p/T_p) (dT_p/T_p) \rangle (\theta)$', fontsize=24)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()

def plot_overall_rho(work, name, bands):

    base_keys = bands
    keys = [ name+'_' + k for k in base_keys ]

    for key in keys:
        stat_file = os.path.join(work, "rho_" + key + ".json")
        if not os.path.isfile(stat_file):
            print('File not found: ',stat_file,' (skipping)')
            continue

        # Read the json file 
        print('Read %s'%stat_file)
        with open(stat_file,'r') as f:
            stats = json.load(f)

        #print' stats = ',stats
        if len(stats) == 1:  # I used to save a list of length 1 that in turn was a list
            stats = stats[0]

        print('len(stats) = ',len(stats))
        ( meanlogr,
          rho1p,
          rho1p_im,
          rho1m,
          rho1m_im,
          var1p,
          var1m,
          rho2p,
          rho2p_im,
          rho2m,
          rho2m_im,
          var2p,
          var2m,
          rho3p,
          rho3p_im,
          rho3m,
          rho3m_im,
          var3p,
          var3m,
          rho4p,
          rho4p_im,
          rho4m,
          rho4m_im,
          var4p,
          var4m,
          rho5p,
          rho5p_im,
          rho5m,
          rho5m_im,
          var5p,
          var5m,
        ) = stats[:31]

        meanr = np.exp(meanlogr)
        rho1p = np.array(rho1p)
        rho1m = np.array(rho1m)
        rho2p = np.array(rho2p)
        rho2m = np.array(rho2m)
        rho3p = np.array(rho3p)
        rho3m = np.array(rho3m)
        rho4p = np.array(rho4p)
        rho4m = np.array(rho4m)
        rho5p = np.array(rho5p)
        rho5m = np.array(rho5m)
        sig_rho1 = np.sqrt(var1p)
        sig_rho2 = np.sqrt(var2p)
        sig_rho3 = np.sqrt(var3p)
        sig_rho4 = np.sqrt(var4p)
        sig_rho5 = np.sqrt(var5p)
        sqrtn = 1

        cols = [meanr,
                rho1p, rho1m, sig_rho1,
                rho2p, rho2m, sig_rho2,
                rho3p, rho3m, sig_rho3,
                rho4p, rho4m, sig_rho4,
                rho5p, rho5m, sig_rho5]
        header = ('meanr  '+
                  'rho1  rho1_xim  sig_rho1  '+
                  'rho2  rho2_xim  sig_rho2  '+
                  'rho3  rho3_xim  sig_rho3  '+
                  'rho4  rho4_xim  sig_rho4  '+
                  'rho5  rho5_xim  sig_rho5  ')

        if len(stats) > 31:
            ( rho0p,
              rho0p_im,
              rho0m,
              rho0m_im,
              var0p,
              var0m,
            ) = stats[31:37]
            rho0p = np.array(rho0p)
            rho0m = np.array(rho0m)
            sig_rho0 = np.sqrt(var0p)
            cols += [rho0p, rho0m, sig_rho0]
            header += 'rho0  rho0_xim  sig_rho0  '
        if len(stats) > 37:
            ( tau0p,
              tau0p_im,
              tau0m,
              tau0m_im,
              vart0p,
              vart0m,
              tau2p,
              tau2p_im,
              tau2m,
              tau2m_im,
              vart2p,
              vart2m,
              tau5p,
              tau5p_im,
              tau5m,
              tau5m_im,
              vart5p,
              vart5m,
            ) = stats[37:55]
            tau0p = np.array(tau0p)
            tau0m = np.array(tau0m)
            tau2p = np.array(tau2p)
            tau2m = np.array(tau2m)
            tau5p = np.array(tau5p)
            tau5m = np.array(tau5m)
            sig_tau0 = np.sqrt(vart0p)
            sig_tau2 = np.sqrt(vart2p)
            sig_tau5 = np.sqrt(vart5p)
            cols += [tau0p, tau0m, sig_tau0, 
                     tau2p, tau2m, sig_tau2, 
                     tau5p, tau5m, sig_tau5,]
            header += ('tau0  tau0_xim  sig_tau0  '+
                       'tau2  tau2_xim  sig_tau2  '+
                       'tau5  tau5_xim  sig_tau5  ')

        outfile = os.path.join(work, 'rho_' + key + '.dat')
        np.savetxt(outfile, np.array(cols), fmt='%.6e', header=header)
        print('wrote',outfile)
 
        plt.clf()
        pretty_rho1(meanr, rho1p, sig_rho1, sqrtn, rho3p, sig_rho3, rho4p, sig_rho4,
                    gband=(key.endswith('g')))
        figfile = os.path.join(work,'rho1_' + key + '.pdf') 
        plt.savefig(figfile)

        plt.clf()
        pretty_rho2(meanr, rho2p, sig_rho2, sqrtn, rho5p, sig_rho5,
                    gband=(key.endswith('g')))
        figfile = os.path.join(work,'rho2_' + key + '.pdf') 
        plt.savefig(figfile)

        if len(stats) > 31:
            plt.clf()
            pretty_rho0(meanr, rho0p, sig_rho0, sqrtn)
            figfile = os.path.join(work,'rho0_' + key + '.pdf')
            plt.savefig(figfile)
        
        if len(stats) > 37:
            plt.clf()
            pretty_tau(meanr, tau0p, sig_rho0, sqrtn, tau2p, sig_tau2, tau5p, sig_tau5)
            figfile = os.path.join(work,'tau_' + key + '.pdf')
            plt.savefig(figfile)

        if len(stats) > 55:
            corr_tt, var_corr_tt = stats[37:39]
            corr_tt = np.array(corr_tt)
            sig_corr_tt = np.sqrt(var_corr_tt)

            plt.clf()
            plot_corr_tt(meanr, corr_tt, sig_corr_tt)
            figfile = os.path.join(work,'corrtt_' + key + '.pdf')
            plt.savefig(figfile)

def plot_overall_tau(work, name, bands, ver):

    # base_keys = ['r', 'i', 'z']
    base_keys = bands
    keys = [ name+'_' + k for k in base_keys ]
    print(keys)
    for key in keys:
        stat_file = os.path.join(work, "tau_%s_%i.json"%(key,ver))
        if not os.path.isfile(stat_file):
            print('File not found: ',stat_file,' (skipping)')
            continue

        # Read the json file 
        print('Read %s'%stat_file)
        with open(stat_file,'r') as f:
            stats = json.load(f)

        #print' stats = ',stats
        if len(stats) == 1:  # I used to save a list of length 1 that in turn was a list
            stats = stats[0]

        print('len(stats) = ',len(stats))
        ( meanlogr,
          tau0p,
          tau0p_im,
          tau0m,
          tau0m_im,
          vart0p,
          vart0m,
          tau2p,
          tau2p_im,
          tau2m,
          tau2m_im,
          vart2p,
          vart2m,
          tau5p,
          tau5p_im,
          tau5m,
          tau5m_im,
          vart5p,
          vart5m,
        ) = stats[:19]
        meanr = np.exp(meanlogr)
        tau0p = np.array(tau0p)
        tau0m = np.array(tau0m)
        tau2p = np.array(tau2p)
        tau2m = np.array(tau2m)
        tau5p = np.array(tau5p)
        tau5m = np.array(tau5m)
        sig_tau0 = np.sqrt(vart0p)
        sig_tau2 = np.sqrt(vart2p)
        sig_tau5 = np.sqrt(vart5p)
        sqrtn = 1
        cols = [meanr, 
                tau0p, tau0m, sig_tau0, 
                tau2p, tau2m, sig_tau2, 
                tau5p, tau5m, sig_tau5]
        header = ('meanr  '+
                  'tau0  tau0_xim  sig_tau0  '+
                  'tau2  tau2_xim  sig_tau2  '+
                  'tau5  tau5_xim  sig_tau5  ')

        outfile = os.path.join(work, 'tau_%s_%i.dat'%(key,ver))
        np.savetxt(outfile, np.array(cols), fmt='%.6e', header=header)
        print('wrote',outfile)
        
        plt.clf()
        pretty_tau(meanr, tau0p, sig_tau0, sqrtn, tau2p, sig_tau2, tau5p, sig_tau5)
        pdffile = os.path.join(work, 'tau_%s_%i.pdf'%(key,ver))
        plt.savefig(pdffile)
        print('wrote',pdffile)
