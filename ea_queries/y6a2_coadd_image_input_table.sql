with x as (
     select /*+materialize */ exp.expnum, exp.mjd_obs
            from Y6A2_EXPOSURE exp
            )
select i.EXPNUM, i.CCDNUM, i.SKYVARA, i.SKYVARB, i.BAND,
       i.SKYSIGMA, i.SKYBRITE, i.FWHM, i.PSF_FWHM, i.PSF_BETA, i.PSFSCALE,
       i.AIRMASS, i.EXPTIME,
       z.mag_zero,
       x.mjd_obs,
       i.TILENAME, i.PFW_ATTEMPT_ID,
       i.RAC1, i.RAC2, i.RAC3, i.RAC4, i.RA_CENT,
       i.DECC1, i.DECC2, i.DECC3, i.DECC4, i.DEC_CENT,
       i.CRPIX1, i.CRPIX2, i.CRVAL1, i.CRVAL2, i.CUNIT1, i.CUNIT2,
       i.CD1_1, i.CD1_2, i.CD2_1, i.CD2_2,
       i.PV1_0, i.PV1_1, i.PV1_2, i.PV1_3, i.PV1_4, i.PV1_5, i.PV1_6, i.PV1_7, i.PV1_8, i.PV1_9, i.PV1_10,
       i.PV2_0, i.PV2_1, i.PV2_2, i.PV2_3, i.PV2_4, i.PV2_5, i.PV2_6, i.PV2_7, i.PV2_8, i.PV2_9, i.PV2_10,
       i.NAXIS1, i.NAXIS2
from Y6A2_IMAGE i, Y6A2_ZEROPOINT z, x
where FILETYPE='coadd_nwgint' and
      i.expnum=z.expnum and
      i.ccdnum=z.ccdnum and
      i.expnum=x.expnum and
      z.source='FGCM' and
      z.version='y6a1_v2.1';
