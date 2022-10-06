select
    star.expnum,
    star.ccdnum,
    -- e.band,
    -- e.mjd_obs,
    model.nstar,
    model.nremoved,
    model.exp_star_t_mean,
    model.exp_star_t_std,
    model.star_t_mean,
    model.star_t_std,
    model.fwhm_cen,
    -- qa.psf_fwhm,
    star.coadd_object_id,
    star.ra,
    star.dec,
    star.u,
    star.v,
    star.x,
    star.y,
    star.snr,
    star.flux,
    star.bdf_t,
    star.t_data,
    star.t_model,
    star.g1_data,
    star.g1_model,
    star.g2_data,
    star.g2_model,
    star.gi_color,
    star.iz_color,
    star.g_mag,
    star.r_mag,
    star.i_mag,
    star.z_mag,
    star.k_mag,
    -- star.is_reserve,
    star.flag_color,
    star.gaia_star,
    star.gaia_source_id,
    star.vhs_obj,
    star.vhs_sourceid
from PROD.PIFF_HSM_STAR_QA star
    join proctag t on star.pfw_attempt_id=t.pfw_attempt_id
    -- join exposure e on star.expnum=e.expnum
    join PROD.PIFF_HSM_MODEL_QA model on star.model_filename=model.filename
    -- join QA_SUMMARY qa on star.expnum=qa.expnum
where
    t.tag = 'Y6A2_PIFF_V3'
    -- and e.program ='supernova'
    and model.flag = 0
    and model.nstar >= 30
    and star.is_reserve = 1
    -- and e.band != 'Y'
    and star.flag_model = 0
    and star.flag_truth = 0;
